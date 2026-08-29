import Foundation

/// Validating reader for q4g64 packed checkpoints — the load side of the
/// schema pin (phase-3.md D1). P3-1 scope: open, validate, and CPU-dequant
/// for the fixture tests; the fp32 materialization front end that feeds the
/// frozen CPU model is P3-2.
///
/// Validation at load (never assumed): format tag + group size, provenance
/// revision, triplet completeness and shape consistency, and 4-byte
/// alignment of every `.q` tensor's absolute file offset (typed u32 loads
/// in the P3-4 kernels are only legal on aligned offsets).
public final class PackedCheckpoint {
    public let path: String
    /// The underlying parsed file (pass-through tensors read via
    /// `file.fp32Values`; GPU residency reuses its mapping in P3-4/P3-5).
    public let file: SafetensorsFile
    public let sourceRevision: String
    /// Base names (suffix stripped) of the packed matrices, sorted.
    public let matrixNames: [String]
    /// Names of the unquantized 1-D pass-through tensors (norms), sorted.
    public let passthroughNames: [String]

    public struct MatrixDims {
        public let outDim: Int
        public let inDim: Int
    }
    private let matrixDims: [String: MatrixDims]

    public init(path: String, expectedRevision: String? = nil) throws {
        self.path = path
        let file = try SafetensorsFile(path: path)

        guard file.metadata[Q4G64.formatMetadataKey] == Q4G64.formatTag else {
            let found = file.metadata[Q4G64.formatMetadataKey] ?? "<absent>"
            throw Q4G64Error.notAPackedCheckpoint(
                path: path,
                detail: "metadata '\(Q4G64.formatMetadataKey)' is '\(found)', "
                    + "expected '\(Q4G64.formatTag)' — bf16/f16 checkpoints "
                    + "load via QwenModel/GPUModel, not the packed loader")
        }
        let groupSize = file.metadata[Q4G64.groupSizeMetadataKey] ?? "<absent>"
        guard groupSize == "\(Q4G64.groupSize)" else {
            throw Q4G64Error.notAPackedCheckpoint(
                path: path,
                detail: "metadata '\(Q4G64.groupSizeMetadataKey)' is "
                    + "'\(groupSize)', expected '\(Q4G64.groupSize)'")
        }
        guard let revision = file.metadata[Q4G64.sourceRevisionMetadataKey] else {
            throw Q4G64Error.missingSourceRevision(path: path)
        }
        if let expected = expectedRevision, revision != expected {
            throw Q4G64Error.revisionMismatch(
                path: path, found: revision, expected: expected)
        }

        var dims: [String: MatrixDims] = [:]
        var claimed = Set<String>()
        for name in file.tensorNames where name.hasSuffix(Q4G64.qSuffix) {
            let base = String(name.dropLast(Q4G64.qSuffix.count))
            let q = try file.info(for: name)
            guard q.dtype == .uint32 else {
                throw Q4G64Error.badTriplet(
                    name: base, detail: "'\(name)' has dtype "
                        + "\(q.dtype.rawValue), expected U32")
            }
            guard q.shape.count == 2, q.shape[0] > 0, q.shape[1] > 0 else {
                throw Q4G64Error.badTriplet(
                    name: base, detail: "'\(name)' shape \(q.shape) is not "
                        + "a 2-D [out, in/8] code matrix")
            }
            let scalesName = base + Q4G64.scalesSuffix
            let biasesName = base + Q4G64.biasesSuffix
            guard let scales = try? file.info(for: scalesName),
                  let biases = try? file.info(for: biasesName) else {
                throw Q4G64Error.badTriplet(
                    name: base, detail: "missing '\(scalesName)' or '\(biasesName)'")
            }
            guard scales.dtype == .float16, biases.dtype == .float16 else {
                throw Q4G64Error.badTriplet(
                    name: base, detail: "scales/biases dtypes are "
                        + "\(scales.dtype.rawValue)/\(biases.dtype.rawValue), expected F16/F16")
            }
            let outDim = q.shape[0]
            let words = q.shape[1]
            let inDim = words * Q4G64.codesPerWord
            let expectedGroupShape = [outDim, inDim / Q4G64.groupSize]
            guard inDim % Q4G64.groupSize == 0,
                  scales.shape == expectedGroupShape,
                  biases.shape == expectedGroupShape else {
                throw Q4G64Error.badTriplet(
                    name: base,
                    detail: "q shape \(q.shape) (in-dim \(inDim)) is "
                        + "inconsistent with scales \(scales.shape) / "
                        + "biases \(biases.shape) at group size \(Q4G64.groupSize)")
            }
            let absoluteOffset = file.dataSectionStart + q.dataOffset
            guard absoluteOffset % 4 == 0 else {
                throw Q4G64Error.misalignedQTensor(
                    name: name, absoluteByteOffset: absoluteOffset)
            }
            dims[base] = MatrixDims(outDim: outDim, inDim: inDim)
            claimed.formUnion([name, scalesName, biasesName])
        }

        var passthrough: [String] = []
        for name in file.tensorNames where !claimed.contains(name) {
            if name.hasSuffix(Q4G64.scalesSuffix) || name.hasSuffix(Q4G64.biasesSuffix) {
                throw Q4G64Error.badTriplet(
                    name: name, detail: "stray scales/biases tensor with no "
                        + "matching '\(Q4G64.qSuffix)' tensor")
            }
            let info = try file.info(for: name)
            guard info.shape.count == 1, info.dtype != .uint32 else {
                throw Q4G64Error.badTriplet(
                    name: name, detail: "unexpected non-triplet tensor "
                        + "(dtype \(info.dtype.rawValue), shape \(info.shape)) — "
                        + "pass-throughs are 1-D float vectors")
            }
            passthrough.append(name)
        }

        self.file = file
        self.sourceRevision = revision
        self.matrixDims = dims
        self.matrixNames = dims.keys.sorted()
        self.passthroughNames = passthrough.sorted()
    }

    public func dims(for baseName: String) throws -> MatrixDims {
        guard let dims = matrixDims[baseName] else {
            throw SafetensorsError.tensorNotFound(name: baseName + Q4G64.qSuffix)
        }
        return dims
    }

    /// CPU dequant of one packed matrix to row-major fp32 `[out, in]` —
    /// exactly the arithmetic the GPU kernels reproduce in registers
    /// (`Q4G64.dequant`; single-rounding argument in the gates entry).
    ///
    /// Bulk-loop shape (P3-2): the 16 possible dequant values of a group are
    /// computed ONCE via the pinned `Q4G64.dequant` and applied by table
    /// lookup — bit-identical to calling it per element (same inputs, same
    /// single-rounded operation), but far faster in debug builds, where the
    /// per-element scalar loop cost ~230 s across the 1.7B checkpoint's 197
    /// matrices (IO-1 precedent; exactness stays pinned by the P3-1/P3-2
    /// `==` tests).
    public func dequantMatrix(_ baseName: String) throws -> [Float] {
        let dims = try dims(for: baseName)
        let qInfo = try file.info(for: baseName + Q4G64.qSuffix)
        let scalesInfo = try file.info(for: baseName + Q4G64.scalesSuffix)
        let biasesInfo = try file.info(for: baseName + Q4G64.biasesSuffix)

        let dataBase = file.mappedBaseAddress + file.dataSectionStart
        let qPtr = dataBase + qInfo.dataOffset
        let scalesPtr = dataBase + scalesInfo.dataOffset
        let biasesPtr = dataBase + biasesInfo.dataOffset

        let totalGroups = dims.outDim * (dims.inDim / Q4G64.groupSize)
        var out = [Float](repeating: 0, count: dims.outDim * dims.inDim)
        out.withUnsafeMutableBufferPointer { dst in
            // Chunks own disjoint group ranges, and groups tile the
            // row-major element space contiguously (in-dim divides by 64),
            // so every element is written exactly once by the same pinned
            // arithmetic regardless of scheduling — parallelism cannot
            // change a single output bit.
            let chunkCount = min(
                totalGroups, max(1, ProcessInfo.processInfo.activeProcessorCount))
            DispatchQueue.concurrentPerform(iterations: chunkCount) { chunk in
                let chunkStart = totalGroups * chunk / chunkCount
                let chunkEnd = totalGroups * (chunk + 1) / chunkCount
                withUnsafeTemporaryAllocation(
                    of: Float.self, capacity: 16
                ) { lutBuf in
                    for groupIndex in chunkStart..<chunkEnd {
                        let scale = Float16(bitPattern: UInt16(littleEndian:
                            scalesPtr.loadUnaligned(
                                fromByteOffset: 2 * groupIndex, as: UInt16.self)))
                        let bias = Float16(bitPattern: UInt16(littleEndian:
                            biasesPtr.loadUnaligned(
                                fromByteOffset: 2 * groupIndex, as: UInt16.self)))
                        for code in 0..<16 {
                            lutBuf[code] = Q4G64.dequant(
                                code: UInt32(code), scale: scale, bias: bias)
                        }
                        let wordBase = groupIndex * Q4G64.wordsPerGroup
                        var element = groupIndex * Q4G64.groupSize
                        for w in 0..<Q4G64.wordsPerGroup {
                            // Low nibble first (schema pin, == Q4G64.code).
                            var word = UInt32(littleEndian: qPtr.loadUnaligned(
                                fromByteOffset: 4 * (wordBase + w), as: UInt32.self))
                            for _ in 0..<Q4G64.codesPerWord {
                                dst[element] = lutBuf[Int(word & 0xF)]
                                word >>= 4
                                element += 1
                            }
                        }
                    }
                }
            }
        }
        return out
    }
}
