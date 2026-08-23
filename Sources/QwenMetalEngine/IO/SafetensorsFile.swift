import Accelerate
import Foundation

/// Errors thrown by `SafetensorsFile`. Each case carries enough context to
/// diagnose a bad checkpoint from the message alone — phase-0-1.md requires
/// clear errors, never crashes, for every enumerated failure mode.
public enum SafetensorsError: Error, CustomStringConvertible {
    case cannotOpen(path: String, reason: String)
    case shardedCheckpoint(path: String, indexFile: String)
    case truncated(path: String, detail: String)
    case malformedHeader(path: String, detail: String)
    case invalidTensor(name: String, detail: String)
    case unsupportedDtype(name: String, dtype: String)
    case tensorNotFound(name: String)

    public var description: String {
        switch self {
        case .cannotOpen(let path, let reason):
            return "safetensors: cannot open \(path): \(reason)"
        case .shardedCheckpoint(let path, let indexFile):
            return "safetensors: \(path) belongs to a SHARDED checkpoint "
                + "(\(indexFile) found alongside it). This engine reads "
                + "single-file checkpoints only — consolidate first with "
                + "tools/consolidate_shards.py (see DECISIONS.md PIN-1 entry)."
        case .truncated(let path, let detail):
            return "safetensors: \(path) is truncated: \(detail)"
        case .malformedHeader(let path, let detail):
            return "safetensors: \(path) has a malformed JSON header: \(detail)"
        case .invalidTensor(let name, let detail):
            return "safetensors: tensor '\(name)' is invalid: \(detail)"
        case .unsupportedDtype(let name, let dtype):
            return "safetensors: tensor '\(name)' has unsupported dtype "
                + "'\(dtype)' (Phase 1 reads F16/BF16 checkpoints only)"
        case .tensorNotFound(let name):
            return "safetensors: no tensor named '\(name)' in this file"
        }
    }
}

/// Source dtypes the Phase 1 engine accepts (phase-0-1.md build step 1).
/// Both upcast to fp32 exactly: bf16 by bit shift, fp16 by construction.
public enum TensorDType: String {
    case float16 = "F16"
    case bfloat16 = "BF16"

    /// Both supported dtypes are 16-bit.
    var byteWidth: Int { 2 }
}

/// One tensor's header entry: dtype, shape, and where its raw little-endian
/// payload lives inside the file's data section.
public struct TensorInfo {
    public let name: String
    public let dtype: TensorDType
    public let shape: [Int]
    public let elementCount: Int
    /// Byte range within the data section (not the file).
    let dataOffset: Int
    let byteCount: Int
}

/// Hand-written, mmap-based safetensors reader.
///
/// Pins that shape this type (changing them needs a DECISIONS.md entry):
/// - Hand-written, not swift-safetensors (DECISIONS.md 2026-08-20 item 1):
///   raw mmap control matters for iOS memory accounting from Phase 2 on.
/// - Weights load via mmap, never heap reads (PLAN.md invariant 5). The file
///   stays mapped for this object's lifetime; `fp32Values` materializes an
///   fp32 copy per tensor because both source dtypes upcast anyway (~7 GB
///   resident on the Mac during Phase 1 — accepted in phase-0-1.md).
/// - Single-file only: a sharded checkpoint (any `*.index.json` next to the
///   file) is rejected loudly; consolidation is an offline tools/ step.
public final class SafetensorsFile {
    public let path: String
    /// Sorted for deterministic iteration.
    public let tensorNames: [String]
    /// The optional `__metadata__` block (string-to-string per the format).
    public let metadata: [String: String]

    private let base: UnsafeMutableRawPointer
    private let fileSize: Int
    private let dataSectionOffset: Int
    private let tensors: [String: TensorInfo]

    public init(path: String) throws {
        self.path = path
        try Self.rejectShardedCheckpoint(at: path)

        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw SafetensorsError.cannotOpen(
                path: path, reason: String(cString: strerror(errno)))
        }
        defer { close(fd) }

        var st = stat()
        guard fstat(fd, &st) == 0 else {
            throw SafetensorsError.cannotOpen(
                path: path, reason: String(cString: strerror(errno)))
        }
        let fileSize = Int(st.st_size)
        guard fileSize >= 8 else {
            throw SafetensorsError.truncated(
                path: path,
                detail: "file is \(fileSize) bytes; the 8-byte header-length prefix does not fit")
        }

        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != MAP_FAILED else {
            throw SafetensorsError.cannotOpen(
                path: path, reason: "mmap failed: \(String(cString: strerror(errno)))")
        }

        // A class deinit does not run if init throws, so the mapping must be
        // released explicitly on every parse-failure path.
        do {
            let parsed = try Self.parseHeader(base: mapped, fileSize: fileSize, path: path)
            self.base = mapped
            self.fileSize = fileSize
            self.dataSectionOffset = parsed.dataSectionOffset
            self.tensors = parsed.tensors
            self.tensorNames = parsed.tensors.keys.sorted()
            self.metadata = parsed.metadata
        } catch {
            munmap(mapped, fileSize)
            throw error
        }
    }

    deinit {
        munmap(base, fileSize)
    }

    // MARK: - Raw mapping access (same-module only)

    /// The live mmap, exposed for `GPUWeights` (P2-1): the no-copy MTLBuffer
    /// aliases these pages directly. The mapping stays valid for this
    /// object's lifetime — holders must retain the file (invariant 5: the
    /// bytes are never copied onto the heap on the mmap path).
    var mappedBaseAddress: UnsafeMutableRawPointer { base }
    var mappedFileSize: Int { fileSize }
    /// Absolute file offset where the tensor data section begins; a tensor's
    /// absolute offset is this plus its `TensorInfo.dataOffset`.
    var dataSectionStart: Int { dataSectionOffset }

    // MARK: - Tensor access

    public func info(for name: String) throws -> TensorInfo {
        guard let info = tensors[name] else {
            throw SafetensorsError.tensorNotFound(name: name)
        }
        return info
    }

    /// Materializes the tensor as fp32, reading straight from the mapping.
    /// Both upcasts are exact, so tests may compare results with == and the
    /// engine's only legitimate divergence from the oracle stays summation
    /// order (phase-0-1.md correctness harness).
    ///
    /// The conversion runs through Accelerate (IO-1): the debug-config dev
    /// loop pays this materialization on every test run, and a scalar Swift
    /// loop at -Onone cost ~190s for the 1.7B checkpoint. Accelerate is
    /// prebuilt, so its speed does not depend on this package's build config.
    /// data_offsets carry no alignment guarantee, so a source that is not
    /// 2-byte aligned takes the scalar path instead (exactness identical).
    public func fp32Values(for name: String) throws -> [Float] {
        let info = try self.info(for: name)
        let src = base + dataSectionOffset + info.dataOffset
        let count = info.elementCount
        var out = [Float](repeating: 0, count: count)
        guard count > 0 else { return out }
        out.withUnsafeMutableBufferPointer { dst in
            if Int(bitPattern: src) % MemoryLayout<UInt16>.alignment == 0 {
                Self.vectorUpcast(src: src, dtype: info.dtype, into: dst)
            } else {
                Self.scalarUpcast(src: src, dtype: info.dtype, into: dst)
            }
        }
        return out
    }

    /// Elements converted per Accelerate call; bounds the bf16 temp buffer
    /// (4 MiB of fp32 at this size) independent of tensor size.
    private static let upcastChunkElements = 1 << 20

    private static func vectorUpcast(
        src: UnsafeMutableRawPointer, dtype: TensorDType, into dst: UnsafeMutableBufferPointer<Float>
    ) {
        switch dtype {
        case .float16: vectorUpcastF16(src: src, into: dst)
        case .bfloat16: vectorUpcastBF16(src: src, into: dst)
        }
    }

    /// Hardware widening conversion; exact for every fp16 pattern.
    private static func vectorUpcastF16(
        src: UnsafeMutableRawPointer, into dst: UnsafeMutableBufferPointer<Float>
    ) {
        let count = dst.count
        var chunkStart = 0
        while chunkStart < count {
            let n = min(upcastChunkElements, count - chunkStart)
            var srcBuf = vImage_Buffer(
                data: src + 2 * chunkStart,
                height: 1, width: vImagePixelCount(n), rowBytes: 2 * n)
            var dstBuf = vImage_Buffer(
                data: dst.baseAddress! + chunkStart,
                height: 1, width: vImagePixelCount(n), rowBytes: 4 * n)
            guard vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags))
                == kvImageNoError else {
                // Cannot happen with well-formed buffers; keep correctness
                // over speed rather than trusting that.
                scalarUpcast(src: src, dtype: .float16, into: dst)
                return
            }
            chunkStart += n
        }
    }

    /// bf16's fp32 bit pattern is bits << 16. vDSP has no integer shift, but
    /// the chain below is exact at every step: u16 -> fp32 value (all u16 are
    /// representable), * 2^16 (power-of-two scale of a <=16-significant-bit
    /// integer), truncate back to u32 (the value is already integral) —
    /// yielding exactly bits << 16, written into the fp32 destination
    /// reinterpreted as u32.
    private static func vectorUpcastBF16(
        src: UnsafeMutableRawPointer, into dst: UnsafeMutableBufferPointer<Float>
    ) {
        let count = dst.count
        let srcU16 = src.bindMemory(to: UInt16.self, capacity: count)
        var scale = Float(65536)
        var tmp = [Float](repeating: 0, count: min(upcastChunkElements, count))
        tmp.withUnsafeMutableBufferPointer { tmpBuf in
            // Scoped rebind: the destination's Float memory is viewed as
            // UInt32 only while the bit patterns are written.
            dst.withMemoryRebound(to: UInt32.self) { dstU32 in
                let tmpPtr = tmpBuf.baseAddress!
                let dstPtr = dstU32.baseAddress!
                var chunkStart = 0
                while chunkStart < count {
                    let n = vDSP_Length(min(upcastChunkElements, count - chunkStart))
                    vDSP_vfltu16(srcU16 + chunkStart, 1, tmpPtr, 1, n)
                    vDSP_vsmul(tmpPtr, 1, &scale, tmpPtr, 1, n)
                    vDSP_vfixu32(tmpPtr, 1, dstPtr + chunkStart, 1, n)
                    chunkStart += Int(n)
                }
            }
        }
    }

    private static func scalarUpcast(
        src: UnsafeMutableRawPointer, dtype: TensorDType, into dst: UnsafeMutableBufferPointer<Float>
    ) {
        switch dtype {
        case .bfloat16:
            for i in 0..<dst.count {
                let bits = UInt16(littleEndian:
                    src.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self))
                // bf16 is the top half of the fp32 bit pattern: exact.
                dst[i] = Float(bitPattern: UInt32(bits) << 16)
            }
        case .float16:
            for i in 0..<dst.count {
                let bits = UInt16(littleEndian:
                    src.loadUnaligned(fromByteOffset: 2 * i, as: UInt16.self))
                dst[i] = Float(Float16(bitPattern: bits))
            }
        }
    }

    // MARK: - Parsing

    private struct ParsedHeader {
        let dataSectionOffset: Int
        let tensors: [String: TensorInfo]
        let metadata: [String: String]
    }

    private static func rejectShardedCheckpoint(at path: String) throws {
        let dir = (path as NSString).deletingLastPathComponent
        let dirPath = dir.isEmpty ? "." : dir
        guard let entries = try? FileManager.default.contentsOfDirectory(atPath: dirPath) else {
            return // unreadable directory surfaces as cannotOpen below
        }
        if let indexFile = entries.filter({ $0.hasSuffix(".index.json") }).sorted().first {
            throw SafetensorsError.shardedCheckpoint(path: path, indexFile: indexFile)
        }
    }

    private static func parseHeader(
        base: UnsafeMutableRawPointer, fileSize: Int, path: String
    ) throws -> ParsedHeader {
        let rawLength = UInt64(littleEndian: base.loadUnaligned(fromByteOffset: 0, as: UInt64.self))
        guard rawLength <= UInt64(fileSize - 8) else {
            throw SafetensorsError.truncated(
                path: path,
                detail: "header claims \(rawLength) bytes but only \(fileSize - 8) follow the length prefix")
        }
        let headerLength = Int(rawLength)

        let json: Any
        do {
            json = try JSONSerialization.jsonObject(
                with: Data(bytes: base + 8, count: headerLength))
        } catch {
            throw SafetensorsError.malformedHeader(path: path, detail: "\(error)")
        }
        guard let dict = json as? [String: Any] else {
            throw SafetensorsError.malformedHeader(
                path: path, detail: "top-level JSON value is not an object")
        }

        let dataSectionOffset = 8 + headerLength
        let dataSectionSize = fileSize - dataSectionOffset

        var metadata: [String: String] = [:]
        var tensors: [String: TensorInfo] = [:]
        for (name, value) in dict {
            if name == "__metadata__" {
                metadata = value as? [String: String] ?? [:]
                continue
            }
            guard let entry = value as? [String: Any] else {
                throw SafetensorsError.invalidTensor(
                    name: name, detail: "header entry is not an object")
            }
            guard let dtypeString = entry["dtype"] as? String else {
                throw SafetensorsError.invalidTensor(name: name, detail: "missing dtype")
            }
            guard let dtype = TensorDType(rawValue: dtypeString) else {
                throw SafetensorsError.unsupportedDtype(name: name, dtype: dtypeString)
            }
            guard let shapeNumbers = entry["shape"] as? [NSNumber] else {
                throw SafetensorsError.invalidTensor(
                    name: name, detail: "missing or non-array shape")
            }
            let shape = shapeNumbers.map(\.intValue)
            guard shape.allSatisfy({ $0 >= 0 }) else {
                throw SafetensorsError.invalidTensor(
                    name: name, detail: "negative dimension in shape \(shape)")
            }
            var elementCount = 1
            for dim in shape {
                let (product, overflow) = elementCount.multipliedReportingOverflow(by: dim)
                guard !overflow else {
                    throw SafetensorsError.invalidTensor(
                        name: name, detail: "shape \(shape) overflows the element count")
                }
                elementCount = product
            }

            guard let offsets = entry["data_offsets"] as? [NSNumber], offsets.count == 2 else {
                throw SafetensorsError.invalidTensor(
                    name: name, detail: "missing or malformed data_offsets")
            }
            let begin = offsets[0].intValue
            let end = offsets[1].intValue
            guard begin >= 0, begin <= end, end <= dataSectionSize else {
                throw SafetensorsError.invalidTensor(
                    name: name,
                    detail: "data_offsets [\(begin), \(end)) fall outside the "
                        + "\(dataSectionSize)-byte data section")
            }
            let (expectedBytes, overflowed) =
                elementCount.multipliedReportingOverflow(by: dtype.byteWidth)
            guard !overflowed, end - begin == expectedBytes else {
                throw SafetensorsError.invalidTensor(
                    name: name,
                    detail: "data_offsets span \(end - begin) bytes but dtype \(dtypeString) "
                        + "with shape \(shape) needs \(expectedBytes)")
            }
            tensors[name] = TensorInfo(
                name: name, dtype: dtype, shape: shape, elementCount: elementCount,
                dataOffset: begin, byteCount: expectedBytes)
        }

        // Enumerated edge case 5: overlapping ranges are corruption, not slack.
        let byOffset = tensors.values.sorted { $0.dataOffset < $1.dataOffset }
        for (earlier, later) in zip(byOffset, byOffset.dropFirst()) {
            if later.dataOffset < earlier.dataOffset + earlier.byteCount {
                throw SafetensorsError.invalidTensor(
                    name: later.name,
                    detail: "data range overlaps tensor '\(earlier.name)'")
            }
        }

        return ParsedHeader(
            dataSectionOffset: dataSectionOffset, tensors: tensors, metadata: metadata)
    }
}
