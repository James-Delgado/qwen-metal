import Foundation

/// Offline packer: bf16 consolidated checkpoint → q4g64 packed checkpoint
/// (phase-3.md D2). Engine code, not tools/ Python, so the parser, the
/// exact bf16→fp32 upcast, and the dequant arithmetic are shared with the
/// loaders it feeds.
///
/// Output is deterministic byte-for-byte across runs (edge case 5): the
/// JSON header is hand-built with sorted keys (JSONSerialization's key
/// order is not stable), tensors are laid out in a fixed sorted order, and
/// the recipe has no hidden state. Layout puts every `.q` payload first —
/// u32 payload sizes are multiples of 4 and the data section starts 8-byte
/// aligned, so `.q` offsets are structurally ≡ 0 (mod 4); D1 says asserted,
/// not assumed, so the packer still validates before writing and the
/// loader re-validates.
public enum Q4Packer {

    public struct Summary {
        public let packedMatrices: Int
        public let passthroughTensors: Int
        public let outputByteCount: Int
        public let sourceRevision: String
        /// True when the source carried a materialized `lm_head.weight`
        /// byte-identical to the embedding and the packer stored the tied
        /// matrix once (schema D1); the pinned checkpoint does.
        public let tiedLmHeadOmitted: Bool
    }

    /// Tensor names of the tied pair (the engine is pinned single-family).
    private static let embeddingName = "model.embed_tokens.weight"
    private static let lmHeadName = "lm_head.weight"

    private struct MatrixPlan {
        let name: String
        let outDim: Int
        let inDim: Int
    }

    /// Progress callback: (tensor name, 1-based index, total source tensors).
    public typealias Progress = (String, Int, Int) -> Void

    public static func pack(
        inputPath: String, outputPath: String, progress: Progress? = nil
    ) throws -> Summary {
        let source = try SafetensorsFile(path: inputPath)
        guard source.metadata[Q4G64.formatMetadataKey] != Q4G64.formatTag else {
            throw Q4G64Error.inputAlreadyPacked(path: inputPath)
        }
        guard let revision = source.metadata[Q4G64.sourceRevisionMetadataKey] else {
            throw Q4G64Error.missingSourceRevision(path: inputPath)
        }

        // Classify: 2-D matrices are packed, 1-D vectors pass through raw
        // (norms stay bf16 per D1); anything else is not a Qwen3 checkpoint.
        // Schema D1: the tied embedding is stored ONCE. The pinned
        // consolidated checkpoint materializes lm_head.weight byte-identical
        // to the embedding; packing it again would waste ~0.17 GB in the
        // artifact for a tensor no loader reads. Omit it only on verified
        // byte identity — an untied lm_head still packs normally.
        let tiedLmHeadOmitted = tiedLmHeadIsByteIdentical(in: source)

        var matrices: [MatrixPlan] = []
        var passthroughs: [TensorInfo] = []
        for name in source.tensorNames { // sorted — deterministic
            if tiedLmHeadOmitted && name == lmHeadName { continue }
            try validateHeaderSafe(name)
            let info = try source.info(for: name)
            switch info.shape.count {
            case 2:
                let outDim = info.shape[0]
                let inDim = info.shape[1]
                guard outDim > 0, inDim > 0, inDim % Q4G64.groupSize == 0 else {
                    throw Q4G64Error.inDimNotMultipleOfGroup(
                        tensor: name, shape: info.shape)
                }
                matrices.append(MatrixPlan(name: name, outDim: outDim, inDim: inDim))
            case 1 where info.elementCount > 0:
                passthroughs.append(info)
            default:
                throw Q4G64Error.unsupportedRank(tensor: name, shape: info.shape)
            }
        }

        // Output layout: all .q tensors (source-sorted), then per-matrix
        // .scales/.biases, then pass-throughs. Offsets are assigned in this
        // order with no gaps.
        struct OutputSpec {
            let name: String
            let dtype: String
            let shape: [Int]
            let byteCount: Int
        }
        var specs: [OutputSpec] = []
        for m in matrices {
            specs.append(OutputSpec(
                name: m.name + Q4G64.qSuffix, dtype: TensorDType.uint32.rawValue,
                shape: [m.outDim, m.inDim / Q4G64.codesPerWord],
                byteCount: m.outDim * (m.inDim / Q4G64.codesPerWord) * 4))
        }
        for m in matrices {
            let groups = m.inDim / Q4G64.groupSize
            for suffix in [Q4G64.scalesSuffix, Q4G64.biasesSuffix] {
                specs.append(OutputSpec(
                    name: m.name + suffix, dtype: TensorDType.float16.rawValue,
                    shape: [m.outDim, groups],
                    byteCount: m.outDim * groups * 2))
            }
        }
        for p in passthroughs {
            specs.append(OutputSpec(
                name: p.name, dtype: p.dtype.rawValue, shape: p.shape,
                byteCount: p.byteCount))
        }

        var offsetOf: [String: Int] = [:]
        var running = 0
        for spec in specs {
            offsetOf[spec.name] = running
            running += spec.byteCount
        }
        let dataSectionSize = running

        // Hand-built header: __metadata__ first, then tensor entries sorted
        // by name; values validated ASCII-safe so no JSON escaping is needed.
        var metadata: [String: String] = [
            Q4G64.formatMetadataKey: Q4G64.formatTag,
            Q4G64.groupSizeMetadataKey: "\(Q4G64.groupSize)",
            Q4G64.packerVersionMetadataKey: Q4G64.packerVersion,
            Q4G64.sourceRevisionMetadataKey: revision,
        ]
        if let repo = source.metadata[Q4G64.sourceRepoMetadataKey] {
            metadata[Q4G64.sourceRepoMetadataKey] = repo
        }
        for (key, value) in metadata {
            try validateHeaderSafe(key)
            try validateHeaderSafe(value)
        }
        var entries: [String] = []
        entries.append("\"__metadata__\":{"
            + metadata.keys.sorted()
                .map { "\"\($0)\":\"\(metadata[$0]!)\"" }
                .joined(separator: ",")
            + "}")
        for spec in specs.sorted(by: { $0.name < $1.name }) {
            let begin = offsetOf[spec.name]!
            entries.append("\"\(spec.name)\":{"
                + "\"dtype\":\"\(spec.dtype)\","
                + "\"shape\":[\(spec.shape.map(String.init).joined(separator: ","))],"
                + "\"data_offsets\":[\(begin),\(begin + spec.byteCount)]}")
        }
        var header = "{" + entries.joined(separator: ",") + "}"
        // Space-pad so the data section starts 8-byte aligned (the standard
        // safetensors alignment trick).
        while (8 + header.utf8.count) % 8 != 0 { header += " " }
        let headerBytes = Data(header.utf8)
        let dataSectionStart = 8 + headerBytes.count

        // D1: assert, don't assume, that every .q payload is 4-byte aligned.
        for m in matrices {
            let qName = m.name + Q4G64.qSuffix
            let absolute = dataSectionStart + offsetOf[qName]!
            guard absolute % 4 == 0 else {
                throw Q4G64Error.misalignedQTensor(
                    name: qName, absoluteByteOffset: absolute)
            }
        }

        guard FileManager.default.createFile(atPath: outputPath, contents: nil) else {
            throw SafetensorsError.cannotOpen(
                path: outputPath, reason: "cannot create output file")
        }
        // A failed pack must not leave a partial artifact behind — it looks
        // complete at a glance and only fails at load time.
        do {
            return try writeAndVerify(
                source: source, outputPath: outputPath, revision: revision,
                matrices: matrices, passthroughs: passthroughs,
                headerBytes: headerBytes, dataSectionStart: dataSectionStart,
                dataSectionSize: dataSectionSize, offsetOf: offsetOf,
                tiedLmHeadOmitted: tiedLmHeadOmitted, progress: progress)
        } catch {
            try? FileManager.default.removeItem(atPath: outputPath)
            throw error
        }
    }

    // MARK: - Internals

    private static func writeAndVerify(
        source: SafetensorsFile, outputPath: String, revision: String,
        matrices: [MatrixPlan], passthroughs: [TensorInfo],
        headerBytes: Data, dataSectionStart: Int, dataSectionSize: Int,
        offsetOf: [String: Int], tiedLmHeadOmitted: Bool, progress: Progress?
    ) throws -> Summary {
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: outputPath))
        defer { try? handle.close() }
        var prefix = Data()
        var headerLength = UInt64(headerBytes.count).littleEndian
        withUnsafeBytes(of: &headerLength) { prefix.append(contentsOf: $0) }
        try handle.write(contentsOf: prefix)
        try handle.write(contentsOf: headerBytes)

        // One source tensor at a time (the fp32 materialization of the
        // embedding alone is ~1.2 GB); each matrix's three payloads are
        // written by seeking into their laid-out ranges.
        let totalTensors = matrices.count + passthroughs.count
        var reported = 0
        for m in matrices {
            reported += 1
            progress?(m.name, reported, totalTensors)
            let packed = try packMatrix(source: source, name: m.name,
                                        outDim: m.outDim, inDim: m.inDim)
            try write(packed.qData, handle: handle,
                      at: dataSectionStart + offsetOf[m.name + Q4G64.qSuffix]!)
            try write(packed.scalesData, handle: handle,
                      at: dataSectionStart + offsetOf[m.name + Q4G64.scalesSuffix]!)
            try write(packed.biasesData, handle: handle,
                      at: dataSectionStart + offsetOf[m.name + Q4G64.biasesSuffix]!)
        }
        for p in passthroughs {
            reported += 1
            progress?(p.name, reported, totalTensors)
            // Raw byte copy straight from the source mapping: bf16 norms
            // stay bit-identical (no upcast round trip).
            let src = source.mappedBaseAddress + source.dataSectionStart + p.dataOffset
            let data = Data(bytes: src, count: p.byteCount)
            try write(data, handle: handle, at: dataSectionStart + offsetOf[p.name]!)
        }
        try handle.close()

        // Self-check: the freshly written file must load through the
        // validating reader with the same provenance.
        _ = try PackedCheckpoint(path: outputPath, expectedRevision: revision)

        return Summary(
            packedMatrices: matrices.count,
            passthroughTensors: passthroughs.count,
            outputByteCount: dataSectionStart + dataSectionSize,
            sourceRevision: revision,
            tiedLmHeadOmitted: tiedLmHeadOmitted)
    }

    private static func tiedLmHeadIsByteIdentical(in source: SafetensorsFile) -> Bool {
        guard let lmHead = try? source.info(for: lmHeadName),
              let embedding = try? source.info(for: embeddingName),
              lmHead.dtype == embedding.dtype,
              lmHead.shape == embedding.shape else {
            return false
        }
        let base = source.mappedBaseAddress + source.dataSectionStart
        return memcmp(base + lmHead.dataOffset, base + embedding.dataOffset,
                      lmHead.byteCount) == 0
    }

    private static func write(_ data: Data, handle: FileHandle, at offset: Int) throws {
        try handle.seek(toOffset: UInt64(offset))
        try handle.write(contentsOf: data)
    }

    /// The hand-built header skips JSON escaping, so every name/value must
    /// stay in the printable-ASCII subset without quotes or backslashes.
    private static func validateHeaderSafe(_ string: String) throws {
        let ok = string.utf8.allSatisfy { byte in
            byte >= 0x20 && byte < 0x7F && byte != UInt8(ascii: "\"")
                && byte != UInt8(ascii: "\\")
        }
        guard ok, !string.isEmpty else {
            throw Q4G64Error.unsafeName(name: string)
        }
    }

    private static func packMatrix(
        source: SafetensorsFile, name: String, outDim: Int, inDim: Int
    ) throws -> (qData: Data, scalesData: Data, biasesData: Data) {
        let values = try source.fp32Values(for: name)
        let groupsPerRow = inDim / Q4G64.groupSize
        var qWords = [UInt32]()
        qWords.reserveCapacity(outDim * inDim / Q4G64.codesPerWord)
        var scaleBits = [UInt16]()
        scaleBits.reserveCapacity(outDim * groupsPerRow)
        var biasBits = [UInt16]()
        biasBits.reserveCapacity(outDim * groupsPerRow)
        for row in 0..<outDim {
            let rowBase = row * inDim
            for g in 0..<groupsPerRow {
                let start = rowBase + g * Q4G64.groupSize
                let group = try Q4G64.packGroup(
                    values[start..<(start + Q4G64.groupSize)],
                    tensor: name, elementOffset: start)
                scaleBits.append(group.scale.bitPattern)
                biasBits.append(group.bias.bitPattern)
                qWords.append(contentsOf: Q4G64.packWords(group.codes))
            }
        }
        // Raw little-endian payloads (the platform is little-endian; the
        // parser reads with explicit littleEndian conversions).
        let qData = qWords.withUnsafeBufferPointer { Data(buffer: $0) }
        let scalesData = scaleBits.withUnsafeBufferPointer { Data(buffer: $0) }
        let biasesData = biasBits.withUnsafeBufferPointer { Data(buffer: $0) }
        return (qData, scalesData, biasesData)
    }
}
