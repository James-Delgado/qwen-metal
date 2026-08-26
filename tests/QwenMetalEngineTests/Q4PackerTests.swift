import XCTest
import Foundation
import QwenMetalEngine

/// File-level tests for the q4g64 packer and the validating loader
/// (phase-3.md edge cases 2/5/6/7/8/9 in their P3-1 slice): end-to-end
/// pack→load→dequant exactness, group-boundary indexing, byte-identical
/// determinism, rejection of unpackable shapes, doctored-file alignment and
/// triplet-consistency rejects, provenance, and the parser's U32 handling.
final class Q4PackerTests: XCTestCase {

    private var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("Q4PackerTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
    }

    private func path(_ name: String) -> String {
        tmpDir.appendingPathComponent(name).path
    }

    // MARK: - Builders

    /// bf16 bit pattern of an EXACTLY representable fp32 value (≤8
    /// significant bits); NaN allowed for the corruption test.
    private func bf16Bits(_ v: Float) -> UInt16 {
        let bits = UInt16(truncatingIfNeeded: v.bitPattern >> 16)
        let back = Float(bitPattern: UInt32(bits) << 16)
        precondition(back == v || (v.isNaN && back.isNaN),
                     "test value \(v) is not bf16-exact")
        return bits
    }

    private func littleEndianBytes(_ halfwords: [UInt16]) -> [UInt8] {
        halfwords.flatMap { [UInt8($0 & 0xFF), UInt8($0 >> 8)] }
    }

    private func littleEndianBytes(_ words: [UInt32]) -> [UInt8] {
        words.flatMap { w in (0..<4).map { UInt8((w >> (8 * $0)) & 0xFF) } }
    }

    /// Well-formed safetensors blob with the header space-padded so the
    /// data section starts 8-byte aligned (mirrors the packer, and lets the
    /// alignment tests control offsets mod 4 exactly).
    private func blob(headerJSON: String, payload: [UInt8]) -> Data {
        var header = headerJSON
        while (8 + header.utf8.count) % 8 != 0 { header += " " }
        let headerData = Data(header.utf8)
        var out = Data()
        var length = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &length) { out.append(contentsOf: $0) }
        out.append(headerData)
        out.append(contentsOf: payload)
        return out
    }

    /// Builds a bf16 source checkpoint from (name, shape, values) triples.
    private func makeSourceFile(
        tensors: [(name: String, shape: [Int], values: [Float])],
        metadata: [String: String] = ["source_revision": "r1"],
        fileName: String = "source.safetensors"
    ) throws -> String {
        var entries: [String] = []
        if !metadata.isEmpty {
            entries.append("\"__metadata__\":{"
                + metadata.keys.sorted().map { "\"\($0)\":\"\(metadata[$0]!)\"" }
                    .joined(separator: ",")
                + "}")
        }
        var payload: [UInt8] = []
        var offset = 0
        for t in tensors {
            precondition(t.values.count == t.shape.reduce(1, *))
            let bytes = littleEndianBytes(t.values.map(bf16Bits))
            entries.append("\"\(t.name)\":{\"dtype\":\"BF16\","
                + "\"shape\":[\(t.shape.map(String.init).joined(separator: ","))],"
                + "\"data_offsets\":[\(offset),\(offset + bytes.count)]}")
            payload += bytes
            offset += bytes.count
        }
        let data = blob(headerJSON: "{" + entries.joined(separator: ",") + "}",
                        payload: payload)
        let filePath = path(fileName)
        try data.write(to: URL(fileURLWithPath: filePath))
        return filePath
    }

    /// The standard two-tensor source used by several tests: one 2-row
    /// matrix with four distinct groups and one 1-D norm vector. All values
    /// bf16-exact.
    private func standardSource() throws -> (path: String, matrix: [Float], norm: [Float]) {
        // Row 0: group 0 = halves 0..31.5, group 1 = integers 100..163.
        // Row 1: group 0 = negative quarters, group 1 = even ints 64..190.
        var matrix = [Float]()
        matrix += (0..<64).map { Float($0) * 0.5 }
        matrix += (0..<64).map { Float(100 + $0) }
        matrix += (0..<64).map { Float($0) * -0.25 }
        matrix += (0..<64).map { Float(64 + 2 * $0) }
        let norm: [Float] = [1.0, -0.5, 0.25, 2, -3, 0.125, 5, -7]
        let source = try makeSourceFile(tensors: [
            ("m.weight", [2, 128], matrix),
            ("n.weight", [8], norm),
        ])
        return (source, matrix, norm)
    }

    /// CPU-side expected dequant of a row-major matrix via the shared
    /// arithmetic — what the packed file must reproduce bit-for-bit.
    private func expectedDequant(_ values: [Float], tensor: String) throws -> [Float] {
        var out = [Float]()
        out.reserveCapacity(values.count)
        var start = 0
        while start < values.count {
            let g = try Q4G64.packGroup(
                values[start..<(start + Q4G64.groupSize)],
                tensor: tensor, elementOffset: start)
            for code in g.codes {
                out.append(Q4G64.dequant(code: UInt32(code), scale: g.scale, bias: g.bias))
            }
            start += Q4G64.groupSize
        }
        return out
    }

    // MARK: - End-to-end pack -> load -> dequant

    func testEndToEndDequantMatchesCPUArithmeticExactly() throws {
        let (source, matrix, norm) = try standardSource()
        let out = path("packed.safetensors")
        let summary = try Q4Packer.pack(inputPath: source, outputPath: out)
        XCTAssertEqual(summary.packedMatrices, 1)
        XCTAssertEqual(summary.passthroughTensors, 1)
        XCTAssertEqual(summary.sourceRevision, "r1")

        let packed = try PackedCheckpoint(path: out, expectedRevision: "r1")
        XCTAssertEqual(packed.matrixNames, ["m.weight"])
        XCTAssertEqual(packed.passthroughNames, ["n.weight"])
        let dims = try packed.dims(for: "m.weight")
        XCTAssertEqual(dims.outDim, 2)
        XCTAssertEqual(dims.inDim, 128)

        // Bitwise: pack->file->load->dequant must equal the in-memory
        // arithmetic (layer-1 exactness premise, CPU side).
        let dequant = try packed.dequantMatrix("m.weight")
        let expected = try expectedDequant(matrix, tensor: "m.weight")
        XCTAssertEqual(dequant, expected)

        // Pass-through norm survives byte-identically (values bf16-exact).
        XCTAssertEqual(try packed.file.fp32Values(for: "n.weight"), norm)

        // Metadata carries format + provenance.
        XCTAssertEqual(packed.file.metadata["format"], "q4g64")
        XCTAssertEqual(packed.file.metadata["group_size"], "64")
        XCTAssertEqual(packed.file.metadata["packer_version"], "1")
    }

    // MARK: - Edge case 2: group-boundary indexing

    func testGroupBoundaryElementsDequantAgainstTheirOwnGroup() throws {
        // Group 0 lives near zero; group 1 lives near 1024 with a distinct
        // scale — an off-by-one group index is a ~1000-magnitude mismatch.
        var values = [Float]()
        values += (0..<64).map { Float($0) }            // 0..63
        values += (0..<64).map { Float(1024 + 8 * $0) } // 1024..1528
        let source = try makeSourceFile(tensors: [("m.weight", [1, 128], values)])
        let out = path("packed.safetensors")
        _ = try Q4Packer.pack(inputPath: source, outputPath: out)

        let packed = try PackedCheckpoint(path: out)
        let dequant = try packed.dequantMatrix("m.weight")
        let expected = try expectedDequant(values, tensor: "m.weight")
        XCTAssertEqual(dequant, expected)
        // Elements 63 / 64 / 65 sit on the classic off-by-one site.
        XCTAssertEqual(dequant[63], expected[63])
        XCTAssertLessThan(dequant[63], 100)
        XCTAssertGreaterThan(dequant[64], 1000)
        XCTAssertGreaterThan(dequant[65], 1000)
    }

    // MARK: - Edge case 5: determinism

    func testPackIsDeterministicByteForByte() throws {
        let (source, _, _) = try standardSource()
        let out1 = path("packed1.safetensors")
        let out2 = path("packed2.safetensors")
        _ = try Q4Packer.pack(inputPath: source, outputPath: out1)
        _ = try Q4Packer.pack(inputPath: source, outputPath: out2)
        let data1 = try Data(contentsOf: URL(fileURLWithPath: out1))
        let data2 = try Data(contentsOf: URL(fileURLWithPath: out2))
        XCTAssertEqual(data1, data2)
        XCTAssertFalse(data1.isEmpty)
    }

    // MARK: - Edge case 6: non-multiple-of-64 in-dim

    func testPackerRejectsNonMultipleOf64InDim() throws {
        let source = try makeSourceFile(tensors: [
            ("m.weight", [2, 96], (0..<192).map { Float($0 % 32) }),
        ])
        XCTAssertThrowsError(
            try Q4Packer.pack(inputPath: source, outputPath: path("out.safetensors"))
        ) { error in
            guard case Q4G64Error.inDimNotMultipleOfGroup(let tensor, let shape) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(tensor, "m.weight")
            XCTAssertEqual(shape, [2, 96])
        }
    }

    func testLoaderRejectsInconsistentTriplet() throws {
        // q claims in-dim 96 (12 words); scales/biases claim 2 groups
        // (in-dim 128) — 96 is not even a multiple of 64.
        let qBytes = littleEndianBytes([UInt32](repeating: 0, count: 24))
        let halfBytes = littleEndianBytes([UInt16](repeating: 0, count: 4))
        let header = "{"
            + "\"__metadata__\":{\"format\":\"q4g64\",\"group_size\":\"64\",\"source_revision\":\"r1\"},"
            + "\"m.weight.q\":{\"dtype\":\"U32\",\"shape\":[2,12],\"data_offsets\":[0,96]},"
            + "\"m.weight.scales\":{\"dtype\":\"F16\",\"shape\":[2,2],\"data_offsets\":[96,104]},"
            + "\"m.weight.biases\":{\"dtype\":\"F16\",\"shape\":[2,2],\"data_offsets\":[104,112]}"
            + "}"
        let filePath = path("bad.safetensors")
        try blob(headerJSON: header, payload: qBytes + halfBytes + halfBytes)
            .write(to: URL(fileURLWithPath: filePath))
        XCTAssertThrowsError(try PackedCheckpoint(path: filePath)) { error in
            guard case Q4G64Error.badTriplet(let name, _) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(name, "m.weight")
        }
    }

    // MARK: - Edge case 7: .q alignment

    func testLoaderRejectsMisalignedQTensor() throws {
        // A 2-byte tensor ahead of .q puts the u32 payload at absolute
        // offset ≡ 2 (mod 4); the header padding keeps the data section
        // 8-aligned, so the misalignment is exactly controlled.
        let padBytes = littleEndianBytes([UInt16(0x3C00)])
        let qBytes = littleEndianBytes([UInt32](repeating: 0, count: 8))
        let halfBytes = littleEndianBytes([UInt16(0)])
        let header = "{"
            + "\"__metadata__\":{\"format\":\"q4g64\",\"group_size\":\"64\",\"source_revision\":\"r1\"},"
            + "\"pad\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]},"
            + "\"m.weight.q\":{\"dtype\":\"U32\",\"shape\":[1,8],\"data_offsets\":[2,34]},"
            + "\"m.weight.scales\":{\"dtype\":\"F16\",\"shape\":[1,1],\"data_offsets\":[34,36]},"
            + "\"m.weight.biases\":{\"dtype\":\"F16\",\"shape\":[1,1],\"data_offsets\":[36,38]}"
            + "}"
        let filePath = path("misaligned.safetensors")
        try blob(headerJSON: header, payload: padBytes + qBytes + halfBytes + halfBytes)
            .write(to: URL(fileURLWithPath: filePath))
        XCTAssertThrowsError(try PackedCheckpoint(path: filePath)) { error in
            guard case Q4G64Error.misalignedQTensor(let name, let offset) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(name, "m.weight.q")
            XCTAssertEqual(offset % 4, 2)
        }
    }

    func testPackedOutputSatisfiesAlignmentInvariants() throws {
        // Read the raw file: data section starts 8-aligned and every .q
        // tensor's absolute offset is ≡ 0 (mod 4) — asserted from the bytes,
        // independent of the loader.
        let (source, _, _) = try standardSource()
        let out = path("packed.safetensors")
        _ = try Q4Packer.pack(inputPath: source, outputPath: out)

        let data = try Data(contentsOf: URL(fileURLWithPath: out))
        let headerLength = data.prefix(8).withUnsafeBytes {
            Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)))
        }
        let dataSectionStart = 8 + headerLength
        XCTAssertEqual(dataSectionStart % 8, 0)

        let headerJSON = try XCTUnwrap(try JSONSerialization.jsonObject(
            with: data.subdata(in: 8..<dataSectionStart)) as? [String: Any])
        var sawQ = false
        for (name, entry) in headerJSON where name.hasSuffix(".q") {
            sawQ = true
            let dict = try XCTUnwrap(entry as? [String: Any])
            XCTAssertEqual(dict["dtype"] as? String, "U32", name)
            let offsets = try XCTUnwrap(dict["data_offsets"] as? [Int])
            XCTAssertEqual((dataSectionStart + offsets[0]) % 4, 0, name)
        }
        XCTAssertTrue(sawQ)
    }

    // MARK: - Edge case 8: provenance

    func testLoaderRejectsRevisionMismatch() throws {
        let (source, _, _) = try standardSource()
        let out = path("packed.safetensors")
        _ = try Q4Packer.pack(inputPath: source, outputPath: out)
        XCTAssertThrowsError(
            try PackedCheckpoint(path: out, expectedRevision: "deadbeef")
        ) { error in
            guard case Q4G64Error.revisionMismatch(_, let found, let expected) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(found, "r1")
            XCTAssertEqual(expected, "deadbeef")
        }
    }

    func testPackerRequiresSourceRevision() throws {
        let source = try makeSourceFile(
            tensors: [("m.weight", [1, 64], (0..<64).map { Float($0) })],
            metadata: [:])
        XCTAssertThrowsError(
            try Q4Packer.pack(inputPath: source, outputPath: path("out.safetensors"))
        ) { error in
            guard case Q4G64Error.missingSourceRevision = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    // MARK: - Edge case 9 (P3-1 side): wrong-format loads

    func testPackedLoaderRejectsBF16Checkpoint() throws {
        let (source, _, _) = try standardSource()
        XCTAssertThrowsError(try PackedCheckpoint(path: source)) { error in
            guard case Q4G64Error.notAPackedCheckpoint(_, let detail) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(detail.contains("q4g64"), detail)
        }
    }

    func testPackerRejectsAlreadyPackedInput() throws {
        let (source, _, _) = try standardSource()
        let out = path("packed.safetensors")
        _ = try Q4Packer.pack(inputPath: source, outputPath: out)
        XCTAssertThrowsError(
            try Q4Packer.pack(inputPath: out, outputPath: path("twice.safetensors"))
        ) { error in
            guard case Q4G64Error.inputAlreadyPacked = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    // MARK: - Edge case 10 (P3-1 side): tied lm_head stored once

    func testTiedLmHeadByteIdenticalToEmbeddingIsStoredOnce() throws {
        // The pinned consolidated checkpoint materializes lm_head.weight
        // byte-identical to the embedding; schema D1 stores the tied matrix
        // once.
        let embedding = (0..<256).map { Float($0 % 32) * 0.5 - 4 }
        let source = try makeSourceFile(tensors: [
            ("lm_head.weight", [4, 64], embedding),
            ("model.embed_tokens.weight", [4, 64], embedding),
        ])
        let out = path("packed.safetensors")
        let summary = try Q4Packer.pack(inputPath: source, outputPath: out)
        XCTAssertTrue(summary.tiedLmHeadOmitted)
        XCTAssertEqual(summary.packedMatrices, 1)
        let packed = try PackedCheckpoint(path: out)
        XCTAssertEqual(packed.matrixNames, ["model.embed_tokens.weight"])
    }

    func testUntiedLmHeadIsPackedSeparately() throws {
        let embedding = (0..<256).map { Float($0 % 32) * 0.5 - 4 }
        var lmHead = embedding
        lmHead[0] = 7.5 // one differing (bf16-exact) value: NOT tied
        let source = try makeSourceFile(tensors: [
            ("lm_head.weight", [4, 64], lmHead),
            ("model.embed_tokens.weight", [4, 64], embedding),
        ])
        let out = path("packed.safetensors")
        let summary = try Q4Packer.pack(inputPath: source, outputPath: out)
        XCTAssertFalse(summary.tiedLmHeadOmitted)
        XCTAssertEqual(summary.packedMatrices, 2)
        let packed = try PackedCheckpoint(path: out)
        XCTAssertEqual(packed.matrixNames,
                       ["lm_head.weight", "model.embed_tokens.weight"])
    }

    // MARK: - Corruption and unpackable shapes

    func testPackerAbortsOnNonFiniteWeight() throws {
        var values = (0..<128).map { Float($0 % 16) }
        values[70] = .nan
        let source = try makeSourceFile(tensors: [("m.weight", [1, 128], values)])
        XCTAssertThrowsError(
            try Q4Packer.pack(inputPath: source, outputPath: path("out.safetensors"))
        ) { error in
            guard case Q4G64Error.nonFiniteWeight(let tensor, let index) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(tensor, "m.weight")
            XCTAssertEqual(index, 70)
        }
    }

    func testFailedPackLeavesNoPartialOutputFile() throws {
        var values = (0..<128).map { Float($0 % 16) }
        values[70] = .nan
        let source = try makeSourceFile(tensors: [("m.weight", [1, 128], values)])
        let out = path("partial.safetensors")
        XCTAssertThrowsError(try Q4Packer.pack(inputPath: source, outputPath: out))
        XCTAssertFalse(FileManager.default.fileExists(atPath: out),
                       "a failed pack must not leave a partial artifact")
    }

    func testPackerRejectsRank3Tensor() throws {
        let source = try makeSourceFile(tensors: [
            ("t", [2, 2, 64], (0..<256).map { Float($0 % 8) }),
        ])
        XCTAssertThrowsError(
            try Q4Packer.pack(inputPath: source, outputPath: path("out.safetensors"))
        ) { error in
            guard case Q4G64Error.unsupportedRank(let tensor, let shape) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(tensor, "t")
            XCTAssertEqual(shape, [2, 2, 64])
        }
    }

    // MARK: - Parser: U32 support boundary

    func testParserReadsU32ButRefusesFP32Materialization() throws {
        let words: [UInt32] = [0x7654_3210, 0xFEDC_BA98, 1, 2]
        let header = "{\"t\":{\"dtype\":\"U32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}"
        let filePath = path("u32.safetensors")
        try blob(headerJSON: header, payload: littleEndianBytes(words))
            .write(to: URL(fileURLWithPath: filePath))
        let file = try SafetensorsFile(path: filePath)
        let info = try file.info(for: "t")
        XCTAssertEqual(info.dtype, .uint32)
        XCTAssertEqual(info.shape, [2, 2])
        XCTAssertThrowsError(try file.fp32Values(for: "t")) { error in
            guard case SafetensorsError.unsupportedDtype(let name, let dtype) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(name, "t")
            XCTAssertEqual(dtype, "U32")
        }
    }
}
