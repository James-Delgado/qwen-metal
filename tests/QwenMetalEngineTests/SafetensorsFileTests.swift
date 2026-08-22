import XCTest
@testable import QwenMetalEngine

/// P1-2 parser tests: the enumerated edge cases 1-5 from
/// docs/phases/phase-0-1.md, plus exact-upcast positive paths. Every input is
/// a tiny hand-built byte buffer written to a per-test temp directory — no
/// test touches a real checkpoint, so the suite runs on any machine.
final class SafetensorsFileTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("SafetensorsFileTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Byte-building helpers

    /// A well-formed blob: 8-byte little-endian header length, UTF-8 JSON
    /// header, then the raw data section.
    private func blob(headerJSON: String, payload: [UInt8] = []) -> Data {
        let header = Data(headerJSON.utf8)
        var out = Data()
        var length = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &length) { out.append(contentsOf: $0) }
        out.append(header)
        out.append(contentsOf: payload)
        return out
    }

    private func halfwordBytes(_ halfwords: [UInt16]) -> [UInt8] {
        halfwords.flatMap { [UInt8($0 & 0xFF), UInt8($0 >> 8)] }
    }

    @discardableResult
    private func write(_ data: Data, as name: String = "model.safetensors") throws -> String {
        let url = tempDir.appendingPathComponent(name)
        try data.write(to: url)
        return url.path
    }

    // MARK: - (1) bf16 -> fp32 exact upcast vs known hand-built bytes

    func testBF16UpcastIsExactAgainstHandBuiltBytes() throws {
        // bf16 is the top half of the fp32 bit pattern, so every expected
        // value below is exact by construction — asserted with ==, no gate.
        let bits: [UInt16] = [0x3F80, 0xBF80, 0x3E00, 0x4049, 0x0000, 0xC2F7]
        let expected: [Float] = [1.0, -1.0, 0.125, 3.140625, 0.0, -123.5]
        let json = #"{"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#
        let path = try write(blob(headerJSON: json, payload: halfwordBytes(bits)))

        let file = try SafetensorsFile(path: path)
        let info = try file.info(for: "w")
        XCTAssertEqual(info.dtype, .bfloat16)
        XCTAssertEqual(info.shape, [2, 3])
        XCTAssertEqual(info.elementCount, 6)
        XCTAssertEqual(try file.fp32Values(for: "w"), expected)
    }

    func testFP16UpcastIsExactAgainstHandBuiltBytes() throws {
        // fp16 -> fp32 is exact for every finite fp16, subnormals included.
        let bits: [UInt16] = [0x3C00, 0xC000, 0x3555, 0x0001, 0x7BFF]
        let expected: [Float] = [1.0, -2.0, 0.333251953125, 0x1p-24, 65504.0]
        let json = #"{"w":{"dtype":"F16","shape":[5],"data_offsets":[0,10]}}"#
        let path = try write(blob(headerJSON: json, payload: halfwordBytes(bits)))

        let file = try SafetensorsFile(path: path)
        XCTAssertEqual(try file.fp32Values(for: "w"), expected)
    }

    // MARK: - (2) truncated file -> clear error

    func testFileShorterThanLengthPrefixThrowsTruncated() throws {
        let path = try write(Data([0x02, 0x00, 0x00]))
        XCTAssertThrowsError(try SafetensorsFile(path: path)) { error in
            guard case SafetensorsError.truncated = error else {
                return XCTFail("Expected .truncated, got \(error)")
            }
        }
    }

    func testHeaderLengthPastEndOfFileThrowsTruncated() throws {
        // Claims a 4096-byte header but the file ends after 5 header bytes.
        var data = Data()
        var length = UInt64(4096).littleEndian
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(Data(#"{"a":"#.utf8))
        let path = try write(data)
        XCTAssertThrowsError(try SafetensorsFile(path: path)) { error in
            guard case SafetensorsError.truncated = error else {
                return XCTFail("Expected .truncated, got \(error)")
            }
        }
    }

    // MARK: - (3) malformed JSON header -> clear error

    func testMalformedJSONHeaderThrows() throws {
        let path = try write(blob(headerJSON: #"{"w": this is not json"#))
        XCTAssertThrowsError(try SafetensorsFile(path: path)) { error in
            guard case SafetensorsError.malformedHeader = error else {
                return XCTFail("Expected .malformedHeader, got \(error)")
            }
        }
    }

    func testHeaderThatIsNotAJSONObjectThrows() throws {
        let path = try write(blob(headerJSON: "[1, 2, 3]"))
        XCTAssertThrowsError(try SafetensorsFile(path: path)) { error in
            guard case SafetensorsError.malformedHeader = error else {
                return XCTFail("Expected .malformedHeader, got \(error)")
            }
        }
    }

    // MARK: - (4) sharded checkpoint -> loud reject

    func testShardedCheckpointIsRejectedLoudly() throws {
        let json = #"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#
        let shardPath = try write(
            blob(headerJSON: json, payload: [0, 0, 0, 0]),
            as: "model-00001-of-00002.safetensors")
        try write(Data("{}".utf8), as: "model.safetensors.index.json")

        XCTAssertThrowsError(try SafetensorsFile(path: shardPath)) { error in
            guard case SafetensorsError.shardedCheckpoint(_, let indexFile) = error else {
                return XCTFail("Expected .shardedCheckpoint, got \(error)")
            }
            XCTAssertEqual(indexFile, "model.safetensors.index.json")
            // The reject must be loud AND actionable: point at consolidation.
            XCTAssertTrue(
                String(describing: error).contains("consolidate"),
                "sharded-reject message should name the consolidation remedy")
        }
    }

    // MARK: - (5) out-of-bounds / overlapping tensor offsets -> clear error

    func testOffsetsBeyondDataSectionThrow() throws {
        // Data section is 4 bytes; the tensor claims [0, 8).
        let json = #"{"w":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}"#
        let path = try write(blob(headerJSON: json, payload: [0, 0, 0, 0]))
        XCTAssertThrowsError(try SafetensorsFile(path: path)) { error in
            guard case SafetensorsError.invalidTensor(let name, _) = error else {
                return XCTFail("Expected .invalidTensor, got \(error)")
            }
            XCTAssertEqual(name, "w")
        }
    }

    func testReversedOffsetsThrow() throws {
        let json = #"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[8,4]}}"#
        let path = try write(blob(headerJSON: json, payload: [UInt8](repeating: 0, count: 8)))
        XCTAssertThrowsError(try SafetensorsFile(path: path)) { error in
            guard case SafetensorsError.invalidTensor = error else {
                return XCTFail("Expected .invalidTensor, got \(error)")
            }
        }
    }

    func testOverlappingTensorsThrow() throws {
        let json = #"""
        {"a":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},
         "b":{"dtype":"BF16","shape":[4],"data_offsets":[4,12]}}
        """#
        let path = try write(blob(headerJSON: json, payload: [UInt8](repeating: 0, count: 12)))
        XCTAssertThrowsError(try SafetensorsFile(path: path)) { error in
            guard case SafetensorsError.invalidTensor(_, let detail) = error else {
                return XCTFail("Expected .invalidTensor, got \(error)")
            }
            XCTAssertTrue(detail.contains("overlap"), "detail should say overlap: \(detail)")
        }
    }

    func testOffsetSpanDtypeShapeMismatchThrows() throws {
        // shape [3] of BF16 needs 6 bytes; the offsets span 8.
        let json = #"{"w":{"dtype":"BF16","shape":[3],"data_offsets":[0,8]}}"#
        let path = try write(blob(headerJSON: json, payload: [UInt8](repeating: 0, count: 8)))
        XCTAssertThrowsError(try SafetensorsFile(path: path)) { error in
            guard case SafetensorsError.invalidTensor = error else {
                return XCTFail("Expected .invalidTensor, got \(error)")
            }
        }
    }

    // MARK: - Supporting behavior

    func testUnsupportedDtypeThrows() throws {
        let json = #"{"w":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}"#
        let path = try write(blob(headerJSON: json, payload: [UInt8](repeating: 0, count: 8)))
        XCTAssertThrowsError(try SafetensorsFile(path: path)) { error in
            guard case SafetensorsError.unsupportedDtype(let name, let dtype) = error else {
                return XCTFail("Expected .unsupportedDtype, got \(error)")
            }
            XCTAssertEqual(name, "w")
            XCTAssertEqual(dtype, "F64")
        }
    }

    func testTensorNotFoundThrows() throws {
        let json = #"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#
        let path = try write(blob(headerJSON: json, payload: [0, 0]))
        let file = try SafetensorsFile(path: path)
        XCTAssertThrowsError(try file.fp32Values(for: "nope")) { error in
            guard case SafetensorsError.tensorNotFound(let name) = error else {
                return XCTFail("Expected .tensorNotFound, got \(error)")
            }
            XCTAssertEqual(name, "nope")
        }
    }

    func testMissingFileThrowsCannotOpen() throws {
        let path = tempDir.appendingPathComponent("does-not-exist.safetensors").path
        XCTAssertThrowsError(try SafetensorsFile(path: path)) { error in
            guard case SafetensorsError.cannotOpen = error else {
                return XCTFail("Expected .cannotOpen, got \(error)")
            }
        }
    }

    func testMetadataAndMultipleTensorsParse() throws {
        let json = #"""
        {"__metadata__":{"source":"unit-test"},
         "b":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]},
         "a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}
        """#
        // a = [1.0, -1.0], b = [0.125, 3.140625]
        let payload = halfwordBytes([0x3F80, 0xBF80, 0x3E00, 0x4049])
        let path = try write(blob(headerJSON: json, payload: payload))

        let file = try SafetensorsFile(path: path)
        XCTAssertEqual(file.tensorNames, ["a", "b"])
        XCTAssertEqual(file.metadata, ["source": "unit-test"])
        XCTAssertEqual(try file.fp32Values(for: "a"), [1.0, -1.0])
        XCTAssertEqual(try file.fp32Values(for: "b"), [0.125, 3.140625])
    }
}
