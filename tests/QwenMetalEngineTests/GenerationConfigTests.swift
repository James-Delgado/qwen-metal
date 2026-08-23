import XCTest
import QwenMetalEngine

/// EOS-1: generation_config.json is the file HF `generate()` consults for
/// stopping (docs/AUDIT.md F2) — its eos_token_id list joins the decode stop
/// set. Parse edge cases mirror ModelConfig's eos_token_id handling: list,
/// scalar, absent key, malformed input, out-of-range ids.
final class GenerationConfigTests: XCTestCase {

    private func parse(_ json: String) throws -> GenerationConfig {
        try GenerationConfig(jsonData: Data(json.utf8))
    }

    func testParsesPinnedCheckpointShape() throws {
        // Field shape of the pinned checkpoint's generation_config.json
        // (sampling keys present but ignored — greedy is a protocol pin).
        let config = try parse("""
            {"bos_token_id": 151643, "do_sample": true,
             "eos_token_id": [151645, 151643], "pad_token_id": 151643,
             "temperature": 0.6, "top_k": 20, "top_p": 0.95}
            """)
        XCTAssertEqual(config.eosTokenIds, [151645, 151643])
    }

    func testScalarEOSTokenIdNormalizesToList() throws {
        XCTAssertEqual(try parse(#"{"eos_token_id": 151645}"#).eosTokenIds, [151645])
    }

    func testAbsentOrNullEOSKeyYieldsEmpty() throws {
        XCTAssertEqual(try parse(#"{"do_sample": false}"#).eosTokenIds, [])
        XCTAssertEqual(try parse(#"{"eos_token_id": null}"#).eosTokenIds, [])
    }

    func testMalformedJSONThrows() {
        XCTAssertThrowsError(try parse("not json"))
        XCTAssertThrowsError(try parse("[151645]")) // top level must be an object
    }

    func testInvalidIdsThrow() {
        // Same Int(exactly:) bound as config.json's eos_token_id (CFG-1):
        // negatives, out-of-Int-range values, and non-numbers all throw.
        XCTAssertThrowsError(try parse(#"{"eos_token_id": [151645, -1]}"#))
        XCTAssertThrowsError(try parse(#"{"eos_token_id": 1e20}"#))
        XCTAssertThrowsError(try parse(#"{"eos_token_id": "151645"}"#))
    }

    func testLoadMissingFileThrows() {
        XCTAssertThrowsError(
            try GenerationConfig.load(path: "/nonexistent/generation_config.json")
        ) { error in
            guard let configError = error as? ModelConfigError,
                  case .cannotOpen = configError else {
                return XCTFail("expected cannotOpen, got \(error)")
            }
        }
    }
}
