import XCTest
@testable import QwenMetalEngine

/// P1-2 config-loader tests: enumerated edge cases 6 (missing key names the
/// key) and 7 (tied-embeddings flag both ways) from docs/phases/phase-0-1.md,
/// plus the family-flag derivations the Phase 1 modules will rely on.
final class ModelConfigTests: XCTestCase {

    /// Shaped exactly like the pinned Qwen/Qwen3-1.7B @ 70d244cc config.json,
    /// including keys the loader must tolerate and ignore.
    private func qwen3Dict() -> [String: Any] {
        [
            "model_type": "qwen3",
            "attention_bias": false,
            "head_dim": 128,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "max_position_embeddings": 40960,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1_000_000,
            "tie_word_embeddings": true,
            "vocab_size": 151936,
            // Ignored keys (present in the real file):
            "torch_dtype": "bfloat16",
            "use_cache": true,
            "rope_scaling": NSNull(),
        ]
    }

    private func config(from dict: [String: Any]) throws -> ModelConfig {
        try ModelConfig(jsonData: JSONSerialization.data(withJSONObject: dict))
    }

    func testParsesPinnedQwen3ShapedConfig() throws {
        let c = try config(from: qwen3Dict())
        XCTAssertEqual(c.modelType, "qwen3")
        XCTAssertEqual(c.hiddenSize, 2048)
        XCTAssertEqual(c.numHiddenLayers, 28)
        XCTAssertEqual(c.numAttentionHeads, 16)
        XCTAssertEqual(c.numKeyValueHeads, 8)
        XCTAssertEqual(c.headDim, 128)
        XCTAssertEqual(c.intermediateSize, 6144)
        XCTAssertEqual(c.rmsNormEps, 1e-06)
        XCTAssertEqual(c.ropeTheta, 1_000_000)
        XCTAssertEqual(c.vocabSize, 151936)
        XCTAssertTrue(c.tieWordEmbeddings)
        XCTAssertEqual(c.maxPositionEmbeddings, 40960)
        XCTAssertFalse(c.attentionBias)
        XCTAssertTrue(c.usesQKNorm)
    }

    // MARK: - eos_token_id (P1-5: decode stop tokens; optional key)

    func testEOSTokenIdAbsentDefaultsToEmpty() throws {
        XCTAssertEqual(try config(from: qwen3Dict()).eosTokenIds, [])
    }

    func testEOSTokenIdParsesScalarAndList() throws {
        var dict = qwen3Dict()
        dict["eos_token_id"] = 151645
        XCTAssertEqual(try config(from: dict).eosTokenIds, [151645])
        dict["eos_token_id"] = [151645, 151643]
        XCTAssertEqual(try config(from: dict).eosTokenIds, [151645, 151643])
    }

    func testEOSTokenIdRejectsNonIntegerValues() throws {
        var dict = qwen3Dict()
        dict["eos_token_id"] = "<|im_end|>"
        XCTAssertThrowsError(try config(from: dict)) { error in
            XCTAssertEqual(
                error as? ModelConfigError,
                .invalidValue(
                    key: "eos_token_id",
                    detail: "expected a token id or list of token ids"))
        }
    }

    // MARK: - (6) missing required key -> clear error naming the key

    func testEveryMissingRequiredKeyIsNamedInTheError() throws {
        let requiredKeys = [
            "model_type", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "intermediate_size",
            "rms_norm_eps", "rope_theta", "vocab_size", "tie_word_embeddings",
            "max_position_embeddings",
        ]
        for key in requiredKeys {
            var dict = qwen3Dict()
            dict.removeValue(forKey: key)
            XCTAssertThrowsError(try config(from: dict), "removing \(key)") { error in
                guard case ModelConfigError.missingKey(let named) = error else {
                    return XCTFail("Expected .missingKey(\(key)), got \(error)")
                }
                XCTAssertEqual(named, key)
            }
        }
    }

    func testJSONNullCountsAsMissing() throws {
        var dict = qwen3Dict()
        dict["vocab_size"] = NSNull()
        XCTAssertThrowsError(try config(from: dict)) { error in
            guard case ModelConfigError.missingKey(let named) = error else {
                return XCTFail("Expected .missingKey, got \(error)")
            }
            XCTAssertEqual(named, "vocab_size")
        }
    }

    // MARK: - (7) tied-embeddings flag exercised both ways

    func testTiedEmbeddingsFlagParsesBothWays() throws {
        var dict = qwen3Dict()
        dict["tie_word_embeddings"] = true
        XCTAssertTrue(try config(from: dict).tieWordEmbeddings)
        dict["tie_word_embeddings"] = false
        XCTAssertFalse(try config(from: dict).tieWordEmbeddings)
    }

    // MARK: - Family flags + derived values

    func testHeadDimFallsBackToHiddenSizeOverHeads() throws {
        // Qwen 2.5 configs omit head_dim; the derived value must appear.
        var dict = qwen3Dict()
        dict.removeValue(forKey: "head_dim")
        XCTAssertEqual(try config(from: dict).headDim, 128) // 2048 / 16
    }

    func testAttentionBiasDefaultsByFamilyWhenAbsent() throws {
        var q3 = qwen3Dict()
        q3.removeValue(forKey: "attention_bias")
        XCTAssertFalse(try config(from: q3).attentionBias)

        var q2 = qwen3Dict()
        q2["model_type"] = "qwen2"
        q2.removeValue(forKey: "attention_bias")
        let c2 = try config(from: q2)
        XCTAssertTrue(c2.attentionBias)  // Qwen 2.x family ships QKV biases
        XCTAssertFalse(c2.usesQKNorm)    // qk-norm is the Qwen 3 fork only
    }

    func testUnknownFamilyWithoutAttentionBiasKeyThrows() throws {
        var dict = qwen3Dict()
        dict["model_type"] = "llama"
        dict.removeValue(forKey: "attention_bias")
        XCTAssertThrowsError(try config(from: dict)) { error in
            guard case ModelConfigError.missingKey(let named) = error else {
                return XCTFail("Expected .missingKey, got \(error)")
            }
            XCTAssertEqual(named, "attention_bias")
        }
    }

    func testGQAHeadCountsMustDivide() throws {
        var dict = qwen3Dict()
        dict["num_key_value_heads"] = 5
        XCTAssertThrowsError(try config(from: dict)) { error in
            guard case ModelConfigError.invalidValue(let key, _) = error else {
                return XCTFail("Expected .invalidValue, got \(error)")
            }
            XCTAssertEqual(key, "num_key_value_heads")
        }
    }

    // MARK: - Malformed input / file loading

    func testMalformedJSONThrows() throws {
        XCTAssertThrowsError(try ModelConfig(jsonData: Data("not json".utf8))) { error in
            guard case ModelConfigError.malformedJSON = error else {
                return XCTFail("Expected .malformedJSON, got \(error)")
            }
        }
    }

    func testLoadFromPathRoundTrips() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelConfigTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: qwen3Dict()).write(to: url)
        let c = try ModelConfig.load(path: url.path)
        XCTAssertEqual(c.hiddenSize, 2048)
        XCTAssertEqual(c.numHiddenLayers, 28)
    }

    func testLoadFromMissingPathThrowsCannotOpen() throws {
        XCTAssertThrowsError(try ModelConfig.load(path: "/nonexistent/config.json")) { error in
            guard case ModelConfigError.cannotOpen = error else {
                return XCTFail("Expected .cannotOpen, got \(error)")
            }
        }
    }
}
