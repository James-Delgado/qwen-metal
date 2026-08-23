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

    // MARK: - CFG-1: numeric bounds/finiteness (docs/AUDIT.md F1/F3/F4)

    /// Serializes `qwen3Dict()` minus `key`, then splices `"key": <literal>`
    /// back in as raw JSON text — for number literals a Swift dictionary
    /// cannot carry losslessly (2^63, 10^20, 1e999).
    private func jsonData(overriding key: String, withRawLiteral literal: String) throws -> Data {
        var dict = qwen3Dict()
        dict.removeValue(forKey: key)
        var text = String(decoding: try JSONSerialization.data(withJSONObject: dict), as: UTF8.self)
        precondition(text.hasSuffix("}"))
        text.removeLast()
        text += ",\"\(key)\":\(literal)}"
        return Data(text.utf8)
    }

    private func assertThrowsInvalidValue(
        _ data: Data, key expectedKey: String,
        _ message: @autoclosure () -> String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(
            try ModelConfig(jsonData: data), message(), file: file, line: line
        ) { error in
            guard case ModelConfigError.invalidValue(let key, _) = error else {
                return XCTFail(
                    "Expected .invalidValue(\(expectedKey)), got \(error) \(message())",
                    file: file, line: line)
            }
            XCTAssertEqual(key, expectedKey, message(), file: file, line: line)
        }
    }

    func testPositiveIntRejectsExactlyTwoToThe63() throws {
        // AUDIT F1: Double(Int.max) rounds UP to 2^63, so the old inclusive
        // `<= Double(Int.max)` admitted exactly 2^63; NSNumber.intValue then
        // wrapped it to Int.min, trapping later in QwenModel.init.
        for key in ["num_hidden_layers", "num_attention_heads"] {
            assertThrowsInvalidValue(
                try jsonData(overriding: key, withRawLiteral: "9223372036854775808"),
                key: key, "for \(key) = 2^63")
        }
    }

    func testPositiveIntRejectsValuesBeyondIntRange() throws {
        for literal in ["100000000000000000000", "1e20"] {
            assertThrowsInvalidValue(
                try jsonData(overriding: "num_attention_heads", withRawLiteral: literal),
                key: "num_attention_heads", "for literal \(literal)")
        }
    }

    func testPositiveIntStillAcceptsIntMax() throws {
        // Int.max is a representable positive integer and must stay accepted:
        // a strict `< Double(2^63)` comparison would wrongly reject it, since
        // Double cannot distinguish Int.max from 2^63.
        let data = try jsonData(
            overriding: "max_position_embeddings", withRawLiteral: "9223372036854775807")
        XCTAssertEqual(try ModelConfig(jsonData: data).maxPositionEmbeddings, Int.max)
    }

    func testDoubleFieldsRejectZeroAndNegative() throws {
        // AUDIT F3: negative rms_norm_eps -> NaN via sqrt(negative) in
        // RMSNorm; negative rope_theta -> NaN via powf(negative, fractional)
        // in RoPE. NaN logits then decode as silent garbage.
        for (key, literal) in [
            ("rms_norm_eps", "-1e-06"), ("rms_norm_eps", "0"),
            ("rope_theta", "-1000000"), ("rope_theta", "0"),
        ] {
            assertThrowsInvalidValue(
                try jsonData(overriding: key, withRawLiteral: literal),
                key: key, "for \(key) = \(literal)")
        }
    }

    func testDoubleFieldsRejectOverflowingLiterals() throws {
        // 1e999 overflows Double. Whether the JSON parser surfaces +inf or
        // rejects the literal, ModelConfig must throw a ModelConfigError
        // rather than carry a non-finite value into RMSNorm/RoPE.
        for key in ["rms_norm_eps", "rope_theta"] {
            let data = try jsonData(overriding: key, withRawLiteral: "1e999")
            XCTAssertThrowsError(try ModelConfig(jsonData: data), key) { error in
                XCTAssertTrue(error is ModelConfigError, "for \(key): got \(error)")
            }
        }
    }

    func testEOSTokenIdRejectsOutOfRangeIds() throws {
        // AUDIT F4: intList lacked the range check its sibling positiveInt
        // performs — 1e20 silently became Int.max, 10^20 wraparound garbage.
        for literal in ["1e20", "100000000000000000000", "[151645, 9223372036854775808]"] {
            assertThrowsInvalidValue(
                try jsonData(overriding: "eos_token_id", withRawLiteral: literal),
                key: "eos_token_id", "for literal \(literal)")
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
