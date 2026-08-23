import XCTest
import Foundation
import QwenMetalEngine

/// Shared checkpoint plumbing for the oracle tests. The fp32-materialized
/// 1.7B model costs ~7 GB resident, so ActivationFixtureTests and
/// LogitMatchSuiteTests share ONE instance (XCTest runs classes serially in
/// one process). The RoPE table is sized 256: covers the activation slices
/// (seq 5) and the logit suite's longest teacher-forced sequence
/// (62-token prompt + 50 steps).
enum SharedCheckpoint {
    static let pinnedRevision = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

    /// Mirror of the pinned checkpoint's config.json (Qwen/Qwen3-1.7B
    /// @ 70d244cc; values also recorded in the DECISIONS.md PIN-1 entry).
    /// The from-disk config path is exercised by the CLI, not here.
    static let pinnedConfigJSON = """
    {
      "attention_bias": false,
      "eos_token_id": 151645,
      "head_dim": 128,
      "hidden_size": 2048,
      "intermediate_size": 6144,
      "max_position_embeddings": 40960,
      "model_type": "qwen3",
      "num_attention_heads": 16,
      "num_hidden_layers": 28,
      "num_key_value_heads": 8,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000,
      "tie_word_embeddings": true,
      "vocab_size": 151936
    }
    """

    static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // SharedCheckpoint.swift
        .deletingLastPathComponent() // QwenMetalEngineTests/
        .deletingLastPathComponent() // tests/ → repo root
    static let fixturesDir =
        repoRoot.appendingPathComponent("tests/fixtures/qwen3-1.7b")
    static let modelsDir = repoRoot.appendingPathComponent("models")
    static let checkpointURL =
        modelsDir.appendingPathComponent("qwen3-1.7b-70d244cc.safetensors")

    struct HelperError: Error, CustomStringConvertible {
        let description: String
    }

    private static var shared: Result<QwenModel, Error>?

    /// Call from setUpWithError of any checkpoint-needing test class.
    static func skipUnlessCheckpointPresent() throws {
        guard FileManager.default.fileExists(atPath: checkpointURL.path) else {
            throw XCTSkip(
                "consolidated checkpoint not present at \(checkpointURL.path) "
                + "(local-only artifact — produce it with tools/consolidate_shards.py)")
        }
    }

    static func model() throws -> QwenModel {
        if shared == nil {
            shared = Result {
                let checkpoint = try SafetensorsFile(path: checkpointURL.path)
                guard checkpoint.metadata["source_revision"] == pinnedRevision else {
                    throw HelperError(description:
                        "checkpoint at \(checkpointURL.path) has source_revision "
                        + "'\(checkpoint.metadata["source_revision"] ?? "<missing>")', "
                        + "expected the pinned \(pinnedRevision) — regenerate with "
                        + "tools/consolidate_shards.py")
                }
                let config = try ModelConfig(
                    jsonData: pinnedConfigJSON.data(using: .utf8)!)
                return try QwenModel(
                    checkpoint: checkpoint, config: config, maxSequenceLength: 256)
            }
        }
        return try shared!.get()
    }

    // MARK: - Fixture blob loading (raw little-endian per the manifest)

    static func floats(at relativePath: String, expectedCount: Int) throws -> [Float] {
        let data = try blob(at: relativePath, expectedCount: expectedCount, elementSize: 4)
        return data.withUnsafeBytes { raw in
            raw.bindMemory(to: UInt32.self).map {
                Float(bitPattern: UInt32(littleEndian: $0))
            }
        }
    }

    static func int32s(at relativePath: String, expectedCount: Int) throws -> [Int] {
        let data = try blob(at: relativePath, expectedCount: expectedCount, elementSize: 4)
        return data.withUnsafeBytes { raw in
            raw.bindMemory(to: Int32.self).map { Int(Int32(littleEndian: $0)) }
        }
    }

    private static func blob(
        at relativePath: String, expectedCount: Int, elementSize: Int
    ) throws -> Data {
        let url = fixturesDir.appendingPathComponent(relativePath)
        let data = try Data(contentsOf: url)
        guard data.count == expectedCount * elementSize else {
            throw HelperError(description:
                "\(relativePath): byte length \(data.count) disagrees with the "
                + "expected \(expectedCount) x \(elementSize)-byte elements")
        }
        return data
    }

    /// One prompt's entry from the committed tokenizer dump. `inputText` is
    /// the fully rendered string (chat_template included — P1-1 DECISIONS
    /// entry), `inputIds` the Python tokenizer's ids for it.
    static func promptFixture(_ name: String) throws -> (inputText: String, inputIds: [Int]) {
        let url = fixturesDir.appendingPathComponent("tokenizer_ids.json")
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        guard let dict = json as? [String: Any],
              let prompt = dict[name] as? [String: Any],
              let text = prompt["input_text"] as? String,
              let ids = prompt["input_ids"] as? [Int] else {
            throw HelperError(description:
                "tokenizer_ids.json missing input_text/input_ids for '\(name)'")
        }
        return (text, ids)
    }
}
