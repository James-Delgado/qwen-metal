import XCTest
import Foundation
import QwenMetalEngine

/// P1-4 per-module oracle tests: each Swift module is fed its reference
/// INPUT slice from the committed HF fp32 activation fixtures and diffed
/// against its reference OUTPUT slice — before any pipeline wiring
/// (phase-0-1.md correctness harness). Gates are pre-committed in the
/// DECISIONS.md 2026-08-23 P1-4 entry and never loosen.
///
/// These tests need the local consolidated checkpoint
/// (models/qwen3-1.7b-70d244cc.safetensors, produced by
/// tools/consolidate_shards.py — local-only, never committed) and skip
/// with a clear message when it is absent.
final class ActivationFixtureTests: XCTestCase {

    // Checkpoint pins, model loading, and blob decoding live in
    // SharedCheckpoint — one 7 GB fp32 model shared with the P1-5 logit
    // suite instead of one per test class.

    private func loadedModel() throws -> QwenModel {
        try SharedCheckpoint.model()
    }

    override func setUpWithError() throws {
        try SharedCheckpoint.skipUnlessCheckpointPresent()
    }

    // MARK: - Fixture loading

    private func floats(at relativePath: String, expectedCount: Int) throws -> [Float] {
        try SharedCheckpoint.floats(at: relativePath, expectedCount: expectedCount)
    }

    private func activation(_ name: String) throws -> [Float] {
        try floats(at: "activations/\(name).bin", expectedCount: 5 * 2048)
    }

    /// Prompt #1 (short_english) token ids from the committed tokenizer dump.
    private func promptIDs() throws -> [Int] {
        try SharedCheckpoint.promptFixture("short_english").inputIds
    }

    // MARK: - Gates (pre-committed, DECISIONS.md 2026-08-23 P1-4 entry)

    /// Hidden-state slices: |Δ| ≤ max(5e-5 · max|ref|, 1e-6) per element.
    private func assertWithinActivationGate(
        _ actual: [Float], _ reference: [Float], slice: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, reference.count,
                       "\(slice): element count", file: file, line: line)
        var maxAbsReference: Float = 0
        for r in reference { maxAbsReference = max(maxAbsReference, abs(r)) }
        let tolerance = max(5e-5 * maxAbsReference, 1e-6)
        var worst: Float = 0
        var worstIndex = -1
        for i in 0..<min(actual.count, reference.count) {
            let diff = abs(actual[i] - reference[i])
            if diff > worst { worst = diff; worstIndex = i }
        }
        XCTAssertLessThanOrEqual(
            worst, tolerance,
            "\(slice): worst |Δ| = \(worst) at index \(worstIndex) exceeds the "
            + "pre-committed gate \(tolerance) (5e-5 · max|ref| \(maxAbsReference))",
            file: file, line: line)
    }

    // MARK: - Per-module tests (isolated: reference input in, diff output)

    func testEmbeddingMatchesReferenceExactly() throws {
        let model = try loadedModel()
        let out = try model.embedding(promptIDs())
        let reference = try activation("embeddings_output")
        XCTAssertEqual(out.count, reference.count)
        // Exact ==, no tolerance: a row copy of exactly-upcast weights.
        for i in 0..<out.count where out[i] != reference[i] {
            return XCTFail(
                "embedding differs at index \(i): \(out[i]) != \(reference[i]) "
                + "(exact match required — DECISIONS.md P1-4 gate)")
        }
    }

    func testInputLayerNormMatchesReference() throws {
        let model = try loadedModel()
        let out = try model.blocks[0].inputNorm(try activation("embeddings_output"))
        assertWithinActivationGate(
            out, try activation("layer0_pre_attn_norm_output"),
            slice: "layer0_pre_attn_norm_output")
    }

    func testAttentionMatchesReference() throws {
        let model = try loadedModel()
        let out = try model.blocks[0].attention(
            try activation("layer0_pre_attn_norm_output"), seqLen: 5)
        assertWithinActivationGate(
            out, try activation("layer0_attn_output"), slice: "layer0_attn_output")
    }

    func testTransformerBlockMatchesReference() throws {
        let model = try loadedModel()
        let out = try model.blocks[0](try activation("embeddings_output"), seqLen: 5)
        assertWithinActivationGate(
            out, try activation("layer0_block_output"), slice: "layer0_block_output")
    }

    func testLayerStackMatchesReference() throws {
        let model = try loadedModel()
        let out = try model.hiddenStates(ids: promptIDs())
        assertWithinActivationGate(
            out, try activation("last_layer_output"), slice: "last_layer_output")
    }

    func testFinalNormMatchesReference() throws {
        let model = try loadedModel()
        let out = try model.finalNorm(try activation("last_layer_output"))
        assertWithinActivationGate(
            out, try activation("final_norm_output"), slice: "final_norm_output")
    }

    func testLMHeadMatchesReferenceLogits() throws {
        let model = try loadedModel()
        let vocab = model.config.vocabSize
        let logits = try model.lmHead(try activation("final_norm_output"), seqLen: 5)
        // Step-0 fixture holds the LAST position's full-vocab logits.
        let lastRow = Array(logits[(4 * vocab)..<(5 * vocab)])
        let reference = try floats(
            at: "prompts/short_english/logits_step0000.bin", expectedCount: vocab)
        // Logit-shaped output: the committed phase gate, 1e-3 absolute.
        var worst: Float = 0
        var worstIndex = -1
        for i in 0..<vocab {
            let diff = abs(lastRow[i] - reference[i])
            if diff > worst { worst = diff; worstIndex = i }
        }
        XCTAssertLessThanOrEqual(
            worst, 1e-3,
            "lm_head logits: worst |Δ| = \(worst) at token \(worstIndex) exceeds "
            + "the committed 1e-3 phase gate")
    }
}
