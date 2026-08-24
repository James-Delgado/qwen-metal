import XCTest
import Foundation
@testable import QwenMetalEngine

/// P2-4 Tier-E teacher-forced logit suite: the P1-5 machinery pointed at the
/// GPU pipeline — same 5 fixture prompts, same 50 reference-argmax
/// teacher-forced steps, same manifest protocol — at the pre-committed
/// Phase 2 gates (DECISIONS.md 2026-08-23 "Phase 2 gates pre-committed"):
///
///   full-vocab checkpoints (steps {0,1,24,49})   |Δ| ≤ 2⁻⁵·M_step
///   per-step top-64 at reference indices          |Δ| ≤ 2⁻⁵·M64
///   per-step fingerprints (float64)               lse/mean ≤ 2⁻⁵·M64, std ≤ 2⁻⁴·M64
///   top-1 agreement, all 250 steps                exact where margin ≥ 2⁻⁴·M64,
///                                                 else top-2 containment — no
///                                                 step is unasserted
///
/// M_step = max|ref| over that step's full-vocab fixture; M64 = max|ref
/// top-64 value| at that step. Gates never loosen (hard rule 6). Decode is
/// INCREMENTAL through the KV cache — teacher-forcing extends the cached
/// prefix by one reference token per step, so this suite also exercises the
/// cache across all 250 steps.
final class GPULogitSuiteTests: XCTestCase {

    // MARK: - Pre-committed gate constants

    private static let relE = exp2(Float(-5))     // 2⁻⁵ (checkpoint/top-64/lse/mean)
    private static let relStd = exp2(Float(-4))   // 2⁻⁴ (std is 2-Lipschitz)
    private static let tieRel = exp2(-4.0)        // ε_tie = 2·2⁻⁵·M64 = 2⁻⁴·M64

    private static let generationSteps = 50
    private static let checkpointSteps: Set<Int> = [0, 1, 24, 49]
    private static let topK = 64

    override func setUpWithError() throws {
        try SharedGPUModel.skipUnlessReady()
    }

    // MARK: - steps.json schema (fixture manifest protocol)

    private struct StepRecord: Decodable {
        let step: Int
        let logsumexp: Double
        let mean: Double
        let std: Double
        let top1_id: Int
        let top2_id: Int
        let margin_top1_top2: Double
    }

    private struct StepsFile: Decodable {
        let prompt_id: String
        let argmax_token_ids: [Int]
        let steps: [StepRecord]
    }

    private func stepsFile(_ prompt: String) throws -> StepsFile {
        let url = SharedCheckpoint.fixturesDir
            .appendingPathComponent("prompts/\(prompt)/steps.json")
        return try JSONDecoder().decode(StepsFile.self, from: Data(contentsOf: url))
    }

    // MARK: - The suite

    private func runSuite(_ prompt: String) throws {
        let model = try SharedGPUModel.model()
        model.reset()
        let vocab = model.config.vocabSize
        let promptIds = try SharedCheckpoint.promptFixture(prompt).inputIds
        let reference = try stepsFile(prompt)
        XCTAssertEqual(reference.prompt_id, prompt)
        XCTAssertEqual(reference.argmax_token_ids.count, Self.generationSteps)
        XCTAssertEqual(reference.steps.count, Self.generationSteps)
        let top64Values = try SharedCheckpoint.floats(
            at: "prompts/\(prompt)/top64_values.bin",
            expectedCount: Self.generationSteps * Self.topK)
        let top64Indices = try SharedCheckpoint.int32s(
            at: "prompts/\(prompt)/top64_indices.bin",
            expectedCount: Self.generationSteps * Self.topK)

        var ids = promptIds
        for step in 0..<Self.generationSteps {
            let logits = try model.lastPositionLogits(ids: ids)
            XCTAssertEqual(logits.count, vocab)
            let record = reference.steps[step]

            // Per-step M64 (max |ref top-64 value|) scales every gate below.
            var m64: Float = 0
            for k in 0..<Self.topK {
                m64 = max(m64, abs(top64Values[step * Self.topK + k]))
            }

            // Full-vocab checkpoints at 2⁻⁵ · M_step.
            if Self.checkpointSteps.contains(step) {
                let full = try SharedCheckpoint.floats(
                    at: String(format: "prompts/%@/logits_step%04d.bin", prompt, step),
                    expectedCount: vocab)
                var mStep: Float = 0
                for value in full { mStep = max(mStep, abs(value)) }
                let gate = Self.relE * mStep
                var worst: Float = 0
                var worstIndex = -1
                for i in 0..<vocab {
                    let diff = abs(logits[i] - full[i])
                    if diff > worst { worst = diff; worstIndex = i }
                }
                XCTAssertLessThanOrEqual(
                    worst, gate,
                    "\(prompt) step \(step): full-vocab worst |Δ| \(worst) at "
                    + "token \(worstIndex) exceeds 2⁻⁵·M_step = \(gate)")
            }

            // Every-step scalar fingerprints (float64, vs the manifest).
            let fingerprint = logitFingerprints(logits)
            let scalarGate = Double(Self.relE * m64)
            XCTAssertLessThanOrEqual(
                abs(fingerprint.logsumexp - record.logsumexp), scalarGate,
                "\(prompt) step \(step): logsumexp \(fingerprint.logsumexp) vs "
                + "reference \(record.logsumexp) (gate 2⁻⁵·M64 = \(scalarGate))")
            XCTAssertLessThanOrEqual(
                abs(fingerprint.mean - record.mean), scalarGate,
                "\(prompt) step \(step): mean \(fingerprint.mean) vs "
                + "reference \(record.mean) (gate 2⁻⁵·M64 = \(scalarGate))")
            XCTAssertLessThanOrEqual(
                abs(fingerprint.std - record.std), Double(Self.relStd * m64),
                "\(prompt) step \(step): std \(fingerprint.std) vs "
                + "reference \(record.std) (gate 2⁻⁴·M64)")

            // Every-step top-64: our logits at the reference indices.
            let topGate = Self.relE * m64
            var worstTop: Float = 0
            var worstRank = -1
            for k in 0..<Self.topK {
                let flat = step * Self.topK + k
                let diff = abs(logits[top64Indices[flat]] - top64Values[flat])
                if diff > worstTop { worstTop = diff; worstRank = k }
            }
            XCTAssertLessThanOrEqual(
                worstTop, topGate,
                "\(prompt) step \(step): top-64 worst |Δ| \(worstTop) at rank "
                + "\(worstRank) exceeds 2⁻⁵·M64 = \(topGate)")

            // Tie-aware top-1 agreement at ε_tie = 2⁻⁴·M64 (OV#5/#11).
            let ours = Argmax.firstIndex(logits)
            XCTAssertTrue(
                tieAwareArgmaxHolds(
                    ours: ours, top1: record.top1_id, top2: record.top2_id,
                    margin: record.margin_top1_top2,
                    epsilon: Self.tieRel * Double(m64)),
                "\(prompt) step \(step): our argmax \(ours) vs reference top-1 "
                + "\(record.top1_id) / top-2 \(record.top2_id) "
                + "(margin \(record.margin_top1_top2), ε \(Self.tieRel * Double(m64)))")

            // Teacher-force the REFERENCE token (identical prefixes at every
            // step) — the incremental path appends exactly one token.
            ids.append(reference.argmax_token_ids[step])
        }
        model.reset()
    }

    func testGPULogitMatchShortEnglish() throws { try runSuite("short_english") }
    func testGPULogitMatchMultiSentence() throws { try runSuite("multi_sentence") }
    func testGPULogitMatchCodeSnippet() throws { try runSuite("code_snippet") }
    func testGPULogitMatchNonASCII() throws { try runSuite("non_ascii") }
    func testGPULogitMatchChatTemplate() throws { try runSuite("chat_template") }
}
