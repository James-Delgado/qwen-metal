import XCTest
import Foundation
import QwenMetalEngine

// MARK: - Suite rules (pre-committed: DECISIONS.md 2026-08-23 P1-5 entry)

/// Tie-aware argmax: exact top-1 match at margins >= epsilon; below it,
/// top-2 containment. Free function so the exemption path is unit-tested
/// even when the checkpoint-backed suite skips.
func tieAwareArgmaxHolds(
    ours: Int, top1: Int, top2: Int, margin: Double, epsilon: Double
) -> Bool {
    margin >= epsilon ? ours == top1 : (ours == top1 || ours == top2)
}

/// Scalar fingerprints in float64 over the fp32 logit vector — the manifest
/// protocol: max-subtracted logsumexp, mean, population std (numpy ddof=0).
func logitFingerprints(_ logits: [Float]) -> (logsumexp: Double, mean: Double, std: Double) {
    let values = logits.map(Double.init)
    let maxValue = values.max() ?? 0
    var sumExp = 0.0
    var sum = 0.0
    for value in values {
        sumExp += exp(value - maxValue)
        sum += value
    }
    let mean = sum / Double(values.count)
    var sumSquaredDeviation = 0.0
    for value in values {
        sumSquaredDeviation += (value - mean) * (value - mean)
    }
    let std = (sumSquaredDeviation / Double(values.count)).squareRoot()
    return (maxValue + log(sumExp), mean, std)
}

/// P1-5 full logit-match suite (phase-0-1.md correctness harness — the
/// Phase 1 exit gate): for each of the 5 fixture prompts, a 50-step greedy
/// generation teacher-forced with the REFERENCE argmax sequence (identical
/// token prefixes at every step; our own argmax is asserted separately),
/// checked at every step against the committed HF fp32 oracle.
///
/// All gates are pre-committed in the DECISIONS.md 2026-08-23 P1-5 entry —
/// derivations of the phase's 1e-3 logit gate — and never loosen.
final class LogitMatchSuiteTests: XCTestCase {

    // MARK: - Pre-committed gates

    private static let logitGate: Float = 1e-3 // the committed phase gate
    private static let logsumexpGate = 1e-3    // 1-Lipschitz in sup norm
    private static let meanGate = 1e-3         // mean of <=1e-3 deviations
    private static let stdGate = 2e-3          // 2-Lipschitz in sup norm
    private static let tieEpsilon = 2e-3       // 2 x the element gate

    private static let generationSteps = 50
    private static let checkpointSteps: Set<Int> = [0, 1, 24, 49]
    private static let topK = 64

    override func setUpWithError() throws {
        try SharedCheckpoint.skipUnlessCheckpointPresent()
    }

    // MARK: - steps.json schema

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
        let model = try SharedCheckpoint.model()
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

            // Full-vocab checkpoints: the committed 1e-3 phase gate.
            if Self.checkpointSteps.contains(step) {
                let full = try SharedCheckpoint.floats(
                    at: String(format: "prompts/%@/logits_step%04d.bin", prompt, step),
                    expectedCount: vocab)
                var worst: Float = 0
                var worstIndex = -1
                for i in 0..<vocab {
                    let diff = abs(logits[i] - full[i])
                    if diff > worst { worst = diff; worstIndex = i }
                }
                XCTAssertLessThanOrEqual(
                    worst, Self.logitGate,
                    "\(prompt) step \(step): full-vocab worst |Δ| \(worst) at token "
                    + "\(worstIndex) exceeds the committed 1e-3 gate")
            }

            // Every-step scalar fingerprints.
            let fingerprint = logitFingerprints(logits)
            XCTAssertLessThanOrEqual(
                abs(fingerprint.logsumexp - record.logsumexp), Self.logsumexpGate,
                "\(prompt) step \(step): logsumexp \(fingerprint.logsumexp) vs "
                + "reference \(record.logsumexp)")
            XCTAssertLessThanOrEqual(
                abs(fingerprint.mean - record.mean), Self.meanGate,
                "\(prompt) step \(step): mean \(fingerprint.mean) vs "
                + "reference \(record.mean)")
            XCTAssertLessThanOrEqual(
                abs(fingerprint.std - record.std), Self.stdGate,
                "\(prompt) step \(step): std \(fingerprint.std) vs "
                + "reference \(record.std)")

            // Every-step top-64: our logits at the reference indices.
            var worstTop: Float = 0
            var worstRank = -1
            for k in 0..<Self.topK {
                let flat = step * Self.topK + k
                let diff = abs(logits[top64Indices[flat]] - top64Values[flat])
                if diff > worstTop { worstTop = diff; worstRank = k }
            }
            XCTAssertLessThanOrEqual(
                worstTop, Self.logitGate,
                "\(prompt) step \(step): top-64 worst |Δ| \(worstTop) at rank "
                + "\(worstRank) exceeds the committed 1e-3 gate")

            // Tie-aware argmax.
            let ours = Argmax.firstIndex(logits)
            XCTAssertTrue(
                tieAwareArgmaxHolds(
                    ours: ours, top1: record.top1_id, top2: record.top2_id,
                    margin: record.margin_top1_top2, epsilon: Self.tieEpsilon),
                "\(prompt) step \(step): our argmax \(ours) vs reference top-1 "
                + "\(record.top1_id) / top-2 \(record.top2_id) "
                + "(margin \(record.margin_top1_top2))")

            // Teacher-force the REFERENCE token: identical prefixes at every
            // step, and steps after a tie-exempt flip stay comparable.
            ids.append(reference.argmax_token_ids[step])
        }
    }

    func testLogitMatchShortEnglish() throws { try runSuite("short_english") }
    func testLogitMatchMultiSentence() throws { try runSuite("multi_sentence") }
    func testLogitMatchCodeSnippet() throws { try runSuite("code_snippet") }
    func testLogitMatchNonASCII() throws { try runSuite("non_ascii") }
    func testLogitMatchChatTemplate() throws { try runSuite("chat_template") }
}

/// The tie-exemption path has no exercising fixture step today (minimum
/// recorded margin 4.8e-3 > epsilon 2e-3 — P1-1 DECISIONS entry), so it is
/// pinned synthetically. Separate class: runs even when the checkpoint-backed
/// suite skips.
final class TieAwareArgmaxRuleTests: XCTestCase {

    func testWideMarginRequiresExactTopOne() {
        XCTAssertTrue(tieAwareArgmaxHolds(
            ours: 7, top1: 7, top2: 3, margin: 0.5, epsilon: 2e-3))
        XCTAssertFalse(tieAwareArgmaxHolds(
            ours: 3, top1: 7, top2: 3, margin: 0.5, epsilon: 2e-3))
    }

    func testSubEpsilonMarginAllowsTopTwoContainment() {
        XCTAssertTrue(tieAwareArgmaxHolds(
            ours: 7, top1: 7, top2: 3, margin: 1e-3, epsilon: 2e-3))
        XCTAssertTrue(tieAwareArgmaxHolds(
            ours: 3, top1: 7, top2: 3, margin: 1e-3, epsilon: 2e-3))
        XCTAssertFalse(tieAwareArgmaxHolds(
            ours: 5, top1: 7, top2: 3, margin: 1e-3, epsilon: 2e-3))
    }

    func testMarginExactlyAtEpsilonIsNotExempt() {
        XCTAssertFalse(tieAwareArgmaxHolds(
            ours: 3, top1: 7, top2: 3, margin: 2e-3, epsilon: 2e-3))
    }
}
