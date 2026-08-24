import XCTest
import Foundation
@testable import QwenMetalEngine

/// P2-4 free-running divergence REPORT (not a gate — DECISIONS.md 2026-08-23
/// Phase 2 gates entry): 128 free-running greedy steps × 5 prompts, GPU
/// pipeline vs CPU reference, each self-feeding its OWN argmax. With
/// per-logit deviation legitimately up to 2⁻⁵·M, any near-tie step can flip
/// and permanently fork the trajectory, so the output here is recorded in
/// DECISIONS.md (first-divergence index + both texts), never asserted.
///
/// Opt-in via QWEN_FREE_RUN_REPORT=1 (the CPU side re-forwards the full
/// sequence every step — minutes of runtime; use `swift test -c release
/// --filter FreeRunReport`). No stop set: trajectories run the full 128
/// steps so divergence position is always comparable.
final class FreeRunReportTests: XCTestCase {

    private static let steps = 128
    private static let prompts = [
        "short_english", "multi_sentence", "code_snippet",
        "non_ascii", "chat_template",
    ]

    func testFreeRunDivergenceReport() async throws {
        guard ProcessInfo.processInfo.environment["QWEN_FREE_RUN_REPORT"] == "1" else {
            throw XCTSkip(
                "free-run divergence report is opt-in: set QWEN_FREE_RUN_REPORT=1 "
                + "(report recorded in DECISIONS.md at P2-4 — not a gate)")
        }
        try SharedGPUModel.skipUnlessReady()
        let gpu = try SharedGPUModel.model()
        let cpu = try SharedCheckpoint.model()
        let tokenizer = try await TextTokenizer(
            modelFolder: SharedCheckpoint.modelsDir)

        print("=== P2-4 free-running greedy divergence report "
            + "(128 steps × 5 prompts, GPU vs CPU reference) ===")
        for prompt in Self.prompts {
            let ids = try SharedCheckpoint.promptFixture(prompt).inputIds
            gpu.reset()
            let gpuLoop = DecodeLoop(model: gpu, maxContext: SharedGPUModel.maxContext)
            let gpuTokens = try gpuLoop.generate(
                promptIds: ids, maxNewTokens: Self.steps, eosTokenIds: [])
            let cpuLoop = DecodeLoop(model: cpu, maxContext: SharedGPUModel.maxContext)
            let cpuTokens = try cpuLoop.generate(
                promptIds: ids, maxNewTokens: Self.steps, eosTokenIds: [])
            XCTAssertEqual(gpuTokens.count, Self.steps)
            XCTAssertEqual(cpuTokens.count, Self.steps)

            let firstDivergence = zip(gpuTokens, cpuTokens)
                .enumerated().first { $1.0 != $1.1 }?.offset
            print("--- prompt: \(prompt)")
            if let index = firstDivergence {
                print("first divergence: step \(index) "
                    + "(gpu \(gpuTokens[index]) vs cpu \(cpuTokens[index]))")
            } else {
                print("first divergence: none (all \(Self.steps) tokens identical)")
            }
            print("gpu text: \(tokenizer.decode(gpuTokens, skipSpecialTokens: false))")
            print("cpu text: \(tokenizer.decode(cpuTokens, skipSpecialTokens: false))")
        }
        gpu.reset()
    }
}
