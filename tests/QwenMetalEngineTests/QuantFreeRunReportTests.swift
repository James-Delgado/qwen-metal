import XCTest
import Foundation
@testable import QwenMetalEngine

/// P3-5 free-running divergence REPORT (not a gate — DECISIONS.md 2026-08-25
/// Phase 3 gates entry, Phase 2 rationale standing): 128 free-running greedy
/// steps × 5 prompts, GPU-quant pipeline vs the CPU-quant reference, each
/// self-feeding its OWN argmax. With per-logit deviation legitimately up to
/// 2⁻⁵·M, any near-tie step can flip and permanently fork the trajectory, so
/// the output is recorded in DECISIONS.md (first-divergence index + both
/// texts), never asserted.
///
/// Opt-in via QWEN_FREE_RUN_REPORT=1 (the CPU-quant side re-forwards the
/// full sequence every step — use `swift test -c release --filter
/// QuantFreeRunReport`). No stop set: trajectories run the full 128 steps so
/// divergence position is always comparable.
///
/// The current swift-test runner captures in-test stdout and does not replay
/// it, so `print` alone silently loses the report; set
/// QWEN_FREE_RUN_REPORT_FILE=<path> to ALSO write the report lines there —
/// the artifact the DECISIONS.md entry quotes.
final class QuantFreeRunReportTests: XCTestCase {

    private static let steps = 128
    private static let prompts = [
        "short_english", "multi_sentence", "code_snippet",
        "non_ascii", "chat_template",
    ]

    func testQuantFreeRunDivergenceReport() async throws {
        guard ProcessInfo.processInfo.environment["QWEN_FREE_RUN_REPORT"] == "1" else {
            throw XCTSkip(
                "free-run divergence report is opt-in: set QWEN_FREE_RUN_REPORT=1 "
                + "(report recorded in DECISIONS.md at P3-5 — not a gate)")
        }
        try SharedQuantGPUModel.skipUnlessReady()
        let gpu = try SharedQuantGPUModel.model()
        let cpu = try SharedQuantModel.model()
        let tokenizer = try await TextTokenizer(
            modelFolder: SharedCheckpoint.modelsDir)

        var report: [String] = []
        func emit(_ line: String) {
            print(line)
            report.append(line)
        }

        emit("=== P3-5 free-running greedy divergence report "
            + "(128 steps × 5 prompts, GPU-quant vs CPU-quant reference) ===")
        for prompt in Self.prompts {
            let ids = try SharedCheckpoint.promptFixture(prompt).inputIds
            gpu.reset()
            let gpuLoop = DecodeLoop(
                model: gpu, maxContext: SharedQuantGPUModel.maxContext)
            let gpuTokens = try gpuLoop.generate(
                promptIds: ids, maxNewTokens: Self.steps, eosTokenIds: [])
            let cpuLoop = DecodeLoop(
                model: cpu, maxContext: SharedQuantGPUModel.maxContext)
            let cpuTokens = try cpuLoop.generate(
                promptIds: ids, maxNewTokens: Self.steps, eosTokenIds: [])
            XCTAssertEqual(gpuTokens.count, Self.steps)
            XCTAssertEqual(cpuTokens.count, Self.steps)

            let firstDivergence = zip(gpuTokens, cpuTokens)
                .enumerated().first { $1.0 != $1.1 }?.offset
            emit("--- prompt: \(prompt)")
            if let index = firstDivergence {
                emit("first divergence: step \(index) "
                    + "(gpu-quant \(gpuTokens[index]) vs cpu-quant \(cpuTokens[index]))")
            } else {
                emit("first divergence: none (all \(Self.steps) tokens identical)")
            }
            emit("gpu-quant text: \(tokenizer.decode(gpuTokens, skipSpecialTokens: false))")
            emit("cpu-quant text: \(tokenizer.decode(cpuTokens, skipSpecialTokens: false))")
        }
        gpu.reset()

        if let path = ProcessInfo.processInfo
            .environment["QWEN_FREE_RUN_REPORT_FILE"] {
            try report.joined(separator: "\n").appending("\n")
                .write(toFile: path, atomically: true, encoding: .utf8)
        }
    }
}
