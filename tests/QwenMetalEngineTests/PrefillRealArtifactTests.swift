import XCTest
@testable import QwenMetalEngine
import Metal

// MARK: - Real-artifact smoke + free-run report (edge 8, artifact-gated)

/// Skip-unless-artifact smoke of the tiled prefill over the REAL packed
/// checkpoint, plus the opt-in free-running divergence REPORT from tiled
/// prefill (128 × 5 vs the CPU-quant reference — REPORTED, never gated;
/// spec D6). The binding Tier-M/E re-verification with the tiled path
/// engaged is P5-4's deliverable.
final class PrefillRealArtifactTests: XCTestCase {

    private static let prompts = [
        "short_english", "multi_sentence", "code_snippet",
        "non_ascii", "chat_template",
    ]

    private func makeTiledRealModel() throws -> GPUModel {
        let packed = try PackedCheckpoint(
            path: SharedQuantModel.packedURL.path,
            expectedRevision: SharedCheckpoint.pinnedRevision)
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        return try GPUModel(
            packed: packed, config: config,
            context: try SharedGPUModel.metalContext(),
            maxContext: SharedQuantGPUModel.maxContext,
            prefillPath: .tiled, prefillChunkSize: 128)
    }

    func testTiledPrefillSmokeOnRealArtifact() throws {
        try SharedQuantGPUModel.skipUnlessReady()
        let model = try makeTiledRealModel()
        let ids = try SharedCheckpoint.promptFixture("short_english").inputIds

        let first = try model.nextGreedyToken(ids: ids)
        XCTAssertTrue(first >= 0 && first < model.vocabSize)
        let span = try XCTUnwrap(model.lastCallSpan)
        XCTAssertEqual(span.stepCount, ids.count)
        XCTAssertGreaterThan(span.gpuSeconds, 0)
        XCTAssertGreaterThanOrEqual(span.wallSeconds, span.gpuSeconds)

        let tokens = try DecodeLoop(
            model: model, maxContext: SharedQuantGPUModel.maxContext
        ).generateTokens(promptIds: ids, maxNewTokens: 8, eosTokenIds: [])
        XCTAssertEqual(tokens.count, 8)
    }

    /// Edge test 9 on the REAL artifact (P5-4): the production DEFAULT
    /// (tiled, C resolved to min(512, maxContext)) and the explicit
    /// sequential option both load, both process the same prompt with a
    /// per-engine-exact span, both continue into the unchanged fused
    /// decode, and DispatchCounter tells them apart: sequential measures
    /// (P−1)·197 + 200 on the selecting path (the P5-1 structural
    /// cross-check), tiled a single-chunk count that is far smaller and
    /// independent of P since PF-1 (one batched SDPA dispatch per layer).
    func testDefaultTiledAndSequentialBothLoadOnRealArtifact() throws {
        try SharedQuantGPUModel.skipUnlessReady()
        let packed = try PackedCheckpoint(
            path: SharedQuantModel.packedURL.path,
            expectedRevision: SharedCheckpoint.pinnedRevision)
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        let context = try SharedGPUModel.metalContext()
        let tiled = try GPUModel(
            packed: packed, config: config, context: context,
            maxContext: SharedQuantGPUModel.maxContext)
        let sequential = try GPUModel(
            packed: packed, config: config, context: context,
            maxContext: SharedQuantGPUModel.maxContext,
            prefillPath: .sequential)
        XCTAssertEqual(tiled.prefillPath, .tiled)
        XCTAssertEqual(tiled.prefillChunkSize, SharedQuantGPUModel.maxContext)
        XCTAssertEqual(sequential.prefillPath, .sequential)

        let ids = try SharedCheckpoint.promptFixture("short_english").inputIds
        let p = ids.count
        let firstTiled = try tiled.nextGreedyToken(ids: ids)
        let firstSequential = try sequential.nextGreedyToken(ids: ids)
        XCTAssertTrue((0..<config.vocabSize).contains(firstTiled))
        XCTAssertTrue((0..<config.vocabSize).contains(firstSequential))

        let tiledSpan = try XCTUnwrap(tiled.lastCallSpan)
        let sequentialSpan = try XCTUnwrap(sequential.lastCallSpan)
        XCTAssertEqual(tiledSpan.stepCount, p)
        XCTAssertEqual(sequentialSpan.stepCount, p)
        XCTAssertEqual(sequentialSpan.dispatchCount, (p - 1) * 197 + 200)
        XCTAssertEqual(
            tiledSpan.dispatchCount,
            1 + 28 * 14 + 3 + 1,
            "one chunk: gather + 28·14 (PF-1: one batched SDPA per layer) + "
            + "logits tail 3 + argmax — chunk-size independent")
        XCTAssertLessThan(tiledSpan.dispatchCount, sequentialSpan.dispatchCount)
        XCTAssertGreaterThanOrEqual(tiledSpan.wallSeconds, tiledSpan.gpuSeconds)
        XCTAssertGreaterThanOrEqual(
            sequentialSpan.wallSeconds, sequentialSpan.gpuSeconds)

        for model in [tiled, sequential] {
            model.reset()
            let tokens = try DecodeLoop(
                model: model, maxContext: SharedQuantGPUModel.maxContext
            ).generateTokens(promptIds: ids, maxNewTokens: 4, eosTokenIds: [])
            XCTAssertEqual(tokens.count, 4)
            XCTAssertEqual(model.lastStepDispatchCount, 200,
                           "\(model.prefillPath): decode after the prompt is the "
                           + "unchanged fused selecting step")
        }
    }

    /// QWEN_PREFILL_CHUNK_SWEEP=1 opt-in: the spec D2 chunk-size selection
    /// harness — warm tiled prefill of the pinned prefill-summarize prompt
    /// (852 HF tokens) at candidate C values, median of 3 warm repeats per
    /// C (D1 span metric). DIAGNOSTIC ONLY: Mac numbers pick the default C
    /// (recorded in DECISIONS.md); they are never benchmark rows. Run
    /// release-mode: `swift test -c release --filter ChunkSweep`.
    func testPrefillChunkSizeSweep() async throws {
        guard ProcessInfo.processInfo
            .environment["QWEN_PREFILL_CHUNK_SWEEP"] == "1" else {
            throw XCTSkip(
                "chunk-size sweep is opt-in: set QWEN_PREFILL_CHUNK_SWEEP=1 "
                + "(diagnostic — picks the default C, recorded in DECISIONS.md)")
        }
        try SharedQuantGPUModel.skipUnlessReady()
        let promptURL = SharedCheckpoint.repoRoot
            .appendingPathComponent("benchmarks/prompts/rendered/prefill-summarize.rendered.txt")
        let text = try String(contentsOf: promptURL, encoding: .utf8)
        let tokenizer = try await TextTokenizer(
            modelFolder: SharedCheckpoint.modelsDir)
        let ids = tokenizer.encode(text)
        print("chunk sweep: prompt tokens = \(ids.count)")

        let packed = try PackedCheckpoint(
            path: SharedQuantModel.packedURL.path,
            expectedRevision: SharedCheckpoint.pinnedRevision)
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        var lines = ["=== P5-3 prefill chunk-size sweep (Mac, diagnostic; "
            + "prompt \(ids.count) tokens, warm median of 3) ==="]
        for chunk in [128, 256, 512, 768] {
            let model = try GPUModel(
                packed: packed, config: config,
                context: try SharedGPUModel.metalContext(),
                maxContext: 4096, prefillPath: .tiled,
                prefillChunkSize: chunk)
            var rates: [Double] = []
            _ = try model.nextGreedyToken(ids: ids)  // cold warmup, discarded
            for _ in 0..<3 {
                model.reset()
                _ = try model.nextGreedyToken(ids: ids)
                let span = try XCTUnwrap(model.lastCallSpan)
                XCTAssertEqual(span.stepCount, ids.count)
                rates.append(Double(ids.count) / span.wallSeconds)
            }
            let sorted = rates.sorted()
            let line = String(
                format: "C=%d: median %.2f tok/s (range %.2f-%.2f), "
                    + "scratch %.1f MiB",
                chunk, sorted[1], sorted[0], sorted[2],
                Double(GPUModel.prefillScratchBytes(
                    config: config, chunkSize: chunk)) / 1_048_576)
            print(line)
            lines.append(line)
        }
        if let path = ProcessInfo.processInfo
            .environment["QWEN_PREFILL_CHUNK_SWEEP_FILE"] {
            try lines.joined(separator: "\n").appending("\n")
                .write(toFile: path, atomically: true, encoding: .utf8)
        }
    }

    /// QWEN_FREE_RUN_REPORT=1 opt-in (QuantFreeRunReportTests protocol):
    /// 128 free-running greedy steps × 5 prompts from a TILED prefill vs
    /// the CPU-quant reference, first divergence + both texts reported
    /// (QWEN_FREE_RUN_REPORT_FILE also captures the lines).
    func testTiledPrefillFreeRunDivergenceReport() async throws {
        guard ProcessInfo.processInfo.environment["QWEN_FREE_RUN_REPORT"] == "1"
        else {
            throw XCTSkip(
                "free-run divergence report is opt-in: set "
                + "QWEN_FREE_RUN_REPORT=1 (recorded in DECISIONS.md — not a gate)")
        }
        try SharedQuantGPUModel.skipUnlessReady()
        let gpu = try makeTiledRealModel()
        let cpu = try SharedQuantModel.model()
        let tokenizer = try await TextTokenizer(
            modelFolder: SharedCheckpoint.modelsDir)

        var report: [String] = []
        func emit(_ line: String) {
            print(line)
            report.append(line)
        }
        emit("=== P5-3 free-running greedy divergence report from TILED "
            + "prefill (128 steps × 5 prompts, C=128, vs CPU-quant) ===")
        for prompt in Self.prompts {
            let ids = try SharedCheckpoint.promptFixture(prompt).inputIds
            gpu.reset()
            let gpuTokens = try DecodeLoop(
                model: gpu, maxContext: SharedQuantGPUModel.maxContext
            ).generate(promptIds: ids, maxNewTokens: 128, eosTokenIds: [])
            let cpuTokens = try DecodeLoop(
                model: cpu, maxContext: SharedQuantGPUModel.maxContext
            ).generate(promptIds: ids, maxNewTokens: 128, eosTokenIds: [])
            let firstDivergence = zip(gpuTokens, cpuTokens)
                .enumerated().first { $1.0 != $1.1 }?.offset
            emit("--- prompt: \(prompt)")
            if let index = firstDivergence {
                emit("first divergence: step \(index) "
                    + "(tiled-prefill gpu \(gpuTokens[index]) vs "
                    + "cpu-quant \(cpuTokens[index]))")
            } else {
                emit("first divergence: none (all 128 tokens identical)")
            }
            emit("gpu text: \(tokenizer.decode(gpuTokens, skipSpecialTokens: false))")
            emit("cpu text: \(tokenizer.decode(cpuTokens, skipSpecialTokens: false))")
        }
        gpu.reset()

        if let path = ProcessInfo.processInfo
            .environment["QWEN_FREE_RUN_REPORT_FILE"] {
            try report.joined(separator: "\n").appending("\n")
                .write(toFile: path, atomically: true, encoding: .utf8)
        }
    }
}
