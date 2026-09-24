import XCTest
import Foundation
@testable import QwenMetalEngine

/// P6-1 Mac sanity run of the energy mode (spec task note: "Mac sanity run
/// of the energy mode with a short duration bound (no numbers of record)").
/// Drives the REAL packed q4g64 GPU pipeline (the shared model, maxContext
/// 256) through `EnergyLoop` + `BenchGenerationRunner` exactly as the app
/// does — once ended by the safety bound, once by a scripted operator stop
/// — and renders the Phase 6 text export + timeline JSON. Mac numbers are
/// PROVISIONAL dev-loop sanity, never rows.
///
/// Opt-in via QWEN_ENERGY_SANITY=1; set QWEN_ENERGY_SANITY_FILE=<path> to
/// also write both exports there (the swift-test runner swallows stdout).
final class EnergyLoopSanityTests: XCTestCase {

    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["QWEN_ENERGY_SANITY"] == "1" else {
            throw XCTSkip("energy-mode Mac sanity run is opt-in: set QWEN_ENERGY_SANITY=1")
        }
        try SharedQuantGPUModel.skipUnlessReady()
    }

    private func runner(_ gpu: GPUModel) -> BenchGenerationRunner {
        BenchGenerationRunner(
            gpuModel: gpu, maxContext: SharedQuantGPUModel.maxContext,
            eosTokenIds: [])
    }

    func testEnergyModeSanityOnTheRealPipeline() throws {
        let gpu = try SharedQuantGPUModel.model()
        let ids = try SharedCheckpoint.promptFixture("multi_sentence").inputIds
        let maxNew = SharedQuantGPUModel.maxContext - ids.count
        var report: [String] = []

        // 1. Safety bound ends the loop (no operator stop): ≥ 1 generation,
        //    the bound is honored at a token boundary, totals add up.
        gpu.reset()
        let bounded = try EnergyLoop(maxDurationSeconds: 12).run(
            operatorStop: { false },
            generate: { shouldStop in
                try self.runner(gpu).run(
                    promptIds: ids, maxNewTokens: maxNew,
                    shouldStop: shouldStop).metrics
            })
        XCTAssertEqual(bounded.endedBy, .durationBound)
        XCTAssertGreaterThanOrEqual(bounded.generations.count, 1)
        XCTAssertEqual(
            bounded.totalGeneratedTokens,
            bounded.generations.reduce(0) { $0 + $1.generatedTokenCount })
        XCTAssertGreaterThanOrEqual(bounded.cycleWallSeconds, 12)

        // 2. Operator stop mid-generation: ends at the next token boundary,
        //    the final generation is flagged, and a report still renders.
        gpu.reset()
        var tokensSeen = 0
        var operatorStopped = false
        let stopped = try EnergyLoop().run(
            operatorStop: { operatorStopped },
            generate: { shouldStop in
                try self.runner(gpu).run(
                    promptIds: ids, maxNewTokens: maxNew,
                    shouldStop: shouldStop,
                    onToken: { _, _ in
                        tokensSeen += 1
                        if tokensSeen == maxNew + 40 { operatorStopped = true }
                    }).metrics
            })
        gpu.reset()
        XCTAssertEqual(stopped.endedBy, .operatorStop)
        XCTAssertEqual(stopped.generations.count, 2)
        XCTAssertTrue(stopped.lastGenerationTruncated)
        XCTAssertEqual(stopped.generations.last?.stopReason, .stopRequested)
        XCTAssertEqual(stopped.totalGeneratedTokens, tokensSeen)

        let export = BenchmarkReport(
            dateStamp: "mac-sanity", deviceLabel: "Mac (PROVISIONAL sanity)",
            osVersion: "n/a", batteryHealthNote: "", coldOrWarmNote: "warm",
            residency: .mmap, weightsFormat: .q4g64,
            kernelPath: gpu.kernelPath, prefillPath: gpu.prefillPath,
            prefillChunkSize: gpu.prefillChunkSize,
            prefillAttention: gpu.prefillAttention,
            promptName: "multi_sentence (fixture)", promptTokenCount: ids.count,
            mode: .energy, energy: stopped,
            physFootprintBytes: MemoryFootprint.currentPhysFootprintBytes(),
            batteryStateOfCharge: BatteryStateOfCharge(
                operatorStartNote: "", operatorEndNote: "",
                programmaticStartPercent: nil, programmaticEndPercent: nil))
        let text = export.exportText()
        let json = try XCTUnwrap(export.timelineJSON())
        XCTAssertTrue(text.contains("energy cycle: 2 generations — ended by operator stop"), text)
        XCTAssertTrue(text.contains("(PROVISIONAL)"))
        let decoded = try JSONDecoder().decode(TimelineExport.self, from: Data(json.utf8))
        XCTAssertEqual(decoded.generations.count, 2)

        report.append("=== P6-1 energy-mode Mac sanity (PROVISIONAL, never a row) ===")
        report.append(String(
            format: "bounded run: %d generations, %d tokens, cycle wall %.1f s, ended by %@",
            bounded.generations.count, bounded.totalGeneratedTokens,
            bounded.cycleWallSeconds, bounded.endedBy.rawValue))
        report.append("--- operator-stop run text export ---")
        report.append(text)
        report.append("--- operator-stop run timeline JSON ---")
        report.append(json)
        for line in report { print(line) }
        if let path = ProcessInfo.processInfo.environment["QWEN_ENERGY_SANITY_FILE"] {
            try report.joined(separator: "\n").write(
                toFile: path, atomically: true, encoding: .utf8)
        }
    }
}
