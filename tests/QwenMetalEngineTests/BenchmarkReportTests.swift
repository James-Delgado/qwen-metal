import XCTest
import QwenMetalEngine

/// P2-6: the row-field export the iOS shell shares out. These pin the fields
/// the phase0-runbook row format needs — PROVISIONAL marker, residency,
/// prompt token count, dual-timing medians, dispatches/token, canonical
/// window labeling, per-generation sustained sequence, operator placeholders.
final class BenchmarkReportTests: XCTestCase {

    /// Builds a real DecodeTimingSummary through the collector (its
    /// memberwise init is deliberately not public).
    private func syntheticMetrics(
        tokens: Int, dispatches: Int = 591
    ) -> GenerationMetrics {
        var collector = DecodeTimingCollector()
        for i in 0..<tokens {
            let base = Double(i)
            collector.append(TokenStepRecord(
                timing: DispatchTiming(
                    wallStart: base, wallEnd: base + 0.25,
                    gpuStart: base, gpuEnd: base + 0.2),
                dispatchCount: dispatches))
        }
        return GenerationMetrics(
            promptTokenCount: 84, generatedTokenCount: tokens,
            wallSeconds: Double(tokens) * 0.25,
            prefillSeconds: 21.0, stopReason: .maxNewTokens,
            timing: collector.summary(),
            overallTokensPerSecond: collector.overallTokensPerSecond(),
            canonicalWindowTokensPerSecond:
                collector.canonicalWindowTokensPerSecond())
    }

    private func report(
        mode: BenchmarkReport.Mode,
        burst: GenerationMetrics? = nil,
        sustained: SustainedLoopResult? = nil,
        batteryNote: String = "88%",
        physFootprint: UInt64? = 4_294_967_296
    ) -> BenchmarkReport {
        BenchmarkReport(
            dateStamp: "2026-08-25", deviceLabel: "iPhone 15 Pro",
            osVersion: "19.0", batteryHealthNote: batteryNote,
            coldOrWarmNote: "warm", residency: .mmap,
            promptName: "decode-essay", promptTokenCount: 84, mode: mode,
            burst: burst, sustained: sustained,
            physFootprintBytes: physFootprint)
    }

    func testBurstExportCarriesTheRowFields() throws {
        let text = report(
            mode: .burst, burst: syntheticMetrics(tokens: 640)).exportText()

        XCTAssertTrue(text.contains("PROVISIONAL"))
        XCTAssertTrue(text.contains("date: 2026-08-25"))
        XCTAssertTrue(text.contains("iPhone 15 Pro (iOS 19.0)"))
        XCTAssertTrue(text.contains("battery health: 88%"))
        XCTAssertTrue(text.contains("residency mmap"))
        XCTAssertTrue(text.contains("decode-essay (84 prompt tokens)"))
        XCTAssertTrue(text.contains("mode: burst | cold/warm: warm"))
        XCTAssertTrue(text.contains("sampling: greedy"))
        XCTAssertTrue(text.contains("Metal API validation"))
        XCTAssertTrue(text.contains("generated: 640 tokens"))
        // Prefill honesty note (spec D6): 84 tokens / 21 s = 4.00 tok/s.
        XCTAssertTrue(text.contains(
            "prefill: 84 tokens in 21.00 s (4.00 tok/s"))
        XCTAssertTrue(text.contains("includes the first generated token's forward"))
        // Dual timing (hard rule 7): GPU, wall, AND the wall−GPU overhead.
        XCTAssertTrue(text.contains("median GPU 200.00 ms"))
        XCTAssertTrue(text.contains("median wall 250.00 ms"))
        XCTAssertTrue(text.contains("median wall-GPU 50.000 ms"))
        XCTAssertTrue(text.contains("591 dispatches/token"))
        // 640 ≥ 512 → the canonical window rate is real: 384 tokens over
        // (511.25 − 127.25) s = 1.00 tok/s.
        XCTAssertTrue(text.contains(
            "canonical window (tokens 128-512) 1.00 tok/s"))
        XCTAssertTrue(text.contains("phys_footprint"))
        XCTAssertTrue(text.contains("4096.0 MB"))
        XCTAssertTrue(text.contains("Xcode gauge is the metric of record"))
    }

    // MARK: - (P3-5) rows record the weights format

    func testDefaultExportIsBF16Phase2() throws {
        let text = report(
            mode: .burst, burst: syntheticMetrics(tokens: 64)).exportText()
        XCTAssertTrue(text.contains("Phase 2 row export"))
        XCTAssertTrue(text.contains("weights bf16"))
        XCTAssertTrue(text.contains("naive fp16 GPU"))
    }

    func testQ4G64ExportRecordsFormatAndPhase() throws {
        let text = BenchmarkReport(
            dateStamp: "2026-09-02", deviceLabel: "iPhone 15 Pro",
            osVersion: "19.0", batteryHealthNote: "88%",
            coldOrWarmNote: "warm", residency: .mmap, weightsFormat: .q4g64,
            promptName: "decode-essay", promptTokenCount: 84, mode: .burst,
            burst: syntheticMetrics(tokens: 64)).exportText()
        XCTAssertTrue(text.contains("Phase 3 row export"))
        XCTAssertTrue(text.contains("weights q4g64"))
        XCTAssertTrue(text.contains("q4g64 fused-dequant GPU"))
        XCTAssertTrue(text.contains("residency mmap"))
    }

    func testBurstExportLabelsWindowUnavailableBelow512() throws {
        let text = report(
            mode: .burst, burst: syntheticMetrics(tokens: 64)).exportText()
        XCTAssertTrue(
            text.contains("n/a (needs >= 512 generated tokens, got 64)"))
    }

    func testSustainedExportListsPerGenerationSequence() throws {
        let generations = [
            syntheticMetrics(tokens: 600),
            syntheticMetrics(tokens: 600),
        ]
        let result = SustainedLoopResult(
            generations: generations, totalElapsedSeconds: 312,
            lastGenerationTruncated: true)
        let text = report(mode: .sustained, sustained: result).exportText()

        XCTAssertTrue(text.contains("sustained loop: 2 generations over 5.2 min"))
        XCTAssertTrue(text.contains("final generation truncated"))
        // Records are 1 s apart → overall (n−1)/span = 1.00 tok/s.
        XCTAssertTrue(text.contains("gen 0: overall 1.00 tok/s, window 1.00 tok/s"))
        XCTAssertTrue(text.contains("gen 1: overall 1.00 tok/s, window 1.00 tok/s"))
        XCTAssertTrue(text.contains("last generation per-token: median GPU"))
    }

    func testOperatorPlaceholdersRenderWhenEmpty() throws {
        var r = report(mode: .burst, burst: nil, batteryNote: "")
        r.coldOrWarmNote = ""
        r.physFootprintBytes = nil
        let text = r.exportText()
        XCTAssertTrue(text.contains("battery health: (record manually)"))
        XCTAssertTrue(text.contains("cold/warm: (record manually)"))
        XCTAssertTrue(text.contains("burst: no generation recorded"))
        XCTAssertFalse(text.contains("phys_footprint"))
    }
}
