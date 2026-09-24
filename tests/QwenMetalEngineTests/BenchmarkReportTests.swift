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
        tokens: Int, dispatches: Int = 591,
        prefillSpan: PrefillSpan? = nil
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
            prefillSeconds: 21.0,
            prefillSpan: prefillSpan, stopReason: .maxNewTokens,
            timing: collector.summary(),
            overallTokensPerSecond: collector.overallTokensPerSecond(),
            canonicalWindowTokensPerSecond:
                collector.canonicalWindowTokensPerSecond(),
            latencyVariance: collector.canonicalWindowLatencyVariance()
                ?? collector.allTokensLatencyVariance())
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
            kernelPath: .naive, prefillPath: .sequential,
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
        // P5-1 (phase-5.md D1): the P2-6 TTFT-style field keeps exporting,
        // honestly labeled as runner-clocked — rows cite the prefill span.
        XCTAssertTrue(text.contains(
            "ttft-style (legacy P2-6 field, runner-clocked): "
            + "84 prompt tokens, 21.00 s to first token available"))
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

    // MARK: - (P5-1) prefill span line (phase-5.md D1 — edge test 12)

    /// The span line is the prefill metric of record: prompt tokens ÷ span
    /// WALL time, with span GPU time and dispatch count reported alongside
    /// (dual timing, hard rule 7) — and it coexists with the legacy field.
    func testBurstExportCarriesPrefillSpanLine() throws {
        let span = PrefillSpan(
            promptTokenCount: 84,
            span: ForwardCallSpan(
                stepCount: 84, wallSeconds: 20.0, gpuSeconds: 19.25,
                dispatchCount: 49_644))
        let text = report(
            mode: .burst,
            burst: syntheticMetrics(tokens: 64, prefillSpan: span)).exportText()
        XCTAssertTrue(text.contains(
            "prefill span: 84 tokens in 20.000 s = 4.20 tok/s (of record) — "
            + "GPU 19.250 s, 49644 dispatches (prompt processing only, "
            + "excl. first decode forward)"), text)
        XCTAssertTrue(
            text.contains("ttft-style (legacy P2-6 field"),
            "legacy TTFT-style field still exports next to the span (D1)")
        XCTAssertFalse(text.contains("WARM PREFIX"))
    }

    /// No span (CPU backend / scripted source) → the line is absent, the
    /// legacy field still renders.
    func testExportWithoutPrefillSpanOmitsTheLine() throws {
        let text = report(
            mode: .burst, burst: syntheticMetrics(tokens: 64)).exportText()
        XCTAssertFalse(text.contains("prefill span:"))
        XCTAssertTrue(text.contains("ttft-style (legacy P2-6 field"))
    }

    // MARK: - (P3-5) rows record the weights format

    func testDefaultExportIsBF16Phase2() throws {
        let text = report(
            mode: .burst, burst: syntheticMetrics(tokens: 64)).exportText()
        XCTAssertTrue(text.contains("Phase 2 row export"))
        XCTAssertTrue(text.contains("weights bf16"))
        XCTAssertTrue(text.contains("naive fp16 GPU"))
        XCTAssertTrue(text.contains("kernels naive"))
        XCTAssertTrue(text.contains("prefill sequential"),
                      "the bf16 backend is permanently sequential — labeled")
    }

    /// P4-4 / P5-4: q4g64 rows record the kernel path AND the prefill path
    /// (+ chunk size C when tiled — spec D2: C is recorded on every row);
    /// the P5-5 before/after rows differ ONLY in the prefill field, so it
    /// exists on every export surface. q4g64 rows are Phase 5 rows now.
    func testQ4G64ExportRecordsFormatPhaseKernelAndPrefillPath() throws {
        func q4Text(
            _ kernelPath: GPUModel.KernelPath,
            prefill: GPUModel.PrefillPath, chunk: Int? = nil,
            attention: GPUModel.PrefillAttention? = nil
        ) -> String {
            BenchmarkReport(
                dateStamp: "2026-09-02", deviceLabel: "iPhone 15 Pro",
                osVersion: "19.0", batteryHealthNote: "88%",
                coldOrWarmNote: "warm", residency: .mmap, weightsFormat: .q4g64,
                kernelPath: kernelPath, prefillPath: prefill,
                prefillChunkSize: chunk, prefillAttention: attention,
                promptName: "decode-essay", promptTokenCount: 84, mode: .burst,
                burst: syntheticMetrics(tokens: 64)).exportText()
        }
        let tiled = q4Text(.fused, prefill: .tiled, chunk: 512)
        XCTAssertTrue(tiled.contains("Phase 6 row export"))
        XCTAssertTrue(tiled.contains("weights q4g64"))
        XCTAssertTrue(tiled.contains("q4g64 fused-dequant GPU"))
        XCTAssertTrue(tiled.contains("residency mmap"))
        XCTAssertTrue(tiled.contains("kernels fused"))
        XCTAssertTrue(tiled.contains("prefill tiled (C=512)"), tiled)

        // PF-2: tiled rows record the attention kernel (the on-device A/B
        // rows differ only in this field); either variant is labeled.
        let queryTiled = q4Text(.fused, prefill: .tiled, chunk: 512, attention: .queryTiled)
        XCTAssertTrue(queryTiled.contains("prefill tiled (C=512), attention query-tiled"),
                      queryTiled)
        let perPosition = q4Text(.fused, prefill: .tiled, chunk: 512, attention: .perPosition)
        XCTAssertTrue(perPosition.contains("prefill tiled (C=512), attention per-position"),
                      perPosition)

        // The sequential A/B arm is still a Phase 5 row, labeled by prefill.
        let sequential = q4Text(.fused, prefill: .sequential)
        XCTAssertTrue(sequential.contains("Phase 6 row export"))
        XCTAssertTrue(sequential.contains("kernels fused"))
        XCTAssertTrue(sequential.contains("prefill sequential"), sequential)
        XCTAssertFalse(sequential.contains("C="),
                       "no chunk size on a sequential row")
        XCTAssertFalse(q4Text(.fused, prefill: .sequential, attention: .queryTiled)
                        .contains("attention"),
                       "no attention kernel label on a sequential row")

        // The naive kernel arm (sequential-only) stays labeled by kernels.
        let naive = q4Text(.naive, prefill: .sequential)
        XCTAssertTrue(naive.contains("Phase 6 row export"))
        XCTAssertTrue(naive.contains("kernels naive"))
        XCTAssertTrue(naive.contains("prefill sequential"))
    }

    // MARK: - (P4-1) latency-variance line (spec D7 — every Phase 4 row)

    func testBurstExportReportsWindowLatencyVariance() throws {
        // Completion-to-completion spans are 1.0 s throughout (records at
        // i + 0.25), so every percentile is 1000 ms with zero stalls.
        let text = report(
            mode: .burst, burst: syntheticMetrics(tokens: 640)).exportText()
        XCTAssertTrue(text.contains(
            "latency (window tokens 128-512): p50 1000.00 ms, "
            + "p95 1000.00 ms, p99 1000.00 ms, max 1000.00 ms, stalls 0"), text)
    }

    func testBurstExportLabelsAllTokensVarianceBelow512() throws {
        let text = report(
            mode: .burst, burst: syntheticMetrics(tokens: 64)).exportText()
        XCTAssertTrue(text.contains("latency (all tokens): p50 1000.00 ms"), text)
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
