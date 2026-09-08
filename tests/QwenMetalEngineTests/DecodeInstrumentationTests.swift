import XCTest
@testable import QwenMetalEngine

/// P2-5 unit tests for the engine-side per-token instrumentation aggregates:
/// median arithmetic, the wall−GPU overhead semantics (per-token, then
/// median), dispatch-count stability reporting, and the canonical-window
/// (tokens 128–512) accounting. Pure arithmetic on synthetic records — no
/// Metal, no checkpoint.
final class DecodeInstrumentationTests: XCTestCase {

    /// A record with explicit dual-timing fields (seconds).
    private func record(
        wallStart: Double = 0, wallEnd: Double,
        gpuStart: Double = 0, gpuEnd: Double,
        dispatches: Int = 591
    ) -> TokenStepRecord {
        TokenStepRecord(
            timing: DispatchTiming(
                wallStart: wallStart, wallEnd: wallEnd,
                gpuStart: gpuStart, gpuEnd: gpuEnd),
            dispatchCount: dispatches)
    }

    /// A record whose only meaningful field is `wallEnd` (window accounting).
    private func completionAt(_ wallEnd: Double) -> TokenStepRecord {
        record(wallStart: wallEnd - 0.01, wallEnd: wallEnd,
               gpuStart: wallEnd - 0.009, gpuEnd: wallEnd - 0.001)
    }

    private func collector(_ records: [TokenStepRecord]) -> DecodeTimingCollector {
        var c = DecodeTimingCollector()
        for r in records { c.append(r) }
        return c
    }

    // MARK: - Summary medians

    func testEmptyCollectorHasNoSummaryAndNoRates() {
        let c = DecodeTimingCollector()
        XCTAssertNil(c.summary())
        XCTAssertNil(c.canonicalWindowTokensPerSecond())
        XCTAssertNil(c.overallTokensPerSecond())
    }

    func testSummaryMediansOddCount() throws {
        // wall durations 10, 20, 30 ms; gpu durations 9, 18, 28 ms.
        let c = collector([
            record(wallEnd: 0.010, gpuEnd: 0.009),
            record(wallEnd: 0.030, gpuEnd: 0.028),
            record(wallEnd: 0.020, gpuEnd: 0.018),
        ])
        let s = try XCTUnwrap(c.summary())
        XCTAssertEqual(s.tokenCount, 3)
        XCTAssertEqual(s.medianWallSeconds, 0.020, accuracy: 1e-12)
        XCTAssertEqual(s.medianGPUSeconds, 0.018, accuracy: 1e-12)
    }

    func testSummaryMediansEvenCountAverageMiddleTwo() throws {
        // wall durations 10, 20, 30, 40 ms → median 25 ms.
        let c = collector([0.010, 0.020, 0.030, 0.040].map {
            record(wallEnd: $0, gpuEnd: $0 - 0.001)
        })
        let s = try XCTUnwrap(c.summary())
        XCTAssertEqual(s.medianWallSeconds, 0.025, accuracy: 1e-12)
    }

    /// The overhead metric is median(per-token wall − GPU), NOT
    /// medianWall − medianGPU — the medians may come from different tokens.
    func testOverheadIsMedianOfPerTokenDeltas() throws {
        // (wall, gpu) ms: (10, 9) → 1; (20, 12) → 8; (30, 28) → 2.
        // median wall = 20, median gpu = 12 → naive subtraction would say 8;
        // the per-token median is 2.
        let c = collector([
            record(wallEnd: 0.010, gpuEnd: 0.009),
            record(wallEnd: 0.020, gpuEnd: 0.012),
            record(wallEnd: 0.030, gpuEnd: 0.028),
        ])
        let s = try XCTUnwrap(c.summary())
        XCTAssertEqual(s.medianOverheadSeconds, 0.002, accuracy: 1e-12)
        XCTAssertNotEqual(
            s.medianOverheadSeconds,
            s.medianWallSeconds - s.medianGPUSeconds,
            "this construction must distinguish the two definitions")
    }

    func testDispatchCountMinMaxAcrossRecords() throws {
        let c = collector([
            record(wallEnd: 0.01, gpuEnd: 0.009, dispatches: 591),
            record(wallEnd: 0.01, gpuEnd: 0.009, dispatches: 589),
            record(wallEnd: 0.01, gpuEnd: 0.009, dispatches: 591),
        ])
        let s = try XCTUnwrap(c.summary())
        XCTAssertEqual(s.minDispatchCount, 589)
        XCTAssertEqual(s.maxDispatchCount, 591)
    }

    // MARK: - Canonical window (tokens 128–512, protocol pin)

    func testWindowPinsMatchPlanProtocol() {
        XCTAssertEqual(CanonicalDecodeWindow.firstToken, 128)
        XCTAssertEqual(CanonicalDecodeWindow.lastToken, 512)
        XCTAssertEqual(CanonicalDecodeWindow.tokenSpan, 384)
    }

    func testWindowUnavailableBelow512Records() {
        let c = collector((1...511).map { completionAt(Double($0)) })
        XCTAssertNil(c.canonicalWindowTokensPerSecond())
    }

    func testWindowAvailableAtExactly512Records() throws {
        // completion(i) = i seconds → token 512 at 512 s, token 128 at 128 s:
        // 384 tokens / 384 s = exactly 1 tok/s.
        let c = collector((1...512).map { completionAt(Double($0)) })
        let rate = try XCTUnwrap(c.canonicalWindowTokensPerSecond())
        XCTAssertEqual(rate, 1.0, accuracy: 1e-12)
    }

    /// Only the completion times of tokens 128 and 512 define the window —
    /// generation may continue past 512 without changing the windowed rate.
    func testWindowUsesTokens128And512CompletionTimesOnly() throws {
        var completions = (1...600).map { Double($0) * 7.0 }
        completions[127] = 100    // token 128 completes at t = 100 s
        completions[511] = 292    // token 512 completes at t = 292 s
        let c = collector(completions.map { completionAt($0) })
        // 384 tokens / 192 s = 2 tok/s regardless of every other record.
        let rate = try XCTUnwrap(c.canonicalWindowTokensPerSecond())
        XCTAssertEqual(rate, 2.0, accuracy: 1e-12)
    }

    func testWindowNilOnNonPositiveSpan() {
        // Token 512 "completes" before token 128 — corrupt sequence, no rate.
        var completions = (1...512).map { Double($0) }
        completions[511] = 0.5
        let c = collector(completions.map { completionAt($0) })
        XCTAssertNil(c.canonicalWindowTokensPerSecond())
    }

    // MARK: - Overall rate

    func testOverallRateSpansFirstToLastCompletion() throws {
        // 6 completions at t = 5, 6, 7, 8, 9, 10 s → 5 tokens / 5 s = 1.
        let c = collector((5...10).map { completionAt(Double($0)) })
        let rate = try XCTUnwrap(c.overallTokensPerSecond())
        XCTAssertEqual(rate, 1.0, accuracy: 1e-12)
    }

    func testOverallRateUnavailableBelow2Records() {
        XCTAssertNil(collector([completionAt(1)]).overallTokensPerSecond())
    }

    func testOverallRateNilOnNonPositiveSpan() {
        let c = collector([completionAt(5), completionAt(5)])
        XCTAssertNil(c.overallTokensPerSecond())
    }

    // MARK: - P4-1 latency-variance stats (spec D7 / edge test 12)
    // Conventions pinned in DECISIONS.md 2026-09-08 (P4-1 sanity bounds):
    // spans = consecutive completion-to-completion wallEnd deltas,
    // nearest-rank percentiles, stall = span strictly > 2 × p50.

    func testVarianceNearestRankPercentilesOn100DistinctSpans() throws {
        // Spans 1..99 ms plus one 250 ms outlier, shuffled: nearest-rank
        // p50 = 50 ms, p95 = 95 ms, p99 = 99 ms, max = 250 ms; the stall
        // threshold is 2 × 50 = 100 ms, so exactly the outlier stalls.
        var spans = (1...99).map { Double($0) / 1000 } + [0.250]
        spans.shuffle()
        let s = try XCTUnwrap(
            LatencyVarianceStats.compute(interTokenSeconds: spans, scope: .allTokens))
        XCTAssertEqual(s.spanCount, 100)
        XCTAssertEqual(s.p50Seconds, 0.050, accuracy: 1e-12)
        XCTAssertEqual(s.p95Seconds, 0.095, accuracy: 1e-12)
        XCTAssertEqual(s.p99Seconds, 0.099, accuracy: 1e-12)
        XCTAssertEqual(s.maxSeconds, 0.250, accuracy: 1e-12)
        XCTAssertEqual(s.stallCount, 1)
    }

    func testVarianceAllEqualDegenerateCase() throws {
        // All-equal spans: every percentile equals the value, zero stalls
        // (2× median is never strictly exceeded).
        let s = try XCTUnwrap(LatencyVarianceStats.compute(
            interTokenSeconds: Array(repeating: 0.048, count: 384),
            scope: .canonicalWindow))
        XCTAssertEqual(s.p50Seconds, 0.048, accuracy: 1e-12)
        XCTAssertEqual(s.p95Seconds, 0.048, accuracy: 1e-12)
        XCTAssertEqual(s.p99Seconds, 0.048, accuracy: 1e-12)
        XCTAssertEqual(s.maxSeconds, 0.048, accuracy: 1e-12)
        XCTAssertEqual(s.stallCount, 0)
    }

    func testVarianceStallThresholdIsStrictlyGreater() throws {
        // Spans (ms): [1, 1, 1, 2] → nearest-rank p50 = sorted[ceil(.5·4)−1]
        // = 1 ms; threshold 2 ms. A span exactly AT 2× the median is not a
        // stall; one epsilon above is.
        let exact = try XCTUnwrap(LatencyVarianceStats.compute(
            interTokenSeconds: [0.001, 0.001, 0.001, 0.002], scope: .allTokens))
        XCTAssertEqual(exact.p50Seconds, 0.001, accuracy: 1e-15)
        XCTAssertEqual(exact.stallCount, 0)
        let above = try XCTUnwrap(LatencyVarianceStats.compute(
            interTokenSeconds: [0.001, 0.001, 0.001, 0.0021], scope: .allTokens))
        XCTAssertEqual(above.stallCount, 1)
    }

    func testVarianceSingleSpanCollapsesAllPercentiles() throws {
        let s = try XCTUnwrap(LatencyVarianceStats.compute(
            interTokenSeconds: [0.033], scope: .allTokens))
        XCTAssertEqual(s.spanCount, 1)
        XCTAssertEqual(s.p50Seconds, 0.033, accuracy: 1e-12)
        XCTAssertEqual(s.p95Seconds, 0.033, accuracy: 1e-12)
        XCTAssertEqual(s.p99Seconds, 0.033, accuracy: 1e-12)
        XCTAssertEqual(s.maxSeconds, 0.033, accuracy: 1e-12)
        XCTAssertEqual(s.stallCount, 0)
    }

    func testVarianceEmptyAndNonPositiveSpansReturnNil() {
        XCTAssertNil(LatencyVarianceStats.compute(
            interTokenSeconds: [], scope: .allTokens))
        // A non-positive span means the records are not one monotonic
        // generation — fail loudly with nil, never report garbage stats.
        XCTAssertNil(LatencyVarianceStats.compute(
            interTokenSeconds: [0.01, 0.0, 0.01], scope: .allTokens))
        XCTAssertNil(LatencyVarianceStats.compute(
            interTokenSeconds: [0.01, -0.01], scope: .allTokens))
    }

    func testCollectorWindowVarianceUsesExactlyTheWindowSpans() throws {
        // Tokens 1...128 complete 1 s apart; window spans (128→512) are
        // 2 s each except one 10 s stall; post-window spans (3 s) must not
        // leak in. 384 window spans → p50 = 2 s, max = 10 s, 1 stall.
        var completions: [Double] = []
        var t = 0.0
        for i in 1...600 {
            if i <= 128 { t += 1 } else if i <= 512 { t += (i == 300 ? 10 : 2) } else { t += 3 }
            completions.append(t)
        }
        let c = collector(completions.map { completionAt($0) })
        let s = try XCTUnwrap(c.canonicalWindowLatencyVariance())
        XCTAssertEqual(s.scope, .canonicalWindow)
        XCTAssertEqual(s.spanCount, CanonicalDecodeWindow.tokenSpan)
        XCTAssertEqual(s.p50Seconds, 2.0, accuracy: 1e-9)
        XCTAssertEqual(s.maxSeconds, 10.0, accuracy: 1e-9)
        XCTAssertEqual(s.stallCount, 1)
    }

    func testCollectorWindowVarianceNilBelow512Records() {
        let c = collector((1...511).map { completionAt(Double($0)) })
        XCTAssertNil(c.canonicalWindowLatencyVariance())
    }

    func testCollectorAllTokensVarianceFromCompletionDeltas() throws {
        // Completions at 1, 2, 4 s → spans [1, 2]: nearest-rank p50 =
        // sorted[ceil(.5·2)−1] = 1 s, max = 2 s, no stall (2 is not > 2).
        let c = collector([1.0, 2.0, 4.0].map { completionAt($0) })
        let s = try XCTUnwrap(c.allTokensLatencyVariance())
        XCTAssertEqual(s.scope, .allTokens)
        XCTAssertEqual(s.spanCount, 2)
        XCTAssertEqual(s.p50Seconds, 1.0, accuracy: 1e-12)
        XCTAssertEqual(s.maxSeconds, 2.0, accuracy: 1e-12)
        XCTAssertEqual(s.stallCount, 0)
    }

    func testCollectorAllTokensVarianceNilBelow2Records() {
        XCTAssertNil(collector([completionAt(1)]).allTokensLatencyVariance())
    }
}
