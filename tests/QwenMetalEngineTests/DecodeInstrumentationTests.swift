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
}
