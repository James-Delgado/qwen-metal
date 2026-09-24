import XCTest
import QwenMetalEngine

/// P6-1 (phase-6.md edge test 1): the energy-mode regenerate loop. The
/// energy cycle is OPERATOR-bounded (stop at exactly 70% SoC), so unlike
/// the P2-6 sustained mode — where Stop aborts without a report — the
/// operator stop ends the loop at the next token boundary AND the loop
/// still returns a result. Fake clock, scripted generations; no device.
final class EnergyLoopTests: XCTestCase {

    private func metrics(
        tokens: Int, wallSeconds: Double = 10,
        stop: GenerationStopReason
    ) -> GenerationMetrics {
        GenerationMetrics(
            promptTokenCount: 84, generatedTokenCount: tokens,
            wallSeconds: wallSeconds, stopReason: stop, timing: nil,
            overallTokensPerSecond: nil, canonicalWindowTokensPerSecond: nil)
    }

    // MARK: - operator stop ends at the token boundary AND reports

    func testOperatorStopEndsLoopAtTokenBoundaryAndStillReports() throws {
        var now = 0.0
        var operatorStopped = false
        var calls = 0
        let result = try EnergyLoop().run(
            clock: { now },
            operatorStop: { operatorStopped },
            generate: { shouldStop in
                calls += 1
                XCTAssertFalse(shouldStop(), "no stop pending at generation start")
                if calls == 3 {
                    // The operator presses Stop mid-generation: the generation
                    // polls shouldStop at its next token boundary and ends
                    // as stopRequested with the tokens generated so far.
                    operatorStopped = true
                    now += 4
                    XCTAssertTrue(shouldStop())
                    return self.metrics(tokens: 37, wallSeconds: 4, stop: .stopRequested)
                }
                now += 10
                return self.metrics(tokens: 200, stop: .contextFull)
            })
        XCTAssertEqual(calls, 3, "the loop must not start a 4th generation")
        XCTAssertEqual(result.generations.count, 3)
        XCTAssertEqual(result.endedBy, .operatorStop)
        XCTAssertTrue(
            result.lastGenerationTruncated,
            "the operator-cut final generation is flagged, never silently mixed")
        XCTAssertEqual(result.generations.last?.generatedTokenCount, 37)
        XCTAssertEqual(result.cycleWallSeconds, 24, accuracy: 1e-12)
    }

    func testFinalGenerationEndingOnItsOwnIsNotFlaggedTruncated() throws {
        var now = 0.0
        var operatorStopped = false
        let result = try EnergyLoop().run(
            clock: { now },
            operatorStop: { operatorStopped },
            generate: { _ in
                now += 10
                // Stop pressed exactly as the generation ends on its own
                // (context fill) — nothing was cut.
                operatorStopped = true
                return self.metrics(tokens: 200, stop: .contextFull)
            })
        XCTAssertEqual(result.generations.count, 1)
        XCTAssertEqual(result.endedBy, .operatorStop)
        XCTAssertFalse(result.lastGenerationTruncated)
    }

    // MARK: - cumulative totals == sum of the per-generation records

    func testCumulativeTokensAndWallEqualTheSumOfRecords() throws {
        var now = 0.0
        var operatorStopped = false
        let script: [(Int, Double)] = [(200, 10), (198, 9.5), (61, 3.25)]
        var index = 0
        let result = try EnergyLoop().run(
            clock: { now },
            operatorStop: { operatorStopped },
            generate: { _ in
                let (tokens, wall) = script[index]
                index += 1
                now += wall
                if index == script.count { operatorStopped = true }
                return self.metrics(
                    tokens: tokens, wallSeconds: wall,
                    stop: index == script.count ? .stopRequested : .contextFull)
            })
        XCTAssertEqual(result.totalGeneratedTokens, 200 + 198 + 61)
        XCTAssertEqual(
            result.totalGeneratedTokens,
            result.generations.reduce(0) { $0 + $1.generatedTokenCount })
        XCTAssertEqual(
            result.totalGenerationWallSeconds, 10 + 9.5 + 3.25, accuracy: 1e-12)
        XCTAssertEqual(
            result.totalGenerationWallSeconds,
            result.generations.reduce(0) { $0 + $1.wallSeconds }, accuracy: 1e-12)
        // Real clock offsets per generation (the thermal chart's x-axis).
        XCTAssertEqual(result.generationEndOffsetsSeconds, [10, 19.5, 22.75])
        XCTAssertEqual(result.cycleWallSeconds, 22.75, accuracy: 1e-12)
    }

    // MARK: - empty-generation refusal (the SustainedLoop rule carried over)

    func testRefusesToSpinOnEmptyGenerationBeforeStop() {
        var now = 0.0
        XCTAssertThrowsError(
            try EnergyLoop().run(
                clock: { now },
                operatorStop: { false },
                generate: { _ in
                    now += 1
                    return self.metrics(tokens: 0, stop: .stopRequested)
                })
        ) { error in
            XCTAssertEqual(error as? EnergyLoopError, .emptyGeneration(index: 0))
        }
    }

    func testKeepsZeroTokenFinalGenerationWhenStopWasAlreadyPending() throws {
        // Stop pressed before the final generation's first token: legitimate
        // (the stop is honored at the first boundary) — kept, flagged, no error.
        var now = 0.0
        var operatorStopped = false
        var calls = 0
        let result = try EnergyLoop().run(
            clock: { now },
            operatorStop: { operatorStopped },
            generate: { shouldStop in
                calls += 1
                if calls == 1 {
                    now += 10
                    operatorStopped = false
                    return self.metrics(tokens: 200, stop: .contextFull)
                }
                operatorStopped = true
                XCTAssertTrue(shouldStop())
                return self.metrics(tokens: 0, wallSeconds: 0.01, stop: .stopRequested)
            })
        XCTAssertEqual(result.generations.count, 2)
        XCTAssertEqual(result.generations[1].generatedTokenCount, 0)
        XCTAssertTrue(result.lastGenerationTruncated)
        XCTAssertEqual(result.totalGeneratedTokens, 200)
    }

    // MARK: - optional safety bound (Mac sanity runs; never the protocol)

    func testDurationBoundEndsTheLoopWhenNoOperatorStopFires() throws {
        var now = 0.0
        var calls = 0
        let result = try EnergyLoop(maxDurationSeconds: 25).run(
            clock: { now },
            operatorStop: { false },
            generate: { shouldStop in
                calls += 1
                now += 10
                return self.metrics(
                    tokens: 200, stop: shouldStop() ? .stopRequested : .contextFull)
            })
        XCTAssertEqual(calls, 3, "10 + 10 < 25 → a third generation runs, then the bound ends it")
        XCTAssertEqual(result.endedBy, .durationBound)
        XCTAssertTrue(result.lastGenerationTruncated)
    }

    func testOperatorStopWinsTheEndReasonOverACoincidentBound() throws {
        var now = 0.0
        let result = try EnergyLoop(maxDurationSeconds: 5).run(
            clock: { now },
            operatorStop: { now >= 5 },
            generate: { _ in
                now += 10
                return self.metrics(tokens: 200, stop: .stopRequested)
            })
        XCTAssertEqual(result.endedBy, .operatorStop)
    }
}
