import XCTest
import QwenMetalEngine

/// P2-6 engine-side harness tests: DecodeLoop's cooperative stop, the
/// instrumented single-generation runner (stop-reason inference + rate
/// plumbing), the sustained regenerate-loop driver (fake clock — no real
/// time), and the phys_footprint cross-check. All scripted; no checkpoint,
/// no Metal device needed.
final class BenchHarnessTests: XCTestCase {

    private struct ScriptedSource: NextTokenLogitsSource {
        let vocabSize: Int
        let script: ([Int]) -> [Float]
        func lastPositionLogits(ids: [Int]) throws -> [Float] { script(ids) }
    }

    /// Logits whose argmax is `index`.
    private func peaked(_ index: Int, vocab: Int = 4) -> [Float] {
        var values = [Float](repeating: 0, count: vocab)
        values[index] = 10
        return values
    }

    /// A source that always emits token 1 and counts its forwards.
    private final class CountingSource: NextTokenLogitsSource {
        let vocabSize = 4
        private(set) var forwardCount = 0
        func lastPositionLogits(ids: [Int]) throws -> [Float] {
            forwardCount += 1
            var values = [Float](repeating: 0, count: vocabSize)
            values[1] = 10
            return values
        }
    }

    /// Synthetic per-token metrics for sustained-loop tests.
    private func metrics(
        tokens: Int, wallSeconds: Double = 10,
        stop: GenerationStopReason
    ) -> GenerationMetrics {
        GenerationMetrics(
            promptTokenCount: 84, generatedTokenCount: tokens,
            wallSeconds: wallSeconds, stopReason: stop, timing: nil,
            overallTokensPerSecond: nil, canonicalWindowTokensPerSecond: nil)
    }

    // MARK: - DecodeLoop shouldStop (cooperative stop at token boundaries)

    func testShouldStopBeforeFirstStepGeneratesNothingAndRunsNoForward() throws {
        let source = CountingSource()
        let generated = try DecodeLoop(model: source, maxContext: 100).generate(
            promptIds: [0], maxNewTokens: 10, shouldStop: { true })
        XCTAssertEqual(generated, [], "stop before step 0 must yield no tokens")
        XCTAssertEqual(
            source.forwardCount, 0,
            "shouldStop is polled BEFORE the forward — no wasted dispatch")
    }

    func testShouldStopEndsGenerationAtTokenBoundary() throws {
        let source = CountingSource()
        var stepsCompleted = 0
        let generated = try DecodeLoop(model: source, maxContext: 100).generate(
            promptIds: [0], maxNewTokens: 10,
            onStep: { _, _, _ in stepsCompleted += 1 },
            shouldStop: { stepsCompleted >= 2 })
        XCTAssertEqual(generated, [1, 1], "must keep the tokens generated so far")
        XCTAssertEqual(source.forwardCount, 2, "no forward after the stop fired")
    }

    func testNilShouldStopLeavesBehaviorUnchanged() throws {
        let source = CountingSource()
        let generated = try DecodeLoop(model: source, maxContext: 100).generate(
            promptIds: [0], maxNewTokens: 3)
        XCTAssertEqual(generated, [1, 1, 1])
    }

    // MARK: - BenchGenerationRunner: stop-reason inference

    func testRunnerInfersMaxNewTokensAndCollectsRecords() throws {
        let source = ScriptedSource(vocabSize: 4) { _ in self.peaked(1) }
        // Synthetic dual-timing records: token i completes at wall i + 1.5 s.
        var calls = 0
        let runner = BenchGenerationRunner(
            model: source, maxContext: 100, eosTokenIds: [3],
            stepRecord: {
                defer { calls += 1 }
                let base = Double(calls)
                return TokenStepRecord(
                    timing: DispatchTiming(
                        wallStart: base, wallEnd: base + 1.5,
                        gpuStart: base, gpuEnd: base + 1.0),
                    dispatchCount: 24)
            })
        let result = try runner.run(promptIds: [0, 0], maxNewTokens: 3)

        XCTAssertEqual(result.tokenIds, [1, 1, 1])
        XCTAssertEqual(result.metrics.stopReason, .maxNewTokens)
        XCTAssertEqual(result.metrics.promptTokenCount, 2)
        XCTAssertEqual(result.metrics.generatedTokenCount, 3)
        let timing = try XCTUnwrap(result.metrics.timing)
        XCTAssertEqual(timing.tokenCount, 3)
        XCTAssertEqual(timing.minDispatchCount, 24)
        XCTAssertEqual(timing.maxDispatchCount, 24)
        XCTAssertEqual(timing.medianGPUSeconds, 1.0, accuracy: 1e-12)
        XCTAssertEqual(timing.medianWallSeconds, 1.5, accuracy: 1e-12)
        // Completion-to-completion: 2 tokens over wallEnd 3.5 − 1.5 = 2 s.
        XCTAssertEqual(
            try XCTUnwrap(result.metrics.overallTokensPerSecond), 1.0,
            accuracy: 1e-12)
        XCTAssertNil(
            result.metrics.canonicalWindowTokensPerSecond,
            "window rate needs ≥ 512 generated tokens")
    }

    func testRunnerInfersEOSEvenAtMaxTokensBoundary() throws {
        // Third token is EOS, and maxNewTokens is also 3 — EOS wins.
        let source = ScriptedSource(vocabSize: 4) { ids in
            self.peaked(ids.count >= 3 ? 3 : 1)
        }
        let runner = BenchGenerationRunner(
            model: source, maxContext: 100, eosTokenIds: [3])
        let result = try runner.run(promptIds: [0], maxNewTokens: 3)
        XCTAssertEqual(result.tokenIds, [1, 1, 3])
        XCTAssertEqual(result.metrics.stopReason, .eos)
        XCTAssertNil(result.metrics.timing, "no stepRecord → no timing summary")
    }

    func testRunnerInfersContextFull() throws {
        let source = ScriptedSource(vocabSize: 4) { _ in self.peaked(1) }
        let runner = BenchGenerationRunner(
            model: source, maxContext: 5, eosTokenIds: [])
        let result = try runner.run(promptIds: [0, 0, 0], maxNewTokens: 10)
        XCTAssertEqual(result.metrics.generatedTokenCount, 2)
        XCTAssertEqual(result.metrics.stopReason, .contextFull)
    }

    func testRunnerInfersStopRequested() throws {
        let source = ScriptedSource(vocabSize: 4) { _ in self.peaked(1) }
        let runner = BenchGenerationRunner(
            model: source, maxContext: 100, eosTokenIds: [])
        var tokensSeen = 0
        let result = try runner.run(
            promptIds: [0], maxNewTokens: 10,
            shouldStop: { tokensSeen >= 2 },
            onToken: { _, _ in tokensSeen += 1 })
        XCTAssertEqual(result.metrics.generatedTokenCount, 2)
        XCTAssertEqual(result.metrics.stopReason, .stopRequested)
    }

    // MARK: - SustainedLoop (fake clock; the pinned regenerate protocol)

    func testSustainedLoopRegeneratesUntilDurationElapsed() throws {
        var now = 0.0
        var calls = 0
        let result = try SustainedLoop(minDurationSeconds: 25).run(
            clock: { now },
            generate: { shouldStop in
                XCTAssertFalse(
                    shouldStop(),
                    "duration not elapsed at generation start")
                calls += 1
                now += 10  // each full generation takes 10 s
                return self.metrics(tokens: 100, stop: .contextFull)
            })
        XCTAssertEqual(calls, 3, "10 s + 10 s < 25 s → a third generation runs")
        XCTAssertEqual(result.generations.count, 3)
        XCTAssertEqual(result.totalElapsedSeconds, 30, accuracy: 1e-12)
        XCTAssertFalse(result.lastGenerationTruncated)
    }

    func testSustainedLoopMarksDurationTruncatedFinalGeneration() throws {
        var now = 0.0
        let result = try SustainedLoop(minDurationSeconds: 15).run(
            clock: { now },
            generate: { shouldStop in
                now += 10
                // A generation that polls shouldStop at its token boundaries:
                // once the bound has passed it ends as stopRequested.
                return self.metrics(
                    tokens: 100, stop: shouldStop() ? .stopRequested : .contextFull)
            })
        XCTAssertEqual(result.generations.count, 2)
        XCTAssertEqual(result.generations[0].stopReason, .contextFull)
        XCTAssertEqual(result.generations[1].stopReason, .stopRequested)
        XCTAssertTrue(result.lastGenerationTruncated)
    }

    func testSustainedLoopThrowsOnEmptyGenerationBeforeDuration() {
        var now = 0.0
        XCTAssertThrowsError(
            try SustainedLoop(minDurationSeconds: 100).run(
                clock: { now },
                generate: { _ in
                    now += 1
                    return self.metrics(tokens: 0, stop: .stopRequested)
                })
        ) { error in
            XCTAssertEqual(
                error as? SustainedLoopError, .emptyGeneration(index: 0))
        }
    }

    func testSustainedLoopKeepsZeroTokenFinalGenerationAfterDuration() throws {
        // A stop that fires before the final generation's first token is
        // legitimate once the duration HAS elapsed — kept, not an error.
        var now = 0.0
        var calls = 0
        let result = try SustainedLoop(minDurationSeconds: 5).run(
            clock: { now },
            generate: { shouldStop in
                calls += 1
                if calls == 1 {
                    now = 4  // ends just before the bound → loop regenerates
                    return self.metrics(tokens: 100, stop: .contextFull)
                }
                now = 6  // bound passes before this generation's first token
                XCTAssertTrue(shouldStop())
                return self.metrics(tokens: 0, stop: .stopRequested)
            })
        XCTAssertEqual(result.generations.count, 2)
        XCTAssertEqual(result.generations[1].generatedTokenCount, 0)
        XCTAssertTrue(result.lastGenerationTruncated)
    }

    // MARK: - Protocol defaults (precedent pins)

    func testBenchDefaultsMatchThePinnedProtocolNumbers() {
        XCTAssertEqual(
            BenchDefaults.burstMaxNewTokens, 640,
            "P2-5 Mac row burst cap — covers the 128–512 window with margin")
        XCTAssertEqual(
            BenchDefaults.sustainedMinDurationSeconds, 300,
            "PLAN.md sustained minimum: ≥ 5 min")
    }

    func testRunnerRecordsPrefillSeconds() throws {
        let source = ScriptedSource(vocabSize: 4) { _ in self.peaked(1) }
        let runner = BenchGenerationRunner(
            model: source, maxContext: 100, eosTokenIds: [])
        let metrics = try runner.run(promptIds: [0, 0], maxNewTokens: 2).metrics
        let prefill = try XCTUnwrap(metrics.prefillSeconds)
        XCTAssertGreaterThanOrEqual(prefill, 0)
        XCTAssertLessThanOrEqual(prefill, metrics.wallSeconds)
    }

    // MARK: - MemoryFootprint (cross-check reader)

    func testPhysFootprintReadsANonTrivialValue() throws {
        let bytes = try XCTUnwrap(MemoryFootprint.currentPhysFootprintBytes())
        XCTAssertGreaterThan(
            bytes, 1 << 20,
            "a running test process has a phys_footprint well over 1 MiB")
    }
}
