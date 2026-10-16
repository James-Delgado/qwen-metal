import XCTest
import Foundation
import Metal
@testable import QwenMetalEngine

// MARK: - Helper unit tests (artifact-free; hard rule 3 — tests land with the harness)

final class LongDepthReportHelperTests: XCTestCase {

    func testCheckpointScheduleIsEveryNPlusFinalStep() throws {
        XCTAssertEqual(try LongDepthReport.checkpointSchedule(steps: 600, every: 256),
                       [256, 512, 600])
        // The final step is never duplicated when it lands on the grid.
        XCTAssertEqual(try LongDepthReport.checkpointSchedule(steps: 512, every: 256),
                       [256, 512])
        // A run shorter than one interval still checkpoints its final step.
        XCTAssertEqual(try LongDepthReport.checkpointSchedule(steps: 100, every: 256),
                       [100])
        XCTAssertEqual(try LongDepthReport.checkpointSchedule(steps: 3, every: 1),
                       [1, 2, 3])
        XCTAssertThrowsError(try LongDepthReport.checkpointSchedule(steps: 0, every: 256))
        XCTAssertThrowsError(try LongDepthReport.checkpointSchedule(steps: 10, every: 0))
    }

    func testPositiveIntKnobParsing() throws {
        XCTAssertEqual(try LongDepthReport.positiveInt(nil, name: "X", default: 7), 7)
        XCTAssertEqual(try LongDepthReport.positiveInt("42", name: "X", default: 7), 42)
        XCTAssertEqual(try LongDepthReport.positiveInt(" 42 ", name: "X", default: 7), 42)
        for bad in ["0", "-1", "abc", "", "1.5"] {
            XCTAssertThrowsError(
                try LongDepthReport.positiveInt(bad, name: "QWEN_X", default: 7),
                "'\(bad)' must be rejected"
            ) { error in
                XCTAssertTrue("\(error)".contains("QWEN_X"),
                              "error names the knob: \(error)")
            }
        }
    }

    func testPromptListParsing() throws {
        XCTAssertEqual(try LongDepthReport.promptList(nil), LongDepthReport.allPrompts)
        XCTAssertEqual(try LongDepthReport.promptList("short_english, code_snippet"),
                       ["short_english", "code_snippet"])
        XCTAssertThrowsError(try LongDepthReport.promptList("short_english,bogus"))
        XCTAssertThrowsError(try LongDepthReport.promptList(""))
    }

    func testTopIndicesMatchesSortReferenceOverFiniteValues() {
        var generator = SystemRandomNumberGenerator()
        var values = (0..<10_000).map { _ in Float.random(in: -8...8, using: &generator) }
        // Duplicates (tie-break by lower index) and non-finite entries (skipped).
        for i in stride(from: 0, to: values.count, by: 97) { values[i] = 5.0 }
        values[10] = .nan
        values[20] = .infinity
        values[30] = -.infinity
        let reference = values.indices
            .filter { values[$0].isFinite }
            .sorted { values[$0] != values[$1] ? values[$0] > values[$1] : $0 < $1 }
        XCTAssertEqual(LongDepthReport.topIndices(values, count: 64),
                       Array(reference.prefix(64)))
        XCTAssertEqual(LongDepthReport.topIndices(values, count: 1), [reference[0]])
        // Fewer finite values than requested: all of them, in order.
        XCTAssertEqual(LongDepthReport.topIndices([2, .nan, 1, 3], count: 64), [3, 0, 2])
        XCTAssertEqual(LongDepthReport.topIndices([], count: 64), [])
    }

    func testStepStatsCountsNonFiniteAndMargin() {
        let stats = LongDepthReport.StepStats([1, 3, 2, .nan, -.infinity])
        XCTAssertEqual(stats.nonFiniteCount, 2)
        XCTAssertEqual(stats.top1, 1)
        XCTAssertEqual(stats.top2, 2)
        XCTAssertEqual(stats.margin, 1.0)
        XCTAssertEqual(stats.m64, 3)
        // Tie-band eligibility uses the suite's ε_tie = 2⁻⁴·M64 on this vector.
        XCTAssertFalse(stats.isForkEligible)
        // A pathological reference (one finite logit) is reported, not a crash.
        let degenerate = LongDepthReport.CheckpointComparison(
            gpu: [1, .nan], cpu: [1, .nan])
        XCTAssertEqual(degenerate.cpuTop2, -1)
        XCTAssertTrue(degenerate.cpuMargin.isNaN)
        XCTAssertTrue(degenerate.tieAwareTop1Holds)
        let tie = LongDepthReport.StepStats([4.0, 4.0 + 0.1, 0])
        XCTAssertEqual(tie.top1, 1)
        XCTAssertTrue(tie.isForkEligible, "margin 0.1 < 2⁻⁴·4.1")
        let single = LongDepthReport.StepStats([.nan])
        XCTAssertEqual(single.nonFiniteCount, 1)
        XCTAssertEqual(single.top1, -1)
        XCTAssertEqual(single.top2, -1)
    }

    func testCheckpointComparisonClassifiesAgainstTheTierESpecies() {
        let cpu: [Float] = (0..<200).map { Float($0 % 17) - 8 } // M_step = 8
        let identical = LongDepthReport.CheckpointComparison(gpu: cpu, cpu: cpu)
        XCTAssertTrue(identical.fullWithinSpecies)
        XCTAssertTrue(identical.top64WithinSpecies)
        XCTAssertTrue(identical.fingerprintsWithinSpecies)
        XCTAssertTrue(identical.argmaxAgrees)
        XCTAssertTrue(identical.tieAwareTop1Holds)
        XCTAssertEqual(identical.worstFull, 0)
        XCTAssertEqual(identical.mStep, 8)

        // One logit nudged past 2⁻⁵·M_step (= 0.25) but nowhere near the top:
        // full-vocab species broken, top-64 untouched, argmax unchanged.
        var nudged = cpu
        let lowIndex = cpu.firstIndex(of: -8)!
        nudged[lowIndex] += 0.3
        let perturbed = LongDepthReport.CheckpointComparison(gpu: nudged, cpu: cpu)
        XCTAssertFalse(perturbed.fullWithinSpecies)
        XCTAssertEqual(perturbed.worstFullIndex, lowIndex)
        XCTAssertTrue(perturbed.top64WithinSpecies)
        XCTAssertTrue(perturbed.argmaxAgrees)
        XCTAssertTrue(perturbed.tieAwareTop1Holds)

        // A flip at a CPU margin inside the tie band holds tie-aware top-1;
        // the same flip beyond the band does not.
        let nearTie: [Float] = [10.0, 10.0 + 0.05, 0, -3]      // ε_tie = 2⁻⁴·10.05 ≈ 0.63
        let flipInBand = LongDepthReport.CheckpointComparison(
            gpu: [10.0 + 0.05, 10.0, 0, -3], cpu: nearTie)
        XCTAssertFalse(flipInBand.argmaxAgrees)
        XCTAssertTrue(flipInBand.tieAwareTop1Holds)
        let wideMargin: [Float] = [10.0, 5.0, 0, -3]
        let flipBeyond = LongDepthReport.CheckpointComparison(
            gpu: [5.0, 10.0, 0, -3], cpu: wideMargin)
        XCTAssertFalse(flipBeyond.argmaxAgrees)
        XCTAssertFalse(flipBeyond.tieAwareTop1Holds)
    }

    func testKVCacheScanCountsOnlyScannedPositions() throws {
        let context: MetalContext
        do {
            context = try SharedGPUModel.metalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: 1, maxContext: 4, headDim: 2)
        let halves = cache.buffer.contents().bindMemory(
            to: Float16.self, capacity: cache.byteCount / 2)
        for i in 0..<(cache.byteCount / 2) { halves[i] = Float16(Float(i) * 0.25) }
        halves[try cache.elementOffset(layer: 0, component: .key, head: 0, position: 1)]
            = .infinity
        halves[try cache.elementOffset(layer: 0, component: .value, head: 0, position: 3) + 1]
            = .nan

        let two = try LongDepthReport.kvCacheScan(cache, positions: 2)
        XCTAssertEqual(two.nonFiniteCount, 1)
        XCTAssertEqual(two.elementsScanned, 2 * 2 * 2) // K|V × 2 positions × headDim
        // Finite max over the scanned slots: value slab position 1 element 1.
        let expectedMax = Float(
            Float16(Float(try cache.elementOffset(
                layer: 0, component: .value, head: 0, position: 1) + 1) * 0.25))
        XCTAssertEqual(two.maxAbsFinite, expectedMax)

        let all = try LongDepthReport.kvCacheScan(cache, positions: 4)
        XCTAssertEqual(all.nonFiniteCount, 2)
        XCTAssertEqual(all.elementsScanned, 2 * 4 * 2)
        XCTAssertThrowsError(try LongDepthReport.kvCacheScan(cache, positions: 5))
        XCTAssertEqual(try LongDepthReport.kvCacheScan(cache, positions: 0).elementsScanned, 0)
    }

    func testDistinctTokenCountOverTail() {
        XCTAssertEqual(LongDepthReport.distinctTokens(in: [1, 2, 3, 1, 2, 3], tail: 4), 3)
        XCTAssertEqual(LongDepthReport.distinctTokens(in: [1, 2, 3, 1, 2, 3], tail: 2), 2)
        XCTAssertEqual(LongDepthReport.distinctTokens(in: [7, 7, 7], tail: 100), 1)
        XCTAssertEqual(LongDepthReport.distinctTokens(in: [], tail: 100), 0)
    }

    func testQuantilesOfSortedMargins() {
        let q = LongDepthReport.quantiles([5, 1, 4, 2, 3])
        XCTAssertEqual(q.min, 1)
        XCTAssertEqual(q.p1, 1)
        XCTAssertEqual(q.p10, 1)
        XCTAssertEqual(q.median, 3)
        let empty = LongDepthReport.quantiles([])
        XCTAssertTrue(empty.min.isNaN && empty.median.isNaN)
    }

    func testExcerptHeadAndTail() {
        XCTAssertEqual(LongDepthReport.excerpt("abcdef", head: 3), "abc…")
        XCTAssertEqual(LongDepthReport.excerpt("abcdef", tail: 3), "…def")
        XCTAssertEqual(LongDepthReport.excerpt("ab", head: 3), "ab")
        XCTAssertEqual(LongDepthReport.excerpt("a\nb", head: 3), "a⏎b")
    }
}

// MARK: - The opt-in long-depth free-run REPORT (LD-1; never a gate)

/// LD-1 long-depth GPU free-run report (REPORT, never a gate — the
/// 2026-08-23 rationale: with per-logit deviation legitimately up to
/// 2⁻⁵·M, any near-tie step can flip and permanently fork a self-fed
/// trajectory). Free-runs the production DEFAULT path (packed q4g64, fused
/// decode, tiled prefill) to ~3900 generated tokens inside the pinned 4096
/// context, one prompt at a time, and records:
///
/// - every step: non-finite-logit count (the fp16-overflow tripwire), the
///   top-1/top-2 margin and the suite's ε_tie = 2⁻⁴·M64 on the GPU's own
///   logits (fork-eligible steps), dual timing (wall over the whole run;
///   GPU Σ of the per-step timing records — the prefill call contributes
///   its last chunk only, so the Σ is a decode-side figure; DIAGNOSTIC,
///   never a row);
/// - after the run: a scan of the fp16 KV cache for non-finite entries and
///   its max finite magnitude against the fp16 range;
/// - at sampled checkpoints (every N steps + the final step): the CPU-quant
///   reference teacher-forced on the GPU's OWN trajectory prefix, compared
///   with the suite's metrics (full-vocab |Δ| vs 2⁻⁵·M_step, top-64 at the
///   CPU indices vs 2⁻⁵·M64, float64 fingerprints, tie-aware top-1) as
///   descriptive classifications — the oracle strategy LD-1 chose (a
///   per-step live oracle is quadratic in depth: hours; one forward per
///   checkpoint is linear: minutes). "Divergence" is reported twice: the
///   first checkpoint where the CPU-quant argmax over the GPU prefix differs
///   from the GPU's token at all, and the first where it differs BEYOND the
///   tie band (the bug-scale signal).
///
/// Opt-in via QWEN_LONG_DEPTH_REPORT=1 (release build; detached via nohup —
/// the CPU-quant oracle materializes ~7 GB). Knobs: QWEN_LONG_DEPTH_STEPS
/// (default 3900), QWEN_LONG_DEPTH_CHECKPOINT_EVERY (default 512),
/// QWEN_LONG_DEPTH_PROMPTS (comma list; default all five fixtures),
/// QWEN_LONG_DEPTH_REPORT_FILE (the DEV-2 lesson: the runner swallows
/// stdout — the file is the artifact DECISIONS.md quotes).
final class LongDepthFreeRunReportTests: XCTestCase {

    func testLongDepthFreeRunReport() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["QWEN_LONG_DEPTH_REPORT"] == "1" else {
            throw XCTSkip(
                "long-depth free-run report is opt-in: set QWEN_LONG_DEPTH_REPORT=1 "
                + "(LD-1 — recorded in DECISIONS.md, never a gate)")
        }
        try SharedQuantGPUModel.skipUnlessReady()
        let steps = try LongDepthReport.positiveInt(
            env["QWEN_LONG_DEPTH_STEPS"], name: "QWEN_LONG_DEPTH_STEPS", default: 3900)
        let every = try LongDepthReport.positiveInt(
            env["QWEN_LONG_DEPTH_CHECKPOINT_EVERY"],
            name: "QWEN_LONG_DEPTH_CHECKPOINT_EVERY", default: 512)
        let prompts = try LongDepthReport.promptList(env["QWEN_LONG_DEPTH_PROMPTS"])
        let schedule = try LongDepthReport.checkpointSchedule(steps: steps, every: every)
        let maxContext = LongDepthReport.pinnedContext

        let packed = try PackedCheckpoint(
            path: SharedQuantModel.packedURL.path,
            expectedRevision: SharedCheckpoint.pinnedRevision)
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        // The production DEFAULT: no path arguments (fused decode, tiled
        // prefill, query-tiled SDPA, C = min(512, maxContext)).
        let gpu = try GPUModel(
            packed: packed, config: config,
            context: try SharedGPUModel.metalContext(), maxContext: maxContext)
        let tokenizer = try await TextTokenizer(modelFolder: SharedCheckpoint.modelsDir)

        var report: [String] = []
        func emit(_ line: String) {
            print(line)
            report.append(line)
        }
        func flush() throws {
            if let path = env["QWEN_LONG_DEPTH_REPORT_FILE"] {
                try report.joined(separator: "\n").appending("\n")
                    .write(toFile: path, atomically: true, encoding: .utf8)
            }
        }

        emit("=== LD-1 long-depth free-running greedy report "
            + "(\(steps) steps × \(prompts.count) prompts, maxContext \(maxContext), "
            + "GPU default path vs sampled CPU-quant checkpoints) ===")
        emit("path: weights \(gpu.weightsFormat) kernel \(gpu.kernelPath.rawValue) "
            + "prefill \(gpu.prefillPath.rawValue)/\(gpu.prefillAttention.rawValue) "
            + "C=\(gpu.prefillChunkSize); checkpoints every \(every) + final "
            + "(\(schedule.count) per prompt); species (descriptive, not gates): "
            + "full 2⁻⁵·M_step, top-64 2⁻⁵·M64, lse/mean 2⁻⁵·M64, std 2⁻⁴·M64, "
            + "ε_tie 2⁻⁴·M64")
        emit("host: \(LongDepthReport.hostDescription())")
        try flush()

        for prompt in prompts {
            let ids = try SharedCheckpoint.promptFixture(prompt).inputIds
            guard ids.count + steps <= maxContext else {
                XCTFail("\(prompt): \(ids.count) prompt tokens + \(steps) steps exceed "
                    + "the pinned context \(maxContext) — lower QWEN_LONG_DEPTH_STEPS")
                continue
            }
            emit("--- prompt: \(prompt) (\(ids.count) tokens; final depth "
                + "\(ids.count + steps - 1))")
            gpu.reset()

            // The GPU free run: logits-observing loop (`generate`, the same
            // entry point every prior report used; token-identical to the
            // selecting path by the P4-8 contract).
            var stepStats: [LongDepthReport.StepStats] = []
            stepStats.reserveCapacity(steps)
            var gpuLogitsAtCheckpoint: [Int: [Float]] = [:]
            let checkpointSet = Set(schedule)
            var gpuSeconds = 0.0
            let wallStart = Date()
            let tokens = try DecodeLoop(model: gpu, maxContext: maxContext).generate(
                promptIds: ids, maxNewTokens: steps, eosTokenIds: [],
                onStep: { step, logits, _ in
                    stepStats.append(LongDepthReport.StepStats(logits))
                    gpuSeconds += gpu.lastStepTiming?.gpuDuration ?? 0
                    if checkpointSet.contains(step + 1) {
                        gpuLogitsAtCheckpoint[step + 1] = logits
                    }
                })
            let wallSeconds = Date().timeIntervalSince(wallStart)
            XCTAssertEqual(tokens.count, steps, "\(prompt): the run must reach \(steps)")
            let cachedDepth = gpu.cachedTokens.count

            // fp16-overflow tripwire 1: non-finite logits.
            let nonFiniteTotal = stepStats.reduce(0) { $0 + $1.nonFiniteCount }
            let nonFiniteSteps = stepStats.filter { $0.nonFiniteCount > 0 }.count
            let firstNonFinite = stepStats.firstIndex { $0.nonFiniteCount > 0 }
            emit("gpu run: \(tokens.count) tokens, wall \(fmt(wallSeconds)) s, "
                + "GPU Σ \(fmt(gpuSeconds)) s, "
                + "\(fmt(Double(tokens.count) / wallSeconds)) tok/s overall; "
                + "cache depth \(cachedDepth)")
            emit("non-finite logits: \(nonFiniteTotal) entries over \(nonFiniteSteps) "
                + "steps; first step "
                + (firstNonFinite.map { "\($0 + 1)" } ?? "none"))

            // fp16-overflow tripwire 2: the KV cache itself.
            let scan = try LongDepthReport.kvCacheScan(gpu.kvCache, positions: cachedDepth)
            emit("kv cache scan (\(cachedDepth) positions, \(scan.elementsScanned) fp16 "
                + "entries): non-finite \(scan.nonFiniteCount), max |K/V| "
                + "\(fmt(Double(scan.maxAbsFinite))) (fp16 max 65504; headroom "
                + "\(fmt(65504 / Double(max(scan.maxAbsFinite, 1e-30))))×)")

            // Fork-eligibility on the GPU's own logits across depth.
            let margins = stepStats.map(\.margin)
            let q = LongDepthReport.quantiles(margins)
            let forkEligible = stepStats.filter(\.isForkEligible).count
            let exactTies = stepStats.filter { $0.margin == 0 }.count
            emit("top-1 margin: min \(fmt(q.min)) p1 \(fmt(q.p1)) p10 \(fmt(q.p10)) "
                + "median \(fmt(q.median)); fork-eligible steps (margin < 2⁻⁴·M64) "
                + "\(forkEligible) of \(steps); exact ties \(exactTies)")
            let tail = 256
            emit("degeneracy: \(LongDepthReport.distinctTokens(in: tokens, tail: tail)) "
                + "distinct tokens in the last \(min(tail, tokens.count))")
            let text = tokenizer.decode(tokens, skipSpecialTokens: false)
            emit("text head: \(LongDepthReport.excerpt(text, head: 240))")
            emit("text tail: \(LongDepthReport.excerpt(text, tail: 240))")
            try flush()

            // Sampled CPU-quant checkpoints on the GPU's own trajectory.
            let cpu = try SharedQuantModel.model()
            var firstArgmaxDiff: Int?
            var firstBeyondBand: Int?
            var checkpointsWithinAllSpecies = 0
            for k in schedule {
                let prefix = ids + tokens.prefix(k - 1)
                let cpuStart = Date()
                let cpuLogits = try cpu.lastPositionLogits(ids: prefix)
                let cpuSeconds = Date().timeIntervalSince(cpuStart)
                let gpuLogits = try XCTUnwrap(gpuLogitsAtCheckpoint[k])
                let c = LongDepthReport.CheckpointComparison(gpu: gpuLogits, cpu: cpuLogits)
                if !c.argmaxAgrees, firstArgmaxDiff == nil { firstArgmaxDiff = k }
                if !c.tieAwareTop1Holds, firstBeyondBand == nil { firstBeyondBand = k }
                if c.fullWithinSpecies && c.top64WithinSpecies
                    && c.fingerprintsWithinSpecies && c.tieAwareTop1Holds {
                    checkpointsWithinAllSpecies += 1
                }
                let verdict = c.argmaxAgrees ? "agree"
                    : (c.tieAwareTop1Holds ? "FLIP in tie band" : "FLIP BEYOND band")
                emit("checkpoint step \(k) depth \(prefix.count): "
                    + "full |Δ| \(fmtE(c.worstFull)) at \(c.worstFullIndex) "
                    + "vs 2⁻⁵·M_step \(fmtE(c.fullSpecies)) "
                    + "[\(c.fullWithinSpecies ? "within" : "OUTSIDE")]; "
                    + "top-64 |Δ| \(fmtE(c.worstTop64)) rank \(c.worstTop64Rank) "
                    + "vs \(fmtE(c.top64Species)) "
                    + "[\(c.top64WithinSpecies ? "within" : "OUTSIDE")]; "
                    + "Δlse \(fmtE(c.dLogsumexp)) Δmean \(fmtE(c.dMean)) "
                    + "Δstd \(fmtE(c.dStd)) "
                    + "[\(c.fingerprintsWithinSpecies ? "within" : "OUTSIDE")]; "
                    + "argmax gpu \(c.gpuArgmax) vs cpu \(c.cpuTop1)/\(c.cpuTop2) "
                    + "margin \(fmt(c.cpuMargin)) ε \(fmt(c.tieEpsilon)) "
                    + "[\(verdict)]; "
                    + "cpu forward \(fmt(cpuSeconds)) s")
                try flush()
            }
            emit("divergence (any argmax difference at a checkpoint): "
                + (firstArgmaxDiff.map { "first at step \($0)" } ?? "none"))
            emit("divergence beyond the tie band (bug-scale): "
                + (firstBeyondBand.map { "first at step \($0)" } ?? "none"))
            emit("checkpoints within every species: \(checkpointsWithinAllSpecies) "
                + "of \(schedule.count)")
            try flush()
        }
        gpu.reset()
        emit("=== end of LD-1 report ===")
        try flush()
    }

    private func fmt(_ value: Double) -> String { String(format: "%.3f", value) }
    private func fmtE(_ value: Float) -> String { String(format: "%.3e", value) }
    private func fmtE(_ value: Double) -> String { String(format: "%.3e", value) }
}
