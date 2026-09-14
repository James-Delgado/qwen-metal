import Foundation

/// OA-1 (seeded by P4-9): the overhead-anatomy diagnostic harness the app's
/// benchmark screen exports — engine-side so the interleave policy, the
/// aggregation, and the report formatting are testable without a device
/// (AttributionHarness precedent, P4-1).
///
/// DIAGNOSTIC ONLY — the numbers this produces are never benchmark rows.
/// The run round-robins three arms per decode forward on the SAME greedy
/// stream (every arm is a token-selecting step under the P4-8
/// exact-equality contract, so the arms cannot diverge):
///   0. production `stepSelectingToken` — the reference wall−GPU overhead
///      and dispatch count the gate consumes;
///   1. anatomy step — the P4-9 span split (encode / commit /
///      commit→GPU-start / GPU / completion-wakeup);
///   2. anatomy step with unretained references — the P4-9 cheap
///      submission experiment, so the device answers the retention
///      question the Mac already answered (zero effect).

/// One overhead-anatomy run's raw results.
public struct OverheadAnatomyRunResult: Sendable {
    public let weightsFormat: WeightsFormat
    public let kernelPath: GPUModel.KernelPath
    public let promptTokenCount: Int
    /// Anatomy records from the retained-references arm.
    public let anatomies: [OverheadAnatomy]
    /// Anatomy records from the unretained-references experiment arm.
    public let unretainedAnatomies: [OverheadAnatomy]
    /// Production wall−GPU overhead per reference step (the gate metric).
    public let productionOverheadSeconds: [Double]
    /// The stable production dispatches/token, nil if the count varied.
    public let productionDispatchCount: Int?
    /// The stable anatomy-arm dispatches/token (both arms), nil if varied.
    public let anatomyDispatchCount: Int?
    /// Cache positions of the first and last decode forwards.
    public let firstDecodePosition: Int
    public let lastDecodePosition: Int
    /// Greedy-decoded token ids (diagnostic context only).
    public let generatedTokenIds: [Int]

    public init(
        weightsFormat: WeightsFormat, kernelPath: GPUModel.KernelPath,
        promptTokenCount: Int, anatomies: [OverheadAnatomy],
        unretainedAnatomies: [OverheadAnatomy],
        productionOverheadSeconds: [Double], productionDispatchCount: Int?,
        anatomyDispatchCount: Int?, firstDecodePosition: Int,
        lastDecodePosition: Int, generatedTokenIds: [Int]
    ) {
        self.weightsFormat = weightsFormat
        self.kernelPath = kernelPath
        self.promptTokenCount = promptTokenCount
        self.anatomies = anatomies
        self.unretainedAnatomies = unretainedAnatomies
        self.productionOverheadSeconds = productionOverheadSeconds
        self.productionDispatchCount = productionDispatchCount
        self.anatomyDispatchCount = anatomyDispatchCount
        self.firstDecodePosition = firstDecodePosition
        self.lastDecodePosition = lastDecodePosition
        self.generatedTokenIds = generatedTokenIds
    }

    /// One span's aggregate over an arm's records — the testable unit the
    /// export table is built from. Labels match the P4-9 DECISIONS table so
    /// the device entry lines up column-for-column with the Mac one.
    public struct SpanSummary: Sendable, Equatable {
        public let label: String
        public let medianSeconds: Double
        public let minSeconds: Double
        public let maxSeconds: Double

        public init(label: String, medianSeconds: Double,
                    minSeconds: Double, maxSeconds: Double) {
            self.label = label
            self.medianSeconds = medianSeconds
            self.minSeconds = minSeconds
            self.maxSeconds = maxSeconds
        }
    }

    /// The seven-row span table for one arm's records (empty input → []).
    public static func spanSummaries(
        _ records: [OverheadAnatomy]
    ) -> [SpanSummary] {
        guard !records.isEmpty else { return [] }
        func row(_ label: String, _ values: [Double]) -> SpanSummary {
            let sorted = values.sorted()
            let mid = sorted.count / 2
            let median = sorted.count.isMultiple(of: 2)
                ? (sorted[mid - 1] + sorted[mid]) / 2
                : sorted[mid]
            return SpanSummary(
                label: label, medianSeconds: median,
                minSeconds: sorted.first!, maxSeconds: sorted.last!)
        }
        return [
            row("encode", records.map(\.encodeSeconds)),
            row("commit call", records.map(\.commitSeconds)),
            row("commit->GPU-start", records.map(\.commitToGPUStartSeconds)),
            row("  (schedule stage)", records.map(\.scheduleStageSeconds)),
            row("GPU execution", records.map(\.gpuSeconds)),
            row("wakeup (GPU->CPU)", records.map(\.wakeupSeconds)),
            row("TOTAL wall-GPU", records.map(\.overheadSeconds)),
        ]
    }

    public var medianProductionOverheadSeconds: Double? {
        Self.median(productionOverheadSeconds)
    }

    /// The report text (exportText precedent — engine-side formatting the
    /// app shares; same table shape as the P4-9 Mac sweep so the DECISIONS
    /// device entry mirrors the Mac one).
    public func exportText(
        dateStamp: String, deviceLabel: String, osVersion: String,
        residency: WeightsResidency
    ) -> String {
        var lines: [String] = []
        lines.append(
            "qwen-metal overhead anatomy — DIAGNOSTIC "
            + "(P4-9 span split; never a benchmark row)")
        lines.append("date: \(dateStamp)")
        lines.append("device: \(deviceLabel) (\(osVersion))")
        let kernelDescription = kernelPath == .fused
            ? "fused (P4-7 split-K SDPA + P4-3/P4-6 folds)"
            : "naive (pre-fusion)"
        lines.append(
            "engine: weights \(weightsFormat.rawValue), residency "
            + "\(residency.rawValue), \(kernelDescription) kernel structure")
        lines.append(
            "prompt: \(promptTokenCount) tokens; decode forwards: "
            + "\(productionOverheadSeconds.count) production + "
            + "\(anatomies.count) anatomy + \(unretainedAnatomies.count) "
            + "unretained, round-robin (cache depth "
            + "\(firstDecodePosition)-\(lastDecodePosition))")

        if let production = medianProductionOverheadSeconds {
            let dispatches = productionDispatchCount.map(String.init)
                ?? "UNSTABLE"
            lines.append(String(
                format: "production reference: median wall-GPU %.4f ms/token "
                    + "@ %@ dispatches (one command buffer)",
                production * 1000, dispatches))
        }
        let anatomyDispatches = anatomyDispatchCount.map(String.init)
            ?? "UNSTABLE"
        func appendTable(_ header: String, _ records: [OverheadAnatomy]) {
            lines.append(header)
            for row in Self.spanSummaries(records) {
                lines.append(String(
                    format: "  %@ median %8.4f ms   min %8.4f   max %8.4f",
                    row.label.padding(toLength: 22, withPad: " ",
                                      startingAt: 0),
                    row.medianSeconds * 1000, row.minSeconds * 1000,
                    row.maxSeconds * 1000))
            }
        }
        appendTable(
            "-- anatomy arm (retained references, \(anatomyDispatches) "
            + "dispatches) --", anatomies)
        appendTable(
            "-- anatomy arm (UNRETAINED references experiment) --",
            unretainedAnatomies)
        lines.append(
            "note: spans telescope to wall-GPU (encode + commit + "
            + "commit->GPU-start + wakeup); compare against the P4-9 Mac "
            + "split and the P4-5 device affine fit "
            + "(~1.17 ms fixed + ~1.5 us/dispatch).")
        return lines.joined(separator: "\n")
    }

    /// Median with the collector's even-count convention (mean of middle
    /// two). nil on empty input.
    private static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid]
    }
}

/// Drives the round-robin production/anatomy/unretained greedy decode over
/// a `GPUModel` (AttributionRunner shape: reset, prefill with production
/// steps, then cycle arms per decode forward).
public struct OverheadAnatomyRunner {
    public let gpuModel: GPUModel
    public let maxContext: Int
    public let eosTokenIds: Set<Int>

    public init(
        gpuModel: GPUModel, maxContext: Int, eosTokenIds: Set<Int> = []
    ) {
        self.gpuModel = gpuModel
        self.maxContext = maxContext
        self.eosTokenIds = eosTokenIds
    }

    /// - Parameters:
    ///   - decodeTokens: requested decode forwards (all three arms
    ///     combined; arm = forward index mod 3). The first decode forward
    ///     runs the LAST prompt token — the standard incremental shape.
    ///   - shouldStop: polled at token boundaries (app Stop control).
    ///   - onStep: per-decode-step progress hook (step index).
    public func run(
        promptIds: [Int], decodeTokens: Int,
        shouldStop: (() -> Bool)? = nil,
        onStep: ((Int) -> Void)? = nil
    ) throws -> OverheadAnatomyRunResult {
        guard !promptIds.isEmpty else { throw DecodeError.emptyPrompt }
        guard decodeTokens >= 1 else {
            throw DecodeError.invalidMaxNewTokens(decodeTokens)
        }
        guard promptIds.count + decodeTokens <= maxContext else {
            throw ModelError.badInput(detail:
                "overhead-anatomy run needs \(promptIds.count) prompt + "
                + "\(decodeTokens) decode tokens but the context limit is "
                + "\(maxContext)")
        }

        // Deterministic depth accounting: always start from an empty cache.
        gpuModel.reset()
        for token in promptIds.dropLast() {
            try gpuModel.step(token: token, computeLogits: false)
        }

        var current = promptIds[promptIds.count - 1]
        var anatomies: [OverheadAnatomy] = []
        var unretainedAnatomies: [OverheadAnatomy] = []
        var productionOverheadSeconds: [Double] = []
        var productionDispatchCounts: Set<Int> = []
        var anatomyDispatchCounts: Set<Int> = []
        var generated: [Int] = []
        let firstDecodePosition = gpuModel.cachedTokens.count
        var lastDecodePosition = firstDecodePosition

        for step in 0..<decodeTokens {
            if shouldStop?() == true { break }
            lastDecodePosition = gpuModel.cachedTokens.count
            let next: Int
            switch step % 3 {
            case 0:
                next = try gpuModel.stepSelectingToken(token: current)
                if let timing = gpuModel.lastStepTiming {
                    productionOverheadSeconds.append(timing.dispatchOverhead)
                }
                if let count = gpuModel.lastStepDispatchCount {
                    productionDispatchCounts.insert(count)
                }
            case 1:
                let result = try gpuModel.anatomyStepSelectingToken(
                    token: current)
                anatomies.append(result.anatomy)
                anatomyDispatchCounts.insert(result.dispatchCount)
                next = result.token
            default:
                let result = try gpuModel.anatomyStepSelectingToken(
                    token: current, unretainedReferences: true)
                unretainedAnatomies.append(result.anatomy)
                anatomyDispatchCounts.insert(result.dispatchCount)
                next = result.token
            }
            generated.append(next)
            onStep?(step)
            if eosTokenIds.contains(next) { break }
            current = next
        }

        return OverheadAnatomyRunResult(
            weightsFormat: gpuModel.weightsFormat,
            kernelPath: gpuModel.kernelPath,
            promptTokenCount: promptIds.count,
            anatomies: anatomies,
            unretainedAnatomies: unretainedAnatomies,
            productionOverheadSeconds: productionOverheadSeconds,
            productionDispatchCount: productionDispatchCounts.count == 1
                ? productionDispatchCounts.first : nil,
            anatomyDispatchCount: anatomyDispatchCounts.count == 1
                ? anatomyDispatchCounts.first : nil,
            firstDecodePosition: firstDecodePosition,
            lastDecodePosition: lastDecodePosition,
            generatedTokenIds: generated)
    }
}
