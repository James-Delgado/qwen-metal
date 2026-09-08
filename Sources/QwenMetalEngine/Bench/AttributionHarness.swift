import Foundation

/// P4-1 (phase-4.md D1 + instrumentation deliverables): the diagnostic
/// attribution harness the CLI `attribute` subcommand and the app's
/// attribution mode share. Engine-side so the interleave policy and the
/// report formatting are testable without a device.
///
/// DIAGNOSTIC ONLY — the numbers this produces are never benchmark rows.
/// The run interleaves attributed (split command buffers) and production
/// (one command buffer) decode steps so the report can cross-check the
/// class-time sum against the production GPU time at adjacent cache depths
/// (the pre-committed [0.5×, 2.0×] sanity band, DECISIONS.md 2026-09-08).

/// One attribution run's raw results.
public struct AttributionRunResult: Sendable {
    public let weightsFormat: WeightsFormat
    public let promptTokenCount: Int
    /// Per-token attributions from the even-offset (attributed) steps.
    public let attributed: [TokenAttribution]
    /// Production single-command-buffer GPU durations from the odd-offset
    /// steps (the cross-check reference).
    public let productionGPUSeconds: [Double]
    /// The stable production dispatches/token, nil if the count varied.
    public let productionDispatchCount: Int?
    /// Cache positions of the first and last decode forwards.
    public let firstDecodePosition: Int
    public let lastDecodePosition: Int
    /// Greedy-decoded token ids (diagnostic context only).
    public let generatedTokenIds: [Int]

    public init(
        weightsFormat: WeightsFormat, promptTokenCount: Int,
        attributed: [TokenAttribution], productionGPUSeconds: [Double],
        productionDispatchCount: Int?, firstDecodePosition: Int,
        lastDecodePosition: Int, generatedTokenIds: [Int]
    ) {
        self.weightsFormat = weightsFormat
        self.promptTokenCount = promptTokenCount
        self.attributed = attributed
        self.productionGPUSeconds = productionGPUSeconds
        self.productionDispatchCount = productionDispatchCount
        self.firstDecodePosition = firstDecodePosition
        self.lastDecodePosition = lastDecodePosition
        self.generatedTokenIds = generatedTokenIds
    }

    /// Median per-attributed-token GPU seconds for one kernel class.
    public func medianClassGPUSeconds(_ kernelClass: KernelClass) -> Double? {
        Self.median(attributed.map { $0.classGPUSeconds(kernelClass) })
    }

    /// Median class-time total per attributed token.
    public var medianGPUSecondsTotal: Double? {
        Self.median(attributed.map(\.gpuSecondsTotal))
    }

    public var medianProductionGPUSeconds: Double? {
        Self.median(productionGPUSeconds)
    }

    /// The report text (exportText precedent — engine-side formatting the
    /// CLI prints and the app shares).
    public func exportText(
        dateStamp: String, deviceLabel: String, osVersion: String,
        residency: WeightsResidency
    ) -> String {
        var lines: [String] = []
        lines.append(
            "qwen-metal per-stage GPU attribution — DIAGNOSTIC "
            + "(D1 opt-in mode; never a benchmark row)")
        lines.append("date: \(dateStamp)")
        lines.append("device: \(deviceLabel) (\(osVersion))")
        lines.append(
            "engine: weights \(weightsFormat.rawValue), residency "
            + "\(residency.rawValue), naive (pre-fusion) kernel structure")
        lines.append(
            "prompt: \(promptTokenCount) tokens; decode steps: "
            + "\(attributed.count) attributed + \(productionGPUSeconds.count) "
            + "production, interleaved (cache depth "
            + "\(firstDecodePosition)-\(lastDecodePosition))")

        let classMedians = KernelClass.allCases.map {
            ($0, medianClassGPUSeconds($0) ?? 0)
        }
        let classSum = classMedians.reduce(0) { $0 + $1.1 }
        lines.append(
            "per-class GPU time, median ms per attributed token "
            + "(share of class-sum):")
        let labelWidth = KernelClass.allCases.map { $0.rawValue.count }.max() ?? 0
        for (kernelClass, seconds) in classMedians {
            let label = kernelClass.rawValue.padding(
                toLength: labelWidth, withPad: " ", startingAt: 0)
            let share = classSum > 0 ? seconds / classSum * 100 : 0
            lines.append(String(
                format: "  %@  %7.2f ms  (%.1f%%)", label, seconds * 1000, share))
        }

        if let total = medianGPUSecondsTotal {
            let span = Self.median(attributed.map(\.spanSeconds)) ?? 0
            let wall = Self.median(attributed.map(\.wallSeconds)) ?? 0
            lines.append(String(
                format: "attributed: class-sum median %.2f ms/token, GPU span "
                    + "median %.2f ms, wall median %.2f ms",
                total * 1000, span * 1000, wall * 1000))
        }
        if let production = medianProductionGPUSeconds {
            let dispatches = productionDispatchCount.map(String.init)
                ?? "UNSTABLE"
            lines.append(String(
                format: "production reference: median GPU %.2f ms/token @ %@ "
                    + "dispatches (one command buffer)",
                production * 1000, dispatches))
            if let total = medianGPUSecondsTotal, production > 0 {
                lines.append(String(
                    format: "sanity: class-sum / production GPU = %.2f "
                        + "(pre-committed band 0.50-2.00, DECISIONS.md "
                        + "2026-09-08)",
                    total / production))
            }
        }
        lines.append(
            "note: split-mode sums include per-command-buffer cost; the "
            + "production wall-GPU overhead metric (P2-5) is unchanged and "
            + "lives in generate/benchmark reports.")
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

/// Drives an interleaved attributed/production greedy decode over a
/// `GPUModel`. Prefills the prompt with production steps, then alternates:
/// even decode offsets run `attributedStep`, odd offsets run `step`.
public struct AttributionRunner {
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
    ///   - decodeTokens: requested decode forwards (attributed + production
    ///     combined). The first decode forward runs the LAST prompt token —
    ///     the standard incremental-decode shape.
    ///   - shouldStop: polled at token boundaries (app Stop control).
    ///   - onStep: per-decode-step progress hook (step index).
    public func run(
        promptIds: [Int], decodeTokens: Int,
        shouldStop: (() -> Bool)? = nil,
        onStep: ((Int) -> Void)? = nil
    ) throws -> AttributionRunResult {
        guard !promptIds.isEmpty else { throw DecodeError.emptyPrompt }
        guard decodeTokens >= 1 else {
            throw DecodeError.invalidMaxNewTokens(decodeTokens)
        }
        guard promptIds.count + decodeTokens <= maxContext else {
            throw ModelError.badInput(detail:
                "attribution run needs \(promptIds.count) prompt + "
                + "\(decodeTokens) decode tokens but the context limit is "
                + "\(maxContext)")
        }

        // Deterministic depth accounting: always start from an empty cache.
        gpuModel.reset()
        for token in promptIds.dropLast() {
            try gpuModel.step(token: token, computeLogits: false)
        }

        var current = promptIds[promptIds.count - 1]
        var attributed: [TokenAttribution] = []
        var productionGPUSeconds: [Double] = []
        var productionDispatchCounts: Set<Int> = []
        var generated: [Int] = []
        let firstDecodePosition = gpuModel.cachedTokens.count
        var lastDecodePosition = firstDecodePosition

        for step in 0..<decodeTokens {
            if shouldStop?() == true { break }
            lastDecodePosition = gpuModel.cachedTokens.count
            let logits: [Float]
            if step.isMultiple(of: 2) {
                let (attribution, attributedLogits) = try gpuModel.attributedStep(
                    token: current, computeLogits: true)
                attributed.append(attribution)
                // computeLogits: true always yields logits.
                logits = attributedLogits!
            } else {
                logits = try gpuModel.step(token: current, computeLogits: true)!
                if let timing = gpuModel.lastStepTiming {
                    productionGPUSeconds.append(timing.gpuDuration)
                }
                if let count = gpuModel.lastStepDispatchCount {
                    productionDispatchCounts.insert(count)
                }
            }
            let next = Argmax.firstIndex(logits)
            generated.append(next)
            onStep?(step)
            if eosTokenIds.contains(next) { break }
            current = next
        }

        return AttributionRunResult(
            weightsFormat: gpuModel.weightsFormat,
            promptTokenCount: promptIds.count,
            attributed: attributed,
            productionGPUSeconds: productionGPUSeconds,
            productionDispatchCount: productionDispatchCounts.count == 1
                ? productionDispatchCounts.first : nil,
            firstDecodePosition: firstDecodePosition,
            lastDecodePosition: lastDecodePosition,
            generatedTokenIds: generated)
    }
}
