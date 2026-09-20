import Foundation

/// PF-1 (phase-5.md D4 "reuse first, batch if it pays" + the PLAN "do not
/// guess at bottlenecks" rule): the DIAGNOSTIC prefill attribution harness
/// the CLI `attribute --mode prefill` and the app's attribution picker
/// share. Engine-side so the interleave policy and the report formatting
/// are testable without a device (the P4-1 AttributionHarness pattern).
///
/// DIAGNOSTIC ONLY — the numbers this produces are never benchmark rows.
/// The run alternates attributed prefills (per-class command-buffer splits
/// inside each chunk, `GPUModel.attributedPrefill`) with production
/// prefills (one command buffer per chunk, the D1 span) of the SAME prompt
/// from an empty cache, so the report can cross-check the class-time sum
/// against the production GPU time — the pre-committed [0.5×, 2.0×] band
/// (DECISIONS.md 2026-09-18 PF-1 bounds, the P4-1 bounds verbatim).

/// One attributed prefill: per-chunk class-split attributions in chunk
/// order, plus the chunk sizes (the last chunk may be ragged).
public struct PrefillAttribution: Sendable {
    public let chunks: [TokenAttribution]
    public let chunkSizes: [Int]

    public init(chunks: [TokenAttribution], chunkSizes: [Int]) {
        precondition(chunks.count == chunkSizes.count)
        self.chunks = chunks
        self.chunkSizes = chunkSizes
    }

    /// Prompt positions processed (Σ chunk sizes).
    public var positionCount: Int { chunkSizes.reduce(0, +) }

    /// GPU time over one class, summed over every chunk's segments (exact
    /// bookkeeping — pinned by test against the segments).
    public func classGPUSeconds(_ kernelClass: KernelClass) -> Double {
        chunks.reduce(0) { $0 + $1.classGPUSeconds(kernelClass) }
    }

    public func classDispatchCount(_ kernelClass: KernelClass) -> Int {
        chunks.reduce(0) { $0 + $1.classDispatchCount(kernelClass) }
    }

    /// Total attributed GPU time over all chunks and classes.
    public var gpuSecondsTotal: Double {
        chunks.reduce(0) { $0 + $1.gpuSecondsTotal }
    }

    public var dispatchCountTotal: Int {
        chunks.reduce(0) { $0 + $1.dispatchCountTotal }
    }

    /// Σ per-chunk GPU windows (first segment start → last segment end,
    /// including inter-buffer gaps within the chunk).
    public var spanSeconds: Double {
        chunks.reduce(0) { $0 + $1.spanSeconds }
    }

    /// Σ per-chunk wall brackets (hard rule 7: wall rides along).
    public var wallSeconds: Double {
        chunks.reduce(0) { $0 + $1.wallSeconds }
    }
}

/// One prefill attribution run's raw results.
public struct PrefillAttributionRunResult: Sendable {
    public let weightsFormat: WeightsFormat
    public let kernelPath: GPUModel.KernelPath
    public let prefillChunkSize: Int
    /// The chunk's causal SDPA kernel (PF-2) — the attention class's
    /// identity on this export.
    public let prefillAttention: GPUModel.PrefillAttention
    public let promptTokenCount: Int
    /// The attributed (class-split) prefills — even run offsets.
    public let attributed: [PrefillAttribution]
    /// Production one-buffer-per-chunk prefill GPU spans (Σ chunk GPU
    /// durations, the D1 span's GPU figure) — odd run offsets.
    public let productionGPUSeconds: [Double]
    /// Production prefill dispatch counts; nil if they varied across runs.
    public let productionDispatchCount: Int?

    public init(
        weightsFormat: WeightsFormat, kernelPath: GPUModel.KernelPath,
        prefillChunkSize: Int, prefillAttention: GPUModel.PrefillAttention,
        promptTokenCount: Int,
        attributed: [PrefillAttribution], productionGPUSeconds: [Double],
        productionDispatchCount: Int?
    ) {
        self.weightsFormat = weightsFormat
        self.kernelPath = kernelPath
        self.prefillChunkSize = prefillChunkSize
        self.prefillAttention = prefillAttention
        self.promptTokenCount = promptTokenCount
        self.attributed = attributed
        self.productionGPUSeconds = productionGPUSeconds
        self.productionDispatchCount = productionDispatchCount
    }

    /// Median per-attributed-prefill GPU seconds for one kernel class.
    public func medianClassGPUSeconds(_ kernelClass: KernelClass) -> Double? {
        BenchMath.medianOrNil(attributed.map { $0.classGPUSeconds(kernelClass) })
    }

    /// Median class-time total per attributed prefill.
    public var medianGPUSecondsTotal: Double? {
        BenchMath.medianOrNil(attributed.map(\.gpuSecondsTotal))
    }

    public var medianProductionGPUSeconds: Double? {
        BenchMath.medianOrNil(productionGPUSeconds)
    }

    /// The report text (exportText precedent — engine-side formatting the
    /// CLI prints and the app shares).
    public func exportText(
        dateStamp: String, deviceLabel: String, osVersion: String,
        residency: WeightsResidency
    ) -> String {
        var lines: [String] = []
        lines.append(
            "qwen-metal PREFILL per-stage GPU attribution — DIAGNOSTIC "
            + "(PF-1 opt-in mode; never a benchmark row)")
        lines.append("date: \(dateStamp)")
        lines.append("device: \(deviceLabel) (\(osVersion))")
        lines.append(
            "engine: weights \(weightsFormat.rawValue), residency "
            + "\(residency.rawValue), kernels \(kernelPath.rawValue), "
            + "prefill tiled (C=\(prefillChunkSize)), attention "
            + prefillAttention.rawValue)
        let chunkCount = attributed.first?.chunks.count ?? 0
        lines.append(
            "prompt: \(promptTokenCount) tokens in \(chunkCount) chunk(s); "
            + "prefills: \(attributed.count) attributed + "
            + "\(productionGPUSeconds.count) production, interleaved, each "
            + "from an empty cache")

        let classMedians = KernelClass.allCases.map {
            ($0, medianClassGPUSeconds($0) ?? 0)
        }
        let classSum = classMedians.reduce(0) { $0 + $1.1 }
        lines.append(
            "per-class GPU time, median ms per attributed prefill "
            + "(share of class-sum):")
        let labelWidth = KernelClass.allCases.map { $0.rawValue.count }.max() ?? 0
        for (kernelClass, seconds) in classMedians {
            let label = kernelClass.rawValue.padding(
                toLength: labelWidth, withPad: " ", startingAt: 0)
            let share = classSum > 0 ? seconds / classSum * 100 : 0
            let dispatches = attributed.first?.classDispatchCount(kernelClass) ?? 0
            lines.append(String(
                format: "  %@  %9.2f ms  (%5.1f%%)  %d dispatches",
                label, seconds * 1000, share, dispatches))
        }

        if let total = medianGPUSecondsTotal {
            let span = BenchMath.medianOrNil(attributed.map(\.spanSeconds)) ?? 0
            let wall = BenchMath.medianOrNil(attributed.map(\.wallSeconds)) ?? 0
            lines.append(String(
                format: "attributed: class-sum median %.1f ms/prefill, GPU span "
                    + "median %.1f ms, wall median %.1f ms",
                total * 1000, span * 1000, wall * 1000))
        }
        if let production = medianProductionGPUSeconds {
            let dispatches = productionDispatchCount.map(String.init)
                ?? "UNSTABLE"
            lines.append(String(
                format: "production reference: median GPU %.1f ms/prefill @ %@ "
                    + "dispatches (one command buffer per chunk) = %.2f tok/s "
                    + "on GPU time",
                production * 1000, dispatches,
                Double(promptTokenCount) / production))
            if let total = medianGPUSecondsTotal, production > 0 {
                lines.append(String(
                    format: "sanity: class-sum / production GPU = %.2f "
                        + "(pre-committed band 0.50-2.00, DECISIONS.md "
                        + "2026-09-18)",
                    total / production))
            }
        }
        lines.append(
            "note: split-mode sums include per-command-buffer cost; the "
            + "production prefill span (P5-1 D1) is unchanged and lives in "
            + "generate/benchmark reports.")
        return lines.joined(separator: "\n")
    }
}

/// Drives interleaved attributed/production tiled prefills of one prompt
/// over a `GPUModel`: even run offsets run `attributedPrefill`, odd
/// offsets run the production `lastPositionLogits` from an empty cache.
public struct PrefillAttributionRunner {
    public let gpuModel: GPUModel
    public let maxContext: Int

    public init(gpuModel: GPUModel, maxContext: Int) {
        self.gpuModel = gpuModel
        self.maxContext = maxContext
    }

    /// - Parameters:
    ///   - runs: total prefills (attributed + production combined, ≥ 2 so
    ///     both arms exist).
    ///   - shouldStop: polled between prefills (app Stop control).
    ///   - onRun: per-prefill progress hook (run index).
    public func run(
        promptIds: [Int], runs: Int,
        shouldStop: (() -> Bool)? = nil,
        onRun: ((Int) -> Void)? = nil
    ) throws -> PrefillAttributionRunResult {
        guard promptIds.count > 1 else {
            throw ModelError.badInput(detail:
                "prefill attribution needs a multi-token prompt (got "
                + "\(promptIds.count) tokens)")
        }
        guard runs >= 2 else {
            throw ModelError.badInput(detail:
                "prefill attribution needs runs >= 2 (one attributed + one "
                + "production), got \(runs)")
        }
        guard promptIds.count <= maxContext else {
            throw ModelError.badInput(detail:
                "prefill attribution prompt of \(promptIds.count) tokens "
                + "exceeds the context limit \(maxContext)")
        }
        guard gpuModel.prefillPath == .tiled else {
            throw ModelError.badInput(detail:
                "prefill attribution needs the tiled prefill path "
                + "(phase-5.md D5) — this model runs sequential prefill")
        }

        var attributed: [PrefillAttribution] = []
        var productionGPUSeconds: [Double] = []
        var productionDispatchCounts: Set<Int> = []
        for run in 0..<runs {
            if shouldStop?() == true { break }
            if run.isMultiple(of: 2) {
                attributed.append(
                    try gpuModel.attributedPrefill(ids: promptIds).attribution)
            } else {
                gpuModel.reset()
                _ = try gpuModel.lastPositionLogits(ids: promptIds)
                if let span = gpuModel.lastCallSpan {
                    productionGPUSeconds.append(span.gpuSeconds)
                    productionDispatchCounts.insert(span.dispatchCount)
                }
            }
            onRun?(run)
        }
        gpuModel.reset()

        return PrefillAttributionRunResult(
            weightsFormat: gpuModel.weightsFormat,
            kernelPath: gpuModel.kernelPath,
            prefillChunkSize: gpuModel.prefillChunkSize,
            prefillAttention: gpuModel.prefillAttention,
            promptTokenCount: promptIds.count,
            attributed: attributed,
            productionGPUSeconds: productionGPUSeconds,
            productionDispatchCount: productionDispatchCounts.count == 1
                ? productionDispatchCounts.first : nil)
    }
}
