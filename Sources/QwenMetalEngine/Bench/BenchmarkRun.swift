import Foundation

/// P2-6 (docs/phases/phase-2.md D8 + instrumentation deliverables): the
/// engine-side benchmark harness the iOS shell drives. The app stays thin —
/// generation orchestration, metric assembly, and stop-reason accounting all
/// live here so they are testable under `swift test` without a device.

/// Benchmark protocol operational defaults the app and CLI share. Values are
/// precedent-pinned: 640-token burst cap = the P2-5 Mac sanity row's cap
/// (covers the canonical 128–512 window with margin); 5 min = the PLAN.md
/// sustained minimum.
public enum BenchDefaults {
    public static let burstMaxNewTokens = 640
    public static let sustainedMinDurationSeconds: Double = 300
}

/// Why one generation ended. Inferred from the decode outcome — `DecodeLoop`
/// reports tokens, not reasons, and the precedence below mirrors its stop
/// order (EOS and context-fill are checked after the token lands, so they
/// win over a same-step max-tokens/stop-request coincidence).
public enum GenerationStopReason: String, Sendable {
    case eos
    case contextFull
    case maxNewTokens
    case stopRequested
}

/// Aggregates of one generation, in benchmark-row vocabulary. Rates come from
/// the per-token `DecodeTimingCollector` (completion-to-completion spans, the
/// P2-5 semantics) — never recomputed from `wallSeconds`, which includes the
/// sequential prefill and so would understate the decode rate.
public struct GenerationMetrics: Sendable {
    public let promptTokenCount: Int
    public let generatedTokenCount: Int
    /// Whole-generation wall time (prefill + decode), seconds.
    public let wallSeconds: Double
    /// Wall time from generation start to the FIRST generated token's
    /// completion — sequential prefill (spec D6) plus one decode forward,
    /// stated as such wherever it's reported. nil when no token was produced.
    public let prefillSeconds: Double?
    public let stopReason: GenerationStopReason
    /// Per-token aggregates (median GPU/wall/overhead, dispatches/token).
    /// nil when no step records were collected.
    public let timing: DecodeTimingSummary?
    public let overallTokensPerSecond: Double?
    public let canonicalWindowTokensPerSecond: Double?

    public init(
        promptTokenCount: Int, generatedTokenCount: Int, wallSeconds: Double,
        prefillSeconds: Double? = nil,
        stopReason: GenerationStopReason, timing: DecodeTimingSummary?,
        overallTokensPerSecond: Double?, canonicalWindowTokensPerSecond: Double?
    ) {
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.wallSeconds = wallSeconds
        self.prefillSeconds = prefillSeconds
        self.stopReason = stopReason
        self.timing = timing
        self.overallTokensPerSecond = overallTokensPerSecond
        self.canonicalWindowTokensPerSecond = canonicalWindowTokensPerSecond
    }
}

/// One generation's ids + metrics.
public struct GenerationRunResult: Sendable {
    public let tokenIds: [Int]
    public let metrics: GenerationMetrics
}

/// Runs one instrumented generation: `DecodeLoop` + `DecodeTimingCollector`
/// wiring (exactly the CLI's P2-5 hookup, factored engine-side so app and CLI
/// report identical numbers). Generic over the logits source so tests script
/// it; the `GPUModel` convenience init wires the real per-step records.
public struct BenchGenerationRunner {
    public let model: any NextTokenLogitsSource
    public let maxContext: Int
    public let eosTokenIds: Set<Int>
    /// Read after each step; returns the step's dual timing + dispatch count.
    public let stepRecord: (() -> TokenStepRecord?)?

    public init(
        model: any NextTokenLogitsSource, maxContext: Int,
        eosTokenIds: Set<Int>, stepRecord: (() -> TokenStepRecord?)? = nil
    ) {
        self.model = model
        self.maxContext = maxContext
        self.eosTokenIds = eosTokenIds
        self.stepRecord = stepRecord
    }

    /// The production wiring: per-step records straight from the model's
    /// dual timing + dispatch count (hard rule 7 — both clocks, always).
    public init(gpuModel: GPUModel, maxContext: Int, eosTokenIds: Set<Int>) {
        self.init(
            model: gpuModel, maxContext: maxContext, eosTokenIds: eosTokenIds,
            stepRecord: {
                guard let timing = gpuModel.lastStepTiming,
                      let dispatches = gpuModel.lastStepDispatchCount
                else { return nil }
                return TokenStepRecord(timing: timing, dispatchCount: dispatches)
            })
    }

    /// - Parameters:
    ///   - shouldStop: polled at token boundaries (the app's Stop control /
    ///     the sustained loop's duration bound).
    ///   - onToken: per-step UI hook (stepIndex, tokenId), called after the
    ///     step's record is collected.
    public func run(
        promptIds: [Int], maxNewTokens: Int,
        shouldStop: (() -> Bool)? = nil,
        onToken: ((Int, Int) -> Void)? = nil
    ) throws -> GenerationRunResult {
        var collector = DecodeTimingCollector()
        var prefillSeconds: Double?
        let start = Date()
        let generated = try DecodeLoop(model: model, maxContext: maxContext)
            .generate(
                promptIds: promptIds, maxNewTokens: maxNewTokens,
                eosTokenIds: eosTokenIds,
                onStep: { step, _, token in
                    if step == 0 {
                        prefillSeconds = Date().timeIntervalSince(start)
                    }
                    if let record = stepRecord?() { collector.append(record) }
                    onToken?(step, token)
                },
                shouldStop: shouldStop)
        let wallSeconds = Date().timeIntervalSince(start)

        let stopReason: GenerationStopReason
        if let last = generated.last, eosTokenIds.contains(last) {
            stopReason = .eos
        } else if promptIds.count + generated.count >= maxContext {
            stopReason = .contextFull
        } else if generated.count == maxNewTokens {
            stopReason = .maxNewTokens
        } else {
            stopReason = .stopRequested
        }

        return GenerationRunResult(
            tokenIds: generated,
            metrics: GenerationMetrics(
                promptTokenCount: promptIds.count,
                generatedTokenCount: generated.count,
                wallSeconds: wallSeconds,
                prefillSeconds: prefillSeconds,
                stopReason: stopReason,
                timing: collector.summary(),
                overallTokensPerSecond: collector.overallTokensPerSecond(),
                canonicalWindowTokensPerSecond:
                    collector.canonicalWindowTokensPerSecond()))
    }
}
