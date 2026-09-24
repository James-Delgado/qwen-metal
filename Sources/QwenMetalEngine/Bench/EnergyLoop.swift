import Foundation

/// P6-1 (phase-6.md D4 + edge test 1): the energy-mode regenerate loop.
/// The PLAN energy protocol is OPERATOR-bounded — start at exactly 80% SoC,
/// stop at exactly 70% — so, unlike `SustainedLoop` (duration-bounded) and
/// the P2-6 app behavior (Stop aborts the sustained loop WITHOUT a report),
/// the operator stop here ends the in-flight generation at its next token
/// boundary and the loop still returns its result. The per-generation
/// records are the D1 sustained timeline (the thermal chart's data); the
/// cumulative totals feed the energy math, which lives in
/// tools/phase6_analyze.py (P6-3) — never in the engine or the app.
public enum EnergyLoopError: Error, Equatable, CustomStringConvertible {
    /// A generation produced no tokens while no stop was pending — the loop
    /// would spin forever. Loud, never silent (the SustainedLoop rule).
    case emptyGeneration(index: Int)

    public var description: String {
        switch self {
        case .emptyGeneration(let index):
            return "energy loop: generation \(index) produced 0 tokens with "
                + "no stop pending — refusing to spin"
        }
    }
}

/// What ended an energy cycle.
public enum EnergyLoopEnd: String, Sendable, Codable, Equatable {
    /// The protocol's stop: the operator at the 70% SoC mark.
    case operatorStop
    /// The optional safety bound (Mac sanity runs; never the protocol).
    case durationBound
}

public struct EnergyLoopResult: Sendable {
    /// Per-generation metrics, in run order (the D1 timeline).
    public let generations: [GenerationMetrics]
    /// Loop-clock offset (seconds since the cycle started) at which each
    /// generation ended — the thermal chart's x-axis, one per generation.
    public let generationEndOffsetsSeconds: [Double]
    /// Loop-clock span from the cycle start to the final generation's end:
    /// the idle-baseline pro-rata basis (D4).
    public let cycleWallSeconds: Double
    public let endedBy: EnergyLoopEnd
    /// True when the final generation was cut at a token boundary by the
    /// stop (vs ending on EOS / context fill) — flagged, never mixed.
    public let lastGenerationTruncated: Bool

    public init(
        generations: [GenerationMetrics], generationEndOffsetsSeconds: [Double],
        cycleWallSeconds: Double, endedBy: EnergyLoopEnd,
        lastGenerationTruncated: Bool
    ) {
        precondition(
            generations.count == generationEndOffsetsSeconds.count,
            "one end offset per generation")
        self.generations = generations
        self.generationEndOffsetsSeconds = generationEndOffsetsSeconds
        self.cycleWallSeconds = cycleWallSeconds
        self.endedBy = endedBy
        self.lastGenerationTruncated = lastGenerationTruncated
    }

    /// Σ generated tokens over the records — the energy denominator.
    public var totalGeneratedTokens: Int {
        generations.reduce(0) { $0 + $1.generatedTokenCount }
    }

    /// Σ per-generation wall time (prefill + decode of every generation).
    /// ≤ `cycleWallSeconds` by construction (loop glue excluded).
    public var totalGenerationWallSeconds: Double {
        generations.reduce(0) { $0 + $1.wallSeconds }
    }
}

public struct EnergyLoop {
    /// Optional safety bound in seconds. nil = operator-bounded only (the
    /// protocol). The bound exists for Mac sanity runs and is reported as
    /// `endedBy: .durationBound` so it can never masquerade as a cycle.
    public let maxDurationSeconds: Double?

    public init(maxDurationSeconds: Double? = nil) {
        if let maxDurationSeconds {
            precondition(maxDurationSeconds > 0, "maxDurationSeconds must be > 0")
        }
        self.maxDurationSeconds = maxDurationSeconds
    }

    /// Runs generations back-to-back until the operator stop (or the safety
    /// bound) fires. The in-flight generation is stopped at its next token
    /// boundary via the `shouldStop` handed to `generate` and kept as the
    /// final, flagged entry — the result is ALWAYS returned on a stop.
    /// - Parameters:
    ///   - clock: monotonic seconds; injectable for tests.
    ///   - operatorStop: the operator's Stop control, polled at token
    ///     boundaries (never mid-forward).
    ///   - generate: runs ONE generation from the pinned prompt, honoring
    ///     `shouldStop` at token boundaries, and returns its metrics.
    public func run(
        clock: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime },
        operatorStop: @escaping () -> Bool,
        generate: (_ shouldStop: @escaping () -> Bool) throws -> GenerationMetrics
    ) throws -> EnergyLoopResult {
        let start = clock()
        let bound = maxDurationSeconds
        let boundHit = { bound.map { clock() - start >= $0 } ?? false }
        let shouldEnd = { operatorStop() || boundHit() }
        var generations: [GenerationMetrics] = []
        var offsets: [Double] = []
        repeat {
            let metrics = try generate(shouldEnd)
            if metrics.generatedTokenCount == 0, !shouldEnd() {
                throw EnergyLoopError.emptyGeneration(index: generations.count)
            }
            generations.append(metrics)
            offsets.append(clock() - start)
        } while !shouldEnd()
        return EnergyLoopResult(
            generations: generations,
            generationEndOffsetsSeconds: offsets,
            cycleWallSeconds: offsets.last ?? 0,
            // The operator's stop is the protocol's; a coincident bound
            // never relabels it.
            endedBy: operatorStop() ? .operatorStop : .durationBound,
            lastGenerationTruncated:
                generations.last?.stopReason == .stopRequested)
    }
}
