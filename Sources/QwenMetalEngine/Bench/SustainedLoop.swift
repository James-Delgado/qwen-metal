import Foundation

/// P2-6: the pinned sustained protocol, engine-side (PLAN.md "Burst vs
/// sustained" + benchmarks/prompts/README operationalization): ≥ `minDuration`
/// of continuous generation, regenerating from the same prompt whenever a
/// generation ends (EOS or context-fill, whichever comes first — the caller's
/// generate closure encodes that by re-running the same prompt). The
/// per-generation tok/s sequence is the OV#9 bimodality signal, so results
/// keep every generation's metrics, never just a mean.
public enum SustainedLoopError: Error, Equatable, CustomStringConvertible {
    /// A generation produced no tokens while the duration had not elapsed —
    /// the loop would spin forever. Loud, never silent.
    case emptyGeneration(index: Int)

    public var description: String {
        switch self {
        case .emptyGeneration(let index):
            return "sustained loop: generation \(index) produced 0 tokens "
                + "before the duration elapsed — refusing to spin"
        }
    }
}

public struct SustainedLoopResult: Sendable {
    /// Per-generation metrics, in run order (the bimodality signal).
    public let generations: [GenerationMetrics]
    public let totalElapsedSeconds: Double
    /// True when the final generation was ended by the loop's duration bound
    /// (its rate spans a shorter window than the others — annotate, don't
    /// silently mix).
    public let lastGenerationTruncated: Bool

    public init(
        generations: [GenerationMetrics], totalElapsedSeconds: Double,
        lastGenerationTruncated: Bool
    ) {
        self.generations = generations
        self.totalElapsedSeconds = totalElapsedSeconds
        self.lastGenerationTruncated = lastGenerationTruncated
    }
}

public struct SustainedLoop {
    public let minDurationSeconds: Double

    public init(minDurationSeconds: Double) {
        precondition(minDurationSeconds > 0, "minDurationSeconds must be > 0")
        self.minDurationSeconds = minDurationSeconds
    }

    /// Runs generations back-to-back until the duration has elapsed. The
    /// in-flight generation when the clock crosses the bound is stopped at
    /// its next token boundary (via the `shouldStop` handed to `generate`)
    /// and kept as the final, possibly truncated, entry.
    /// - Parameters:
    ///   - clock: monotonic seconds; injectable for tests. Defaults to
    ///     `ProcessInfo.systemUptime`.
    ///   - generate: runs ONE generation from the pinned prompt, honoring
    ///     `shouldStop` at token boundaries, and returns its metrics.
    public func run(
        clock: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime },
        generate: (_ shouldStop: @escaping () -> Bool) throws -> GenerationMetrics
    ) throws -> SustainedLoopResult {
        let start = clock()
        let durationElapsed = { clock() - start >= minDurationSeconds }
        var generations: [GenerationMetrics] = []
        repeat {
            let metrics = try generate(durationElapsed)
            if metrics.generatedTokenCount == 0, !durationElapsed() {
                throw SustainedLoopError.emptyGeneration(index: generations.count)
            }
            generations.append(metrics)
        } while !durationElapsed()
        return SustainedLoopResult(
            generations: generations,
            totalElapsedSeconds: clock() - start,
            lastGenerationTruncated:
                generations.last?.stopReason == .stopRequested)
    }
}
