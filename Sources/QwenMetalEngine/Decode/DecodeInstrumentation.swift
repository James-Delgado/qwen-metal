import Foundation

/// P2-5 (docs/phases/phase-2.md D5 + instrumentation deliverables): per-token
/// decode instrumentation, aggregated engine-side so the CLI and the Phase 2
/// iOS app consume one implementation.
///
/// One record per generated token: the dual timing of the command buffer that
/// produced it (hard rule 7) plus the number of compute dispatches it encoded.
/// Record `i` (0-based) belongs to generated token `i + 1` (1-based) — the
/// forward pass whose logits produced that token.
public struct TokenStepRecord: Sendable {
    public let timing: DispatchTiming
    public let dispatchCount: Int

    public init(timing: DispatchTiming, dispatchCount: Int) {
        self.timing = timing
        self.dispatchCount = dispatchCount
    }
}

/// The canonical decode measurement window — a benchmark protocol pin
/// (PLAN.md: "decode tok/s = generated tokens ÷ decode wall time, canonical
/// window = tokens 128–512"; METHODOLOGY rule 5). The windowed rate spans
/// completion of generated token 128 to completion of generated token 512
/// (1-based), i.e. 384 tokens of steady-state decode — so KV-depth-dependent
/// bytes/token and prefill/warmup effects don't skew comparisons.
public enum CanonicalDecodeWindow {
    public static let firstToken = 128
    public static let lastToken = 512
    public static let tokenSpan = lastToken - firstToken  // 384
}

/// Aggregates of one generation's per-token records. All times are seconds.
public struct DecodeTimingSummary: Sendable {
    public let tokenCount: Int
    /// Median per-token GPU execution time (command-buffer GPU timestamps).
    public let medianGPUSeconds: Double
    /// Median per-token wall time of the command buffer (encode → completed).
    public let medianWallSeconds: Double
    /// Median of the PER-TOKEN wall − GPU deltas — the dispatch-overhead
    /// metric Phase 4 consumes (hard rule 7). Not medianWall − medianGPU:
    /// the medians may come from different tokens.
    public let medianOverheadSeconds: Double
    /// Dispatches/token across the records. Equal min/max is the expected
    /// steady state (every decode step encodes the same pipeline).
    public let minDispatchCount: Int
    public let maxDispatchCount: Int
}

/// Which slice of a generation a `LatencyVarianceStats` describes (P4-1,
/// phase-4.md D7). Canonical-window stats are the row-reportable form; the
/// all-tokens form exists for short runs and is always labeled as such.
public enum LatencyScope: String, Sendable {
    case canonicalWindow
    case allTokens

    /// Human-readable scope label for CLI/report lines.
    public var label: String {
        switch self {
        case .canonicalWindow:
            return "window tokens \(CanonicalDecodeWindow.firstToken)-"
                + "\(CanonicalDecodeWindow.lastToken)"
        case .allTokens:
            return "all tokens"
        }
    }
}

/// P4-1 (phase-4.md D7): per-token wall-clock latency distribution — reported
/// on every Phase 4 row, never gated. Conventions pinned in the DECISIONS.md
/// 2026-09-08 P4-1 sanity-bounds entry:
/// - a span is the completion-to-completion wall delta between consecutive
///   `TokenStepRecord.timing.wallEnd`s (the canonical-window rate's
///   semantics — host work between command buffers is included);
/// - percentiles are nearest-rank on the sorted spans (an observed value,
///   index ⌈q·n⌉−1), so p50 here may differ minutely from the
///   mean-of-middle-two `DecodeTimingSummary` median on even counts;
/// - a stall is a span strictly greater than 2 × the same distribution's
///   p50 — the page-fault/preemption signature detector.
public struct LatencyVarianceStats: Sendable {
    public let scope: LatencyScope
    public let spanCount: Int
    public let p50Seconds: Double
    public let p95Seconds: Double
    public let p99Seconds: Double
    public let maxSeconds: Double
    public let stallCount: Int

    /// nil when `interTokenSeconds` is empty or contains a non-positive span
    /// (records not from one monotonic generation — fail loudly, never
    /// report garbage statistics).
    public static func compute(
        interTokenSeconds: [Double], scope: LatencyScope
    ) -> LatencyVarianceStats? {
        guard !interTokenSeconds.isEmpty,
              interTokenSeconds.allSatisfy({ $0 > 0 }) else { return nil }
        let sorted = interTokenSeconds.sorted()
        let n = sorted.count
        func nearestRank(_ q: Double) -> Double {
            sorted[max(0, Int((q * Double(n)).rounded(.up)) - 1)]
        }
        let p50 = nearestRank(0.50)
        return LatencyVarianceStats(
            scope: scope,
            spanCount: n,
            p50Seconds: p50,
            p95Seconds: nearestRank(0.95),
            p99Seconds: nearestRank(0.99),
            maxSeconds: sorted[n - 1],
            stallCount: sorted.count(where: { $0 > 2 * p50 }))
    }

    /// The one-line report form the CLI, BenchmarkReport, and the app share.
    public var summaryLine: String {
        String(
            format: "latency (%@): p50 %.2f ms, p95 %.2f ms, p99 %.2f ms, "
                + "max %.2f ms, stalls %d (spans > 2x p50, n = %d)",
            scope.label, p50Seconds * 1000, p95Seconds * 1000,
            p99Seconds * 1000, maxSeconds * 1000, stallCount, spanCount)
    }
}

/// Collects `TokenStepRecord`s during a generation (the CLI/app hook them in
/// via `DecodeLoop`'s `onStep`) and computes the P2-5 aggregates.
public struct DecodeTimingCollector: Sendable {
    public private(set) var records: [TokenStepRecord] = []

    public init() {}

    public mutating func append(_ record: TokenStepRecord) {
        records.append(record)
    }

    /// nil when no records were collected.
    public func summary() -> DecodeTimingSummary? {
        guard !records.isEmpty else { return nil }
        let counts = records.map(\.dispatchCount)
        return DecodeTimingSummary(
            tokenCount: records.count,
            medianGPUSeconds: Self.median(records.map(\.timing.gpuDuration)),
            medianWallSeconds: Self.median(records.map(\.timing.wallDuration)),
            medianOverheadSeconds: Self.median(records.map(\.timing.dispatchOverhead)),
            minDispatchCount: counts.min()!,
            maxDispatchCount: counts.max()!)
    }

    /// Decode tok/s over the canonical window (`CanonicalDecodeWindow`):
    /// 384 tokens ÷ (wallEnd of token 512's forward − wallEnd of token 128's
    /// forward). Spanning completion-to-completion includes the host work
    /// between command buffers (argmax, loop overhead) — the honest cadence.
    /// nil when fewer than 512 tokens were generated, or when the span is not
    /// positive (records not from one monotonic generation).
    public func canonicalWindowTokensPerSecond() -> Double? {
        guard records.count >= CanonicalDecodeWindow.lastToken else { return nil }
        let elapsed = records[CanonicalDecodeWindow.lastToken - 1].timing.wallEnd
            - records[CanonicalDecodeWindow.firstToken - 1].timing.wallEnd
        guard elapsed > 0 else { return nil }
        return Double(CanonicalDecodeWindow.tokenSpan) / elapsed
    }

    /// Overall decode tok/s across all records: (n − 1) tokens ÷ (last
    /// wallEnd − first wallEnd). The first token's own forward is the span's
    /// start marker, not a counted token — its duration belongs to the
    /// prefill-to-decode transition, and the span between completions is
    /// what includes per-token host overhead. nil below 2 records or on a
    /// non-positive span.
    public func overallTokensPerSecond() -> Double? {
        guard records.count >= 2,
              let first = records.first, let last = records.last else { return nil }
        let elapsed = last.timing.wallEnd - first.timing.wallEnd
        guard elapsed > 0 else { return nil }
        return Double(records.count - 1) / elapsed
    }

    /// P4-1 (spec D7): latency-variance stats over the canonical window —
    /// the 384 completion-to-completion spans between generated tokens 128
    /// and 512. nil when fewer than 512 tokens were generated, or when the
    /// window's spans are not strictly positive (corrupt sequence).
    public func canonicalWindowLatencyVariance() -> LatencyVarianceStats? {
        guard records.count >= CanonicalDecodeWindow.lastToken else { return nil }
        let window = records[
            (CanonicalDecodeWindow.firstToken - 1)..<CanonicalDecodeWindow.lastToken]
        return LatencyVarianceStats.compute(
            interTokenSeconds: Self.interTokenSpans(Array(window)),
            scope: .canonicalWindow)
    }

    /// The all-tokens fallback for runs too short for the canonical window —
    /// always reported with its scope label, never passed off as the window
    /// form. nil below 2 records or on non-positive spans.
    public func allTokensLatencyVariance() -> LatencyVarianceStats? {
        LatencyVarianceStats.compute(
            interTokenSeconds: Self.interTokenSpans(records), scope: .allTokens)
    }

    /// Consecutive completion-to-completion wall deltas (n − 1 spans).
    private static func interTokenSpans(_ records: [TokenStepRecord]) -> [Double] {
        guard records.count >= 2 else { return [] }
        return (1..<records.count).map {
            records[$0].timing.wallEnd - records[$0 - 1].timing.wallEnd
        }
    }

    /// Median with the even-count convention: mean of the two middle values.
    private static func median(_ values: [Double]) -> Double {
        precondition(!values.isEmpty, "median of an empty array")
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count.isMultiple(of: 2)
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid]
    }
}
