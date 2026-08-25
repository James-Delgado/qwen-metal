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
