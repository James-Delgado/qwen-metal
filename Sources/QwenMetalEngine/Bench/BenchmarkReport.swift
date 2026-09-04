import Foundation

/// P2-6: benchmark row-field export (spec D8 "displays and exports the row
/// fields"). Assembles the fields the phase0-runbook row format and the
/// Phase 2 "before" row need, as text James can share/paste next to the
/// results.md table. Engine-side so the formatting is testable and the app
/// stays presentation-only. Operator-supplied context (battery health,
/// cold/warm, validation setting) is passed in — the app never guesses it.
public struct BenchmarkReport: Sendable {
    public enum Mode: String, Sendable {
        case burst
        case sustained
    }

    /// ISO date (YYYY-MM-DD), supplied by the caller — deterministic tests.
    public var dateStamp: String
    public var deviceLabel: String
    public var osVersion: String
    /// Operator-entered; empty renders a record-me placeholder.
    public var batteryHealthNote: String
    /// Operator-entered cold/warm annotation; empty renders a placeholder.
    public var coldOrWarmNote: String
    public var residency: WeightsResidency
    /// Which weight encoding produced the row (P3-5): Phase 2 bf16 or the
    /// Phase 3 q4g64 packed artifact — rows must record it.
    public var weightsFormat: WeightsFormat
    public var promptName: String
    public var promptTokenCount: Int
    public var mode: Mode
    /// The burst generation (mode .burst).
    public var burst: GenerationMetrics?
    /// The sustained loop result (mode .sustained).
    public var sustained: SustainedLoopResult?
    /// In-app phys_footprint reading. Cross-check only — the Xcode memory
    /// gauge stays the metric of record (PLAN.md protocol pin).
    public var physFootprintBytes: UInt64?

    public init(
        dateStamp: String, deviceLabel: String, osVersion: String,
        batteryHealthNote: String, coldOrWarmNote: String,
        residency: WeightsResidency, weightsFormat: WeightsFormat = .bf16,
        promptName: String,
        promptTokenCount: Int, mode: Mode,
        burst: GenerationMetrics? = nil,
        sustained: SustainedLoopResult? = nil,
        physFootprintBytes: UInt64? = nil
    ) {
        self.dateStamp = dateStamp
        self.deviceLabel = deviceLabel
        self.osVersion = osVersion
        self.batteryHealthNote = batteryHealthNote
        self.coldOrWarmNote = coldOrWarmNote
        self.residency = residency
        self.weightsFormat = weightsFormat
        self.promptName = promptName
        self.promptTokenCount = promptTokenCount
        self.mode = mode
        self.burst = burst
        self.sustained = sustained
        self.physFootprintBytes = physFootprintBytes
    }

    public func exportText() -> String {
        var lines: [String] = []
        let phase = weightsFormat == .bf16 ? "Phase 2" : "Phase 3"
        lines.append("qwen-metal \(phase) row export (PROVISIONAL)")
        lines.append("date: \(dateStamp)")
        lines.append("device: \(deviceLabel) (iOS \(osVersion))")
        lines.append("battery health: \(orPlaceholder(batteryHealthNote))")
        let engineDescription = weightsFormat == .bf16
            ? "naive fp16 GPU"
            : "q4g64 fused-dequant GPU"
        lines.append(
            "engine: qwen-metal \(engineDescription) — weights "
                + "\(weightsFormat.rawValue), residency \(residency.rawValue)")
        lines.append("prompt: \(promptName) (\(promptTokenCount) prompt tokens)")
        lines.append("mode: \(mode.rawValue) | cold/warm: \(orPlaceholder(coldOrWarmNote))")
        lines.append(
            "sampling: greedy | Metal API validation: confirm OFF and record "
                + "(P0A-1 addendum)")

        switch mode {
        case .burst:
            if let burst {
                lines.append(contentsOf: generationLines(burst))
            } else {
                lines.append("burst: no generation recorded")
            }
        case .sustained:
            if let sustained {
                lines.append(contentsOf: sustainedLines(sustained))
            } else {
                lines.append("sustained: no loop recorded")
            }
        }

        if let physFootprintBytes {
            lines.append(String(
                format: "phys_footprint (in-app cross-check; Xcode gauge is "
                    + "the metric of record): %.1f MB",
                Double(physFootprintBytes) / 1_048_576))
        }
        return lines.joined(separator: "\n")
    }

    private func orPlaceholder(_ note: String) -> String {
        note.isEmpty ? "(record manually)" : note
    }

    /// The CLI's P2-5 per-token block vocabulary, one field per line.
    private func generationLines(_ m: GenerationMetrics) -> [String] {
        var lines: [String] = []
        lines.append(String(
            format: "generated: %d tokens in %.1f s (stop: %@)",
            m.generatedTokenCount, m.wallSeconds, m.stopReason.rawValue))
        if let prefill = m.prefillSeconds {
            lines.append(String(
                format: "prefill: %d tokens in %.2f s (%.2f tok/s — "
                    + "sequential per spec D6; includes the first generated "
                    + "token's forward)",
                m.promptTokenCount, prefill,
                Double(m.promptTokenCount) / max(prefill, 1e-9)))
        }
        if let t = m.timing {
            let dispatches = t.minDispatchCount == t.maxDispatchCount
                ? "\(t.minDispatchCount)"
                : "UNSTABLE \(t.minDispatchCount)-\(t.maxDispatchCount)"
            lines.append(String(
                format: "per-token (%d tokens): median GPU %.2f ms, median "
                    + "wall %.2f ms, median wall-GPU %.3f ms, %@ dispatches/token",
                t.tokenCount, t.medianGPUSeconds * 1000,
                t.medianWallSeconds * 1000, t.medianOverheadSeconds * 1000,
                dispatches))
        }
        let overall = m.overallTokensPerSecond.map {
            String(format: "%.2f tok/s", $0)
        } ?? "n/a"
        let windowed = m.canonicalWindowTokensPerSecond.map {
            String(format: "%.2f tok/s", $0)
        } ?? String(
            format: "n/a (needs >= %d generated tokens, got %d)",
            CanonicalDecodeWindow.lastToken, m.generatedTokenCount)
        lines.append(
            "decode rate: overall \(overall), canonical window (tokens "
                + "\(CanonicalDecodeWindow.firstToken)-"
                + "\(CanonicalDecodeWindow.lastToken)) \(windowed)")
        return lines
    }

    private func sustainedLines(_ result: SustainedLoopResult) -> [String] {
        var lines: [String] = []
        lines.append(String(
            format: "sustained loop: %d generations over %.1f min%@",
            result.generations.count, result.totalElapsedSeconds / 60,
            result.lastGenerationTruncated
                ? " (final generation truncated by the duration bound)" : ""))
        // Per-generation sequence — the OV#9 bimodality signal.
        for (index, m) in result.generations.enumerated() {
            let overall = m.overallTokensPerSecond.map {
                String(format: "%.2f tok/s", $0)
            } ?? "n/a"
            let windowed = m.canonicalWindowTokensPerSecond.map {
                String(format: ", window %.2f tok/s", $0)
            } ?? ""
            lines.append(String(
                format: "  gen %d: overall %@%@ — %d tokens in %.1f s (stop: %@)",
                index, overall, windowed, m.generatedTokenCount,
                m.wallSeconds, m.stopReason.rawValue))
        }
        if let last = result.generations.last, let t = last.timing {
            let dispatches = t.minDispatchCount == t.maxDispatchCount
                ? "\(t.minDispatchCount)"
                : "UNSTABLE \(t.minDispatchCount)-\(t.maxDispatchCount)"
            lines.append(String(
                format: "last generation per-token: median GPU %.2f ms, median "
                    + "wall %.2f ms, median wall-GPU %.3f ms, %@ dispatches/token",
                t.medianGPUSeconds * 1000, t.medianWallSeconds * 1000,
                t.medianOverheadSeconds * 1000, dispatches))
        }
        return lines
    }
}
