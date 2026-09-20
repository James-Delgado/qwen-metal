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
    /// Which kernel structure produced the row (P4-4, spec D4): the P4-5
    /// before/after rows are naive vs fused, so every row must record it.
    /// No default on purpose — the compiler forces call sites to label.
    public var kernelPath: GPUModel.KernelPath
    /// Which prefill structure produced the row (P5-4, phase-5.md D5): the
    /// P5-5 before/after rows are sequential vs tiled, so every row must
    /// record it. No default on purpose, like `kernelPath`.
    public var prefillPath: GPUModel.PrefillPath
    /// The tiled chunk size C (spec D2: recorded on every tiled row).
    /// Rendered only when `prefillPath == .tiled`.
    public var prefillChunkSize: Int?
    /// The tiled chunk's causal SDPA kernel (PF-2: recorded on every tiled
    /// row — the on-device A/B rows differ only in this field). Rendered
    /// only when `prefillPath == .tiled`.
    public var prefillAttention: GPUModel.PrefillAttention?
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
        kernelPath: GPUModel.KernelPath,
        prefillPath: GPUModel.PrefillPath,
        prefillChunkSize: Int? = nil,
        prefillAttention: GPUModel.PrefillAttention? = nil,
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
        self.kernelPath = kernelPath
        self.prefillPath = prefillPath
        self.prefillChunkSize = prefillChunkSize
        self.prefillAttention = prefillAttention
        self.promptName = promptName
        self.promptTokenCount = promptTokenCount
        self.mode = mode
        self.burst = burst
        self.sustained = sustained
        self.physFootprintBytes = physFootprintBytes
    }

    public func exportText() -> String {
        var lines: [String] = []
        // bf16 rows are the Phase 2 correctness artifact; q4g64 rows are
        // Phase 5 rows now (both prefill paths — the P5-5 sequential arm is
        // a Phase 5 A/B row, distinguished by the prefill field below; the
        // P4-era kernels field stays for the naive arm).
        let phase = weightsFormat == .bf16 ? "Phase 2" : "Phase 5"
        lines.append("qwen-metal \(phase) row export (PROVISIONAL)")
        lines.append("date: \(dateStamp)")
        lines.append("device: \(deviceLabel) (iOS \(osVersion))")
        lines.append("battery health: \(orPlaceholder(batteryHealthNote))")
        let engineDescription = weightsFormat == .bf16
            ? "naive fp16 GPU"
            : "q4g64 fused-dequant GPU"
        let prefillLabel: String
        if prefillPath == .tiled, let chunk = prefillChunkSize {
            prefillLabel = "prefill tiled (C=\(chunk))"
                + (prefillAttention.map { ", attention \($0.rawValue)" } ?? "")
        } else {
            prefillLabel = "prefill \(prefillPath.rawValue)"
        }
        lines.append(
            "engine: qwen-metal \(engineDescription) — weights "
                + "\(weightsFormat.rawValue), residency \(residency.rawValue), "
                + "kernels \(kernelPath.rawValue), \(prefillLabel)")
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
        // P5-1 (phase-5.md D1): the prefill span is the metric of record;
        // the P2-6 TTFT-style field keeps exporting underneath, honestly
        // labeled — rows cite the span.
        if let span = m.prefillSpan {
            lines.append(span.summaryLine)
        }
        if let prefill = m.prefillSeconds {
            lines.append(String(
                format: "ttft-style (legacy P2-6 field, runner-clocked): "
                    + "%d prompt tokens, %.2f s to first token available",
                m.promptTokenCount, prefill))
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
        // P4-1 (spec D7): the latency-variance line on every Phase 4 row.
        if let variance = m.latencyVariance {
            lines.append(variance.summaryLine)
        }
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
        // P4-1 (spec D7): variance for the last (steady-state) generation.
        if let variance = result.generations.last?.latencyVariance {
            lines.append("last generation " + variance.summaryLine)
        }
        return lines
    }
}
