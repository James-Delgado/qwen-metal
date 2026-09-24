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
        /// P6-1 (phase-6.md D4): the operator-bounded energy cycle.
        case energy
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
    /// The energy cycle result (mode .energy; P6-1).
    public var energy: EnergyLoopResult?
    /// In-app phys_footprint reading. Cross-check only — the Xcode memory
    /// gauge stays the metric of record (PLAN.md protocol pin).
    public var physFootprintBytes: UInt64?
    /// P6-1: state of charge around the run, SEPARATE from
    /// `batteryHealthNote` (the 2026-09-05 obligation). nil omits the lines.
    public var batteryStateOfCharge: BatteryStateOfCharge?
    /// P6-1 (spec D5): the Phase 6 round marker. Set ONLY on rows produced
    /// inside the Phase 6 round — it lifts the PROVISIONAL marker on q4g64
    /// rows. nil / blank ⇒ PROVISIONAL. Operator-typed; the app never
    /// guesses it.
    public var round: String?

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
        energy: EnergyLoopResult? = nil,
        physFootprintBytes: UInt64? = nil,
        batteryStateOfCharge: BatteryStateOfCharge? = nil,
        round: String? = nil
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
        self.energy = energy
        self.physFootprintBytes = physFootprintBytes
        self.batteryStateOfCharge = batteryStateOfCharge
        self.round = round
    }

    /// The round marker, trimmed; nil when unset or blank.
    public var effectiveRound: String? { Self.nonEmpty(round ?? "") }

    /// bf16 rows are the permanent Phase 2 correctness artifact; q4g64 rows
    /// are Phase 6 rows now (the P5-EXEC → SPEC-P6 handover, spec D5).
    var phaseLabel: String { weightsFormat == .bf16 ? "Phase 2" : "Phase 6" }

    /// PROVISIONAL unless the row was produced inside the Phase 6 round
    /// (spec D5). bf16 rows never leave PROVISIONAL — they are not
    /// head-to-head rows.
    public var isProvisional: Bool {
        weightsFormat == .bf16 || effectiveRound == nil
    }

    public func exportText() -> String {
        var lines: [String] = []
        // bf16 rows are the Phase 2 correctness artifact; q4g64 rows are
        // Phase 6 rows now (both prefill paths and both kernel arms —
        // distinguished by the engine fields below). P6-1 (spec D5): the
        // PROVISIONAL marker drops ONLY inside the Phase 6 round.
        let marker = isProvisional
            ? "PROVISIONAL" : "round: \(effectiveRound ?? "")"
        lines.append("qwen-metal \(phaseLabel) row export (\(marker))")
        lines.append("date: \(dateStamp)")
        lines.append("device: \(deviceLabel) (iOS \(osVersion))")
        lines.append(roundLine)
        lines.append(contentsOf: batteryLines)
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
        case .energy:
            if let energy {
                lines.append(contentsOf: energyLines(energy))
            } else {
                lines.append("energy: no cycle recorded")
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

    /// Operator-typed percent notes: "88", "88%", " 88 % " all render as
    /// "88%"; empty renders the placeholder (never a number).
    private func percentOrPlaceholder(_ note: String) -> String {
        guard var trimmed = Self.nonEmpty(note) else { return "(record manually)" }
        while trimmed.hasSuffix("%") || trimmed.hasSuffix(" ") {
            trimmed.removeLast()
        }
        return trimmed.isEmpty ? "(record manually)" : trimmed + "%"
    }

    private var roundLine: String {
        guard let round = effectiveRound else {
            return "round: (none — PROVISIONAL)"
        }
        return weightsFormat == .bf16
            ? "round: \(round) (bf16 rows stay PROVISIONAL — Phase 2 "
                + "correctness artifact, never a head-to-head row)"
            : "round: \(round)"
    }

    /// P6-1 (edge test 2): health and state of charge are SEPARATE fields.
    /// Health is the operator-typed Battery Health maximum capacity % (no
    /// public API); SoC lines render only when the app supplied readings
    /// (the energy mode), with the programmatic value labeled a cross-check.
    private var batteryLines: [String] {
        var lines = [
            "battery health: \(percentOrPlaceholder(batteryHealthNote)) "
                + "(maximum capacity %, Settings → Battery → Battery Health, "
                + "operator-typed — NOT state of charge)"
        ]
        if let soc = batteryStateOfCharge {
            lines.append(
                "state of charge (Settings → Battery, operator-read at the "
                    + "band marks — value of record): start "
                    + "\(percentOrPlaceholder(soc.operatorStartNote)), end "
                    + "\(percentOrPlaceholder(soc.operatorEndNote))")
            func programmatic(_ value: Double?) -> String {
                value.map { String(format: "%.1f%%", $0) } ?? "n/a (unavailable)"
            }
            lines.append(
                "state of charge (programmatic UIDevice.batteryLevel — "
                    + "cross-check only, never the value of record): start "
                    + "\(programmatic(soc.programmaticStartPercent)), end "
                    + "\(programmatic(soc.programmaticEndPercent))")
        }
        return lines
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
        // Per-generation sequence — the OV#9 bimodality signal. P6-1: the
        // lines render from the timeline entries, the JSON export's source.
        lines.append(contentsOf: Self.timelineEntries(
            result.generations, offsets: result.generationEndOffsetsSeconds,
            lastTruncated: result.lastGenerationTruncated).map(\.textLine))
        lines.append(contentsOf: lastGenerationLines(result.generations.last))
        return lines
    }

    /// P6-1 (phase-6.md D4): the energy cycle's raw fields. The energy math
    /// (J per 1% from health, idle scaling, net J/token, implied watts)
    /// lives in tools/phase6_analyze.py (P6-3) — no number is derived here.
    private func energyLines(_ result: EnergyLoopResult) -> [String] {
        var lines: [String] = []
        let ended: String
        switch result.endedBy {
        case .operatorStop: ended = "operator stop"
        case .durationBound:
            ended = "duration bound (safety cap — NOT the protocol's operator stop)"
        }
        lines.append(String(
            format: "energy cycle: %d generations — ended by %@%@",
            result.generations.count, ended,
            result.lastGenerationTruncated
                ? " (final generation truncated at a token boundary)" : ""))
        lines.append(String(
            format: "  cumulative: %d generated tokens; Σ generation wall %.1f s; "
                + "cycle wall %.1f s (idle-baseline pro-rata basis)",
            result.totalGeneratedTokens, result.totalGenerationWallSeconds,
            result.cycleWallSeconds))
        lines.append(contentsOf: Self.timelineEntries(
            result.generations, offsets: result.generationEndOffsetsSeconds,
            lastTruncated: result.lastGenerationTruncated).map(\.textLine))
        lines.append(contentsOf: lastGenerationLines(result.generations.last))
        lines.append(
            "energy J/token: not computed in-app — tools/phase6_analyze.py "
                + "(P6-3) derives it from the health/SoC fields and the token "
                + "total above")
        return lines
    }

    /// The last (steady-state) generation's per-token block, shared by the
    /// sustained and energy exports.
    private func lastGenerationLines(_ last: GenerationMetrics?) -> [String] {
        var lines: [String] = []
        if let last, let t = last.timing {
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
        if let variance = last?.latencyVariance {
            lines.append("last generation " + variance.summaryLine)
        }
        return lines
    }
}
