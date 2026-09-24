import Foundation

/// P6-1 (phase-6.md D1/D4/D5, edge tests 2–3): the battery fields and the
/// per-generation timeline export that ride every Phase 6 row.

/// Battery state of charge around a run — SEPARATE from battery health
/// (the binding 2026-09-05 obligation: every earlier "health" field was
/// SoC). The operator-read Settings → Battery value at the band marks is
/// the value of record; the programmatic `UIDevice.batteryLevel` reading
/// is a cross-check only. The engine never reads a battery itself — the
/// app supplies both, so this stays platform-neutral and testable.
public struct BatteryStateOfCharge: Sendable, Equatable {
    /// Settings → Battery reading at the start mark (e.g. "80"); empty
    /// renders the record-me placeholder.
    public var operatorStartNote: String
    /// Settings → Battery reading at the stop mark; typed AFTER the stop.
    public var operatorEndNote: String
    /// Programmatic reading in percent at start / end; nil when the
    /// platform reports none (Mac, monitoring unavailable).
    public var programmaticStartPercent: Double?
    public var programmaticEndPercent: Double?

    public init(
        operatorStartNote: String, operatorEndNote: String,
        programmaticStartPercent: Double?, programmaticEndPercent: Double?
    ) {
        self.operatorStartNote = operatorStartNote
        self.operatorEndNote = operatorEndNote
        self.programmaticStartPercent = programmaticStartPercent
        self.programmaticEndPercent = programmaticEndPercent
    }
}

/// One generation of a timeline — the JSON form of the text export's
/// `gen N:` line. Both render from the same values (`textLine` IS the text
/// export's line), so the two surfaces can never disagree.
public struct TimelineEntry: Codable, Equatable, Sendable {
    public let index: Int
    public let generatedTokens: Int
    /// Whole-generation wall time (prefill + decode), seconds.
    public let wallSeconds: Double
    /// Elapsed seconds at this generation's end (basis: the export's
    /// `elapsedBasis`).
    public let elapsedAtEndSeconds: Double
    public let overallTokensPerSecond: Double?
    /// The D1 decode rate of record (tokens 128–512); nil below 512 tokens.
    public let canonicalWindowTokensPerSecond: Double?
    /// `GenerationStopReason` raw value.
    public let stopReason: String
    /// True only for a final generation cut at a token boundary by the
    /// loop's stop (duration bound or operator stop).
    public let truncated: Bool

    public init(
        index: Int, generatedTokens: Int, wallSeconds: Double,
        elapsedAtEndSeconds: Double, overallTokensPerSecond: Double?,
        canonicalWindowTokensPerSecond: Double?, stopReason: String,
        truncated: Bool
    ) {
        self.index = index
        self.generatedTokens = generatedTokens
        self.wallSeconds = wallSeconds
        self.elapsedAtEndSeconds = elapsedAtEndSeconds
        self.overallTokensPerSecond = overallTokensPerSecond
        self.canonicalWindowTokensPerSecond = canonicalWindowTokensPerSecond
        self.stopReason = stopReason
        self.truncated = truncated
    }

    /// The text export's per-generation line (P2-6 vocabulary, unchanged).
    public var textLine: String {
        let overall = overallTokensPerSecond.map {
            String(format: "%.2f tok/s", $0)
        } ?? "n/a"
        let windowed = canonicalWindowTokensPerSecond.map {
            String(format: ", window %.2f tok/s", $0)
        } ?? ""
        return String(
            format: "  gen %d: overall %@%@ — %d tokens in %.1f s (stop: %@)",
            index, overall, windowed, generatedTokens, wallSeconds, stopReason)
    }
}

/// The JSON timeline export of one row (spec D8: archived next to the text
/// export under benchmarks/phase6/; tools/phase6_analyze.py consumes it).
public struct TimelineExport: Codable, Equatable, Sendable {
    public static let currentSchema = "qwen-metal-timeline/1"

    public struct EngineFields: Codable, Equatable, Sendable {
        public let weightsFormat: String
        public let residency: String
        public let kernelPath: String
        public let prefillPath: String
        public let prefillChunkSize: Int?
        public let prefillAttention: String?
    }

    public struct BatteryFields: Codable, Equatable, Sendable {
        /// Operator-typed Battery Health maximum capacity %; nil when empty.
        public let healthNote: String?
        public let operatorSoCStart: String?
        public let operatorSoCEnd: String?
        public let programmaticSoCStartPercent: Double?
        public let programmaticSoCEndPercent: Double?
    }

    public struct CycleTotals: Codable, Equatable, Sendable {
        public let generations: Int
        public let totalGeneratedTokens: Int
        public let sumGenerationWallSeconds: Double
        public let cycleWallSeconds: Double
        /// `EnergyLoopEnd` raw value ("durationBound" for sustained loops).
        public let endedBy: String
        public let lastGenerationTruncated: Bool
    }

    public let schema: String
    public let phase: String
    /// The round marker; nil ⇒ PROVISIONAL (spec D5).
    public let round: String?
    public let provisional: Bool
    public let date: String
    public let device: String
    public let osVersion: String
    public let engine: EngineFields
    public let promptName: String
    public let promptTokenCount: Int
    /// `BenchmarkReport.Mode` raw value.
    public let mode: String
    public let coldOrWarm: String?
    public let battery: BatteryFields?
    /// Loop totals (sustained / energy); nil for a burst row.
    public let cycle: CycleTotals?
    /// "loopClock" when the loop recorded real end offsets;
    /// "sumOfGenerationWall" when elapsed is the running Σ of wall times.
    public let elapsedBasis: String
    public let generations: [TimelineEntry]
    public let physFootprintBytes: UInt64?
}

extension BenchmarkReport {
    /// The report's timeline, or nil when no generation was recorded.
    public func timelineExport() -> TimelineExport? {
        let generations: [GenerationMetrics]
        let offsets: [Double]?
        var cycle: TimelineExport.CycleTotals?
        let lastTruncated: Bool
        switch mode {
        case .burst:
            guard let burst else { return nil }
            generations = [burst]
            offsets = nil
            lastTruncated = false
        case .sustained:
            guard let sustained else { return nil }
            generations = sustained.generations
            offsets = sustained.generationEndOffsetsSeconds
            lastTruncated = sustained.lastGenerationTruncated
            cycle = TimelineExport.CycleTotals(
                generations: generations.count,
                totalGeneratedTokens:
                    generations.reduce(0) { $0 + $1.generatedTokenCount },
                sumGenerationWallSeconds:
                    generations.reduce(0) { $0 + $1.wallSeconds },
                cycleWallSeconds: sustained.totalElapsedSeconds,
                endedBy: EnergyLoopEnd.durationBound.rawValue,
                lastGenerationTruncated: lastTruncated)
        case .energy:
            guard let energy else { return nil }
            generations = energy.generations
            offsets = energy.generationEndOffsetsSeconds
            lastTruncated = energy.lastGenerationTruncated
            cycle = TimelineExport.CycleTotals(
                generations: generations.count,
                totalGeneratedTokens: energy.totalGeneratedTokens,
                sumGenerationWallSeconds: energy.totalGenerationWallSeconds,
                cycleWallSeconds: energy.cycleWallSeconds,
                endedBy: energy.endedBy.rawValue,
                lastGenerationTruncated: lastTruncated)
        }
        guard !generations.isEmpty else { return nil }
        let battery = batteryStateOfCharge.map { soc in
            TimelineExport.BatteryFields(
                healthNote: Self.nonEmpty(batteryHealthNote),
                operatorSoCStart: Self.nonEmpty(soc.operatorStartNote),
                operatorSoCEnd: Self.nonEmpty(soc.operatorEndNote),
                programmaticSoCStartPercent: soc.programmaticStartPercent,
                programmaticSoCEndPercent: soc.programmaticEndPercent)
        }
        return TimelineExport(
            schema: TimelineExport.currentSchema,
            phase: phaseLabel,
            round: effectiveRound,
            provisional: isProvisional,
            date: dateStamp, device: deviceLabel, osVersion: osVersion,
            engine: TimelineExport.EngineFields(
                weightsFormat: weightsFormat.rawValue,
                residency: residency.rawValue,
                kernelPath: kernelPath.rawValue,
                prefillPath: prefillPath.rawValue,
                prefillChunkSize: prefillPath == .tiled ? prefillChunkSize : nil,
                prefillAttention: prefillPath == .tiled
                    ? prefillAttention?.rawValue : nil),
            promptName: promptName, promptTokenCount: promptTokenCount,
            mode: mode.rawValue,
            coldOrWarm: Self.nonEmpty(coldOrWarmNote),
            battery: battery,
            cycle: cycle,
            elapsedBasis: offsets == nil ? "sumOfGenerationWall" : "loopClock",
            generations: Self.timelineEntries(
                generations, offsets: offsets, lastTruncated: lastTruncated),
            physFootprintBytes: physFootprintBytes)
    }

    /// The timeline as deterministic JSON (sorted keys, pretty-printed), or
    /// nil when no generation was recorded. Throws only on an encoder
    /// failure, which the Codable shapes above cannot produce.
    public func timelineJSON() throws -> String? {
        guard let export = timelineExport() else { return nil }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        return String(decoding: try encoder.encode(export), as: UTF8.self)
    }

    /// The single source of the per-generation lines — text and JSON.
    static func timelineEntries(
        _ generations: [GenerationMetrics], offsets: [Double]?,
        lastTruncated: Bool
    ) -> [TimelineEntry] {
        var elapsed = 0.0
        return generations.enumerated().map { index, m in
            elapsed += m.wallSeconds
            return TimelineEntry(
                index: index,
                generatedTokens: m.generatedTokenCount,
                wallSeconds: m.wallSeconds,
                elapsedAtEndSeconds: offsets?[index] ?? elapsed,
                overallTokensPerSecond: m.overallTokensPerSecond,
                canonicalWindowTokensPerSecond: m.canonicalWindowTokensPerSecond,
                stopReason: m.stopReason.rawValue,
                truncated: lastTruncated && index == generations.count - 1)
        }
    }

    static func nonEmpty(_ note: String) -> String? {
        let trimmed = note.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}
