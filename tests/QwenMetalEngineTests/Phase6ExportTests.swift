import XCTest
import Foundation
import QwenMetalEngine

/// P6-1 (phase-6.md edge tests 2–4): the Phase 6 row-export surfaces —
/// battery health and state of charge as SEPARATE fields (the 2026-09-05
/// capacity-basis obligation), the per-generation timeline JSON export,
/// and the `round` marker that gates the PROVISIONAL header (spec D5).
final class Phase6ExportTests: XCTestCase {

    /// Real per-token records through the collector so the rates are the
    /// engine's own numbers, not typed.
    private func generation(
        tokens: Int, spacing: Double = 0.25, stop: GenerationStopReason = .contextFull
    ) -> GenerationMetrics {
        var collector = DecodeTimingCollector()
        for i in 0..<tokens {
            let base = Double(i) * spacing
            collector.append(TokenStepRecord(
                timing: DispatchTiming(
                    wallStart: base, wallEnd: base + 0.2 * spacing,
                    gpuStart: base, gpuEnd: base + 0.15 * spacing),
                dispatchCount: 591))
        }
        return GenerationMetrics(
            promptTokenCount: 84, generatedTokenCount: tokens,
            wallSeconds: Double(tokens) * spacing + 0.7,
            prefillSeconds: 0.7, stopReason: stop,
            timing: collector.summary(),
            overallTokensPerSecond: collector.overallTokensPerSecond(),
            canonicalWindowTokensPerSecond:
                collector.canonicalWindowTokensPerSecond(),
            latencyVariance: collector.canonicalWindowLatencyVariance()
                ?? collector.allTokensLatencyVariance())
    }

    private func energyResult() -> EnergyLoopResult {
        let gens = [
            generation(tokens: 640, spacing: 0.25),
            generation(tokens: 640, spacing: 0.30),
            generation(tokens: 130, spacing: 0.30, stop: .stopRequested),
        ]
        var offsets: [Double] = []
        var elapsed = 0.0
        for g in gens { elapsed += g.wallSeconds; offsets.append(elapsed) }
        return EnergyLoopResult(
            generations: gens, generationEndOffsetsSeconds: offsets,
            cycleWallSeconds: elapsed, endedBy: .operatorStop,
            lastGenerationTruncated: true)
    }

    private func report(
        mode: BenchmarkReport.Mode = .energy,
        health: String = "100",
        soc: BatteryStateOfCharge? = BatteryStateOfCharge(
            operatorStartNote: "80", operatorEndNote: "70",
            programmaticStartPercent: 81, programmaticEndPercent: 70),
        round: String? = nil,
        weightsFormat: WeightsFormat = .q4g64,
        burst: GenerationMetrics? = nil,
        sustained: SustainedLoopResult? = nil,
        energy: EnergyLoopResult? = nil
    ) -> BenchmarkReport {
        BenchmarkReport(
            dateStamp: "2026-10-01", deviceLabel: "iPhone16,1",
            osVersion: "26.6.1", batteryHealthNote: health,
            coldOrWarmNote: "warm", residency: .mmap,
            weightsFormat: weightsFormat,
            kernelPath: .fused, prefillPath: .tiled, prefillChunkSize: 512,
            prefillAttention: .queryTiled,
            promptName: "decode-essay", promptTokenCount: 84, mode: mode,
            burst: burst, sustained: sustained, energy: energy,
            physFootprintBytes: 601_000_000,
            batteryStateOfCharge: soc, round: round)
    }

    // MARK: - edge test 2: battery health and SoC are separate fields

    func testHealthAndStateOfChargeAreSeparateExportFields() {
        let text = report(energy: energyResult()).exportText()
        XCTAssertTrue(text.contains("battery health: 100%"), text)
        XCTAssertTrue(
            text.contains("maximum capacity %"),
            "health is labeled as Battery Health max capacity, never SoC")
        XCTAssertTrue(
            text.contains("NOT state of charge"), text)
        XCTAssertTrue(text.contains(
            "state of charge (Settings → Battery, operator-read at the band "
            + "marks — value of record): start 80%, end 70%"), text)
        XCTAssertTrue(text.contains(
            "state of charge (programmatic UIDevice.batteryLevel — cross-check "
            + "only, never the value of record): start 81.0%, end 70.0%"), text)
    }

    func testEmptyOperatorHealthRendersPlaceholderNeverANumber() {
        // The programmatic SoC is present — it must never leak into the
        // health field (every pre-2026-09-05 "health" value WAS SoC).
        let text = report(health: "", energy: energyResult()).exportText()
        XCTAssertTrue(text.contains("battery health: (record manually)"), text)
        XCTAssertFalse(text.contains("battery health: 81"), text)
        XCTAssertFalse(text.contains("battery health: 80"), text)
    }

    func testEmptyOperatorSoCMarksRenderPlaceholdersAndMissingProgrammaticIsLabeled() {
        let soc = BatteryStateOfCharge(
            operatorStartNote: "", operatorEndNote: "",
            programmaticStartPercent: nil, programmaticEndPercent: nil)
        let text = report(soc: soc, energy: energyResult()).exportText()
        XCTAssertTrue(text.contains(
            "value of record): start (record manually), end (record manually)"), text)
        XCTAssertTrue(text.contains(
            "never the value of record): start n/a (unavailable), "
            + "end n/a (unavailable)"), text)
    }

    func testNoStateOfChargeStructOmitsTheSoCLines() {
        let text = report(
            mode: .burst, soc: nil, burst: generation(tokens: 64)).exportText()
        XCTAssertFalse(text.contains("state of charge (Settings"), text)
        XCTAssertFalse(text.contains("state of charge (programmatic"), text)
        XCTAssertTrue(text.contains("battery health: 100%"))
        XCTAssertTrue(text.contains("NOT state of charge"),
                      "the health label still disambiguates itself")
    }

    // MARK: - energy lines: raw fields only; J/token belongs to P6-3

    func testEnergyExportCarriesCycleTotalsAndDefersTheEnergyMath() {
        let result = energyResult()
        let text = report(energy: result).exportText()
        XCTAssertTrue(text.contains("mode: energy"), text)
        XCTAssertTrue(text.contains(
            "energy cycle: 3 generations — ended by operator stop "
            + "(final generation truncated at a token boundary)"), text)
        XCTAssertTrue(text.contains(String(
            format: "cumulative: %d generated tokens; Σ generation wall %.1f s; "
                + "cycle wall %.1f s",
            result.totalGeneratedTokens, result.totalGenerationWallSeconds,
            result.cycleWallSeconds)), text)
        XCTAssertTrue(text.contains(
            "energy J/token: not computed in-app — tools/phase6_analyze.py "
            + "(P6-3) derives it"), text)
        XCTAssertFalse(text.contains("J/token:") && text.contains(" J = "),
                       "no energy arithmetic in the export")
        XCTAssertTrue(text.contains("last generation per-token: median GPU"), text)
    }

    func testEnergyExportWithoutResultSaysSo() {
        let text = report(energy: nil).exportText()
        XCTAssertTrue(text.contains("energy: no cycle recorded"), text)
    }

    // MARK: - edge test 3: timeline JSON round-trips and matches the text

    func testTimelineJSONRoundTripsAndTextLinesCiteTheSameNumbers() throws {
        let r = report(round: "phase6-energy-2026-10-01", energy: energyResult())
        let export = try XCTUnwrap(r.timelineExport())
        let json = try XCTUnwrap(r.timelineJSON())
        let decoded = try JSONDecoder().decode(
            TimelineExport.self, from: Data(json.utf8))
        XCTAssertEqual(decoded, export, "JSON round-trip must be lossless")

        XCTAssertEqual(decoded.schema, TimelineExport.currentSchema)
        XCTAssertEqual(decoded.phase, "Phase 6")
        XCTAssertEqual(decoded.round, "phase6-energy-2026-10-01")
        XCTAssertFalse(decoded.provisional)
        XCTAssertEqual(decoded.mode, "energy")
        XCTAssertEqual(decoded.generations.count, 3)
        XCTAssertEqual(decoded.generations.map(\.index), [0, 1, 2])
        XCTAssertEqual(decoded.generations.map(\.stopReason),
                       ["contextFull", "contextFull", "stopRequested"])
        XCTAssertEqual(decoded.generations.last?.truncated, true)
        XCTAssertEqual(decoded.generations.first?.truncated, false)
        let cycle = try XCTUnwrap(decoded.cycle)
        XCTAssertEqual(cycle.generations, 3)
        XCTAssertEqual(cycle.totalGeneratedTokens, 640 + 640 + 130)
        XCTAssertEqual(cycle.endedBy, "operatorStop")
        let battery = try XCTUnwrap(decoded.battery)
        XCTAssertEqual(battery.healthNote, "100")
        XCTAssertEqual(battery.operatorSoCStart, "80")
        XCTAssertEqual(battery.operatorSoCEnd, "70")
        XCTAssertEqual(battery.programmaticSoCStartPercent, 81)
        XCTAssertEqual(battery.programmaticSoCEndPercent, 70)

        // The text export's per-generation lines cite the same numbers, in
        // order — rendered from the JSON-decoded entries with the text
        // formatter, every line must appear verbatim.
        let text = r.exportText()
        var lastRange: Range<String.Index>? = nil
        for entry in decoded.generations {
            let line = entry.textLine
            let range = try XCTUnwrap(text.range(of: line), "missing: \(line)")
            if let lastRange {
                XCTAssertGreaterThan(range.lowerBound, lastRange.upperBound,
                                     "timeline lines out of order in the text")
            }
            lastRange = range
        }
        // Window rate present where the generation covers the window.
        XCTAssertNotNil(decoded.generations[0].canonicalWindowTokensPerSecond)
        XCTAssertNil(decoded.generations[2].canonicalWindowTokensPerSecond)
        XCTAssertTrue(decoded.generations[0].textLine.contains("window"))
        // Elapsed offsets are the loop's clock, monotone.
        let offsets = decoded.generations.map(\.elapsedAtEndSeconds)
        XCTAssertEqual(offsets, offsets.sorted())
    }

    func testSustainedModeTimelineUsesTheLoopOffsetsWhenPresent() throws {
        let gens = [generation(tokens: 600), generation(tokens: 600, stop: .stopRequested)]
        let result = SustainedLoopResult(
            generations: gens, totalElapsedSeconds: 312,
            lastGenerationTruncated: true,
            generationEndOffsetsSeconds: [151.2, 312])
        let r = report(mode: .sustained, soc: nil, sustained: result)
        let export = try XCTUnwrap(r.timelineExport())
        XCTAssertEqual(export.mode, "sustained")
        XCTAssertEqual(export.generations.map(\.elapsedAtEndSeconds), [151.2, 312])
        XCTAssertNil(export.battery, "no SoC struct → no battery block")
        XCTAssertEqual(export.cycle?.endedBy, "durationBound")
        XCTAssertEqual(export.cycle?.lastGenerationTruncated, true)
        // Round-trip through the JSON string too.
        let json = try XCTUnwrap(r.timelineJSON())
        XCTAssertEqual(
            try JSONDecoder().decode(TimelineExport.self, from: Data(json.utf8)),
            export)
        // The text export cites the same lines.
        let text = r.exportText()
        for entry in export.generations {
            XCTAssertTrue(text.contains(entry.textLine), entry.textLine)
        }
    }

    func testSustainedResultWithoutOffsetsFallsBackToSummedWallLabeled() throws {
        let result = SustainedLoopResult(
            generations: [generation(tokens: 600), generation(tokens: 600)],
            totalElapsedSeconds: 312, lastGenerationTruncated: false)
        let export = try XCTUnwrap(
            report(mode: .sustained, soc: nil, sustained: result).timelineExport())
        XCTAssertEqual(export.elapsedBasis, "sumOfGenerationWall")
        let w = 600 * 0.25 + 0.7
        XCTAssertEqual(export.generations.map(\.elapsedAtEndSeconds), [w, 2 * w])
    }

    func testBurstModeTimelineIsASingleEntry() throws {
        let export = try XCTUnwrap(
            report(mode: .burst, soc: nil, burst: generation(tokens: 640)).timelineExport())
        XCTAssertEqual(export.mode, "burst")
        XCTAssertEqual(export.generations.count, 1)
        XCTAssertNil(export.cycle)
    }

    func testTimelineIsNilWithoutGenerations() {
        XCTAssertNil(report(mode: .burst, soc: nil).timelineExport())
        XCTAssertNil(report(mode: .energy, soc: nil).timelineExport())
        XCTAssertNil(try report(mode: .energy, soc: nil).timelineJSON())
    }

    func testTimelineJSONIsDeterministic() throws {
        let r = report(energy: energyResult())
        XCTAssertEqual(try r.timelineJSON(), try r.timelineJSON())
        let json = try XCTUnwrap(r.timelineJSON())
        XCTAssertTrue(json.contains("\"schema\""))
        // Sorted keys: "battery" precedes "cycle" precedes "generations".
        let b = try XCTUnwrap(json.range(of: "\"battery\""))
        let c = try XCTUnwrap(json.range(of: "\"cycle\""))
        let g = try XCTUnwrap(json.range(of: "\"generations\""))
        XCTAssertLessThan(b.lowerBound, c.lowerBound)
        XCTAssertLessThan(c.lowerBound, g.lowerBound)
    }

    // MARK: - edge test 4: round marker gates the PROVISIONAL header

    func testHeaderIsProvisionalUnlessTheRoundIsSet() {
        let provisional = report(round: nil, energy: energyResult()).exportText()
        XCTAssertTrue(provisional.hasPrefix("qwen-metal Phase 6 row export (PROVISIONAL)"),
                      provisional)
        XCTAssertTrue(provisional.contains("round: (none — PROVISIONAL)"), provisional)

        let blank = report(round: "   ", energy: energyResult()).exportText()
        XCTAssertTrue(blank.hasPrefix("qwen-metal Phase 6 row export (PROVISIONAL)"),
                      "a whitespace-only round is not a round")

        let inRound = report(
            round: "phase6-speed-2026-10-01", energy: energyResult()).exportText()
        XCTAssertTrue(inRound.hasPrefix(
            "qwen-metal Phase 6 row export (round: phase6-speed-2026-10-01)"), inRound)
        XCTAssertFalse(inRound.contains("PROVISIONAL"), inRound)
        XCTAssertTrue(inRound.contains("round: phase6-speed-2026-10-01"), inRound)
    }

    func testRoundMarkerAppliesToBurstAndSustainedRowsToo() {
        let burst = report(
            mode: .burst, soc: nil, round: "phase6-speed-2026-10-01",
            burst: generation(tokens: 640)).exportText()
        XCTAssertTrue(burst.hasPrefix("qwen-metal Phase 6 row export (round: "), burst)
        XCTAssertFalse(burst.contains("PROVISIONAL"))
        let sustained = report(
            mode: .sustained, soc: nil, round: "phase6-speed-2026-10-01",
            sustained: SustainedLoopResult(
                generations: [generation(tokens: 600)], totalElapsedSeconds: 300,
                lastGenerationTruncated: false)).exportText()
        XCTAssertFalse(sustained.contains("PROVISIONAL"))
    }

    func testBF16RowsStayPhase2AndProvisionalEvenInsideARound() {
        // bf16 is the permanent Phase 2 correctness artifact — never a Phase
        // 6 headline row; the round marker does not lift its marker.
        let text = report(
            mode: .burst, soc: nil, round: "phase6-speed-2026-10-01",
            weightsFormat: .bf16, burst: generation(tokens: 64)).exportText()
        XCTAssertTrue(text.hasPrefix("qwen-metal Phase 2 row export (PROVISIONAL)"), text)
        XCTAssertTrue(text.contains("round: phase6-speed-2026-10-01 (bf16 rows stay PROVISIONAL"), text)
    }

    func testTimelineProvisionalFlagFollowsTheRound() throws {
        let provisional = try XCTUnwrap(report(energy: energyResult()).timelineExport())
        XCTAssertTrue(provisional.provisional)
        XCTAssertNil(provisional.round)
        let inRound = try XCTUnwrap(
            report(round: "phase6-energy-2026-10-01", energy: energyResult()).timelineExport())
        XCTAssertFalse(inRound.provisional)
    }
}
