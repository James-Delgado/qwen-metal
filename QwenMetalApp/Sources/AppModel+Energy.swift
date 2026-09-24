import Foundation
import UIKit
import QwenMetalEngine

// P6-1 (phase-6.md D4): the energy-mode run. Thin by rule — the loop, the
// stop semantics, and the export all live in QwenMetalEngine; the app only
// reads what it can read (the programmatic SoC cross-check) and hands the
// operator fields through.

extension AppModel {
    /// The energy cycle: decode-essay regenerate loop, OPERATOR-bounded.
    /// The operator presses Run at exactly 80% SoC (Settings → Battery) and
    /// Stop at exactly 70%; the loop ends at the next token boundary and
    /// ALWAYS reports (unlike the P2-6 sustained mode, where Stop aborts).
    /// No safety bound on device — the protocol's stop is the operator's.
    /// Programmatic SoC is read at start and end as the cross-check field;
    /// the Settings readings are typed by the operator (SoC start before,
    /// SoC end after) — read live from `operatorFields` at publish time and
    /// re-rendered on later edits via `operatorFieldsChanged`.
    func runEnergy() async {
        guard !isRunning, !isLoading else { return }
        isRunning = true
        errorMessage = nil
        clearExports()
        stopFlag.reset()
        defer { isRunning = false }
        do {
            let engine = try await loadEngineIfNeeded()
            let promptText = try BundledPrompt.decodeEssay.text()
            let stopFlag = self.stopFlag
            let socStart = Self.programmaticStateOfChargePercent()
            statusLine = "energy cycle running — press Stop at exactly 70% SoC…"
            let result: EnergyLoopResult =
                try await Task.detached(priority: .userInitiated) {
                    let promptIds = engine.tokenizer.encode(promptText)
                    let runner = BenchGenerationRunner(
                        gpuModel: engine.gpuModel,
                        maxContext: engine.contextLimit,
                        eosTokenIds: engine.stopTokenIds)
                    return try EnergyLoop().run(
                        operatorStop: { stopFlag.isSet },
                        generate: { shouldStop in
                            // Context-fill regenerate policy, as sustained.
                            try runner.run(
                                promptIds: promptIds,
                                maxNewTokens: engine.contextLimit - promptIds.count,
                                shouldStop: shouldStop,
                                onToken: { step, _ in self.postProgress(step) }
                            ).metrics
                        })
                }.value
            let socEnd = Self.programmaticStateOfChargePercent()
            publishReport(makeReport(
                mode: .energy,
                promptName: BundledPrompt.decodeEssay.rawValue,
                promptTokenCount:
                    result.generations.first?.promptTokenCount ?? 0,
                residency: engine.residency,
                weightsFormat: engine.weightsFormat,
                kernelPath: engine.gpuModel.kernelPath,
                prefillPath: engine.gpuModel.prefillPath,
                prefillChunkSize: engine.gpuModel.prefillChunkSize,
                prefillAttention: engine.gpuModel.prefillAttention,
                energy: result,
                batteryStateOfCharge: BatteryStateOfCharge(
                    operatorStartNote: operatorFields.socStart,
                    operatorEndNote: operatorFields.socEnd,
                    programmaticStartPercent: socStart,
                    programmaticEndPercent: socEnd)))
            statusLine = String(
                format: "energy cycle stopped — %d generations, %d tokens; "
                    + "now type the Settings SoC at stop (end) and the "
                    + "battery health, then share both exports",
                result.generations.count, result.totalGeneratedTokens)
        } catch {
            show(error)
        }
    }

    /// `UIDevice.batteryLevel` in percent; nil when the platform reports
    /// none (-1). Cross-check only — Settings → Battery is the value of
    /// record at the band marks (phase-6.md D4).
    static func programmaticStateOfChargePercent() -> Double? {
        let device = UIDevice.current
        device.isBatteryMonitoringEnabled = true
        let level = device.batteryLevel
        guard level >= 0 else { return nil }
        return Double(level) * 100
    }
}
