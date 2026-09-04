import Foundation
import SwiftUI
import UIKit
import QwenMetalEngine

// P2-6 (phase-2.md D8): the app-side glue. THIN by rule — model discovery,
// load-state publishing, and view wiring only; generation, the benchmark
// protocol, timing, and row export all live in QwenMetalEngine.

/// Thread-safe stop flag; the engine polls it at token boundaries via
/// `shouldStop` (never mid-forward).
final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false
    func request() { lock.lock(); stopped = true; lock.unlock() }
    func reset() { lock.lock(); stopped = false; lock.unlock() }
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return stopped }
}

/// The two pinned benchmark prompts, bundled in-app (D8 tester-friction
/// controls — the P0A-1 clipboard lesson). Bundled bytes are drift-tested
/// byte-identical to benchmarks/prompts/rendered/ by AppBundledPromptTests.
enum BundledPrompt: String, CaseIterable, Identifiable {
    case decodeEssay = "decode-essay"
    case prefillSummarize = "prefill-summarize"
    var id: String { rawValue }

    /// Exact rendered bytes — trailing newlines preserved (the CLI-1
    /// `$(cat)` footgun is why these are bundled files, not pasted strings).
    func text() throws -> String {
        guard let url = Bundle.main.url(
            forResource: "\(rawValue).rendered", withExtension: "txt")
        else { throw AppError.missingBundledPrompt(rawValue) }
        return try String(contentsOf: url, encoding: .utf8)
    }
}

enum AppError: Error, CustomStringConvertible {
    case missingBundledPrompt(String)
    case noModelDirectory(searched: String)

    var description: String {
        switch self {
        case .missingBundledPrompt(let name):
            return "bundled prompt '\(name).rendered.txt' missing from the app bundle"
        case .noModelDirectory(let searched):
            return "no model directory found under \(searched)\n\nCopy the "
                + "pinned model folder (one .safetensors checkpoint + "
                + "config.json + tokenizer.json + tokenizer_config.json "
                + "[+ generation_config.json]) into this app's Documents "
                + "folder via Finder file sharing, then reload."
        }
    }
}

/// Everything held after a successful load — engine objects only.
struct LoadedEngine {
    let modelDirectoryName: String
    let residency: WeightsResidency
    let weightsFormat: WeightsFormat
    let contextLimit: Int
    let gpuModel: GPUModel
    let tokenizer: TextTokenizer
    let stopTokenIds: Set<Int>
    let loadSeconds: Double
}

@MainActor
final class AppModel: ObservableObject {
    /// Same cap as the CLI: the GPU KV cache is preallocated at this size
    /// (phase-2.md D3); the model's max_position_embeddings wins if smaller.
    nonisolated static let contextCap = 4096

    @Published var residency: WeightsResidency = .mmap
    /// P3-5: q4g64 default — Phase 3 device rows (P3-7) run on the packed
    /// artifact; a missing artifact fails with the clear noPackedCheckpoint
    /// error, and the toggle drops back to bf16 for Phase 2-style rows.
    @Published var weightsFormat: WeightsFormat = .q4g64
    @Published var isLoading = false
    @Published var isRunning = false {
        // A locked screen suspends the app mid-generation and ruins the
        // run (P2-7 sustained loops are ≥ 5 min) — keep the display awake
        // exactly while a run is active.
        didSet { UIApplication.shared.isIdleTimerDisabled = isRunning }
    }
    @Published var loadSummary: String?
    @Published var errorMessage: String?
    @Published var statusLine = ""
    @Published var outputText = ""
    @Published var lastReport: String?
    @Published var lastPrompt: String?

    private(set) var engine: LoadedEngine?
    private let stopFlag = StopFlag()

    func requestStop() { stopFlag.request() }

    /// Residency is baked into the weights buffer at load (spec D1), so a
    /// toggle drops the engine; the next run reloads in the new mode.
    func residencyChanged() {
        if let engine, engine.residency != residency {
            self.engine = nil
            loadSummary = nil
        }
    }

    /// Same contract for the weights format (P3-5): the kernels and offsets
    /// are resolved at load, so a toggle reloads.
    func weightsFormatChanged() {
        if let engine, engine.weightsFormat != weightsFormat {
            self.engine = nil
            loadSummary = nil
        }
    }

    func loadModel() async {
        guard !isRunning, !isLoading else { return }
        isLoading = true
        errorMessage = nil
        engine = nil
        loadSummary = nil
        do {
            _ = try await loadEngineIfNeeded()
        } catch {
            show(error)
        }
        isLoading = false
    }

    // MARK: - Generate screen

    func generate(prompt: String, maxNewTokens: Int) async {
        guard !isRunning, !isLoading else { return }
        isRunning = true
        errorMessage = nil
        outputText = ""
        stopFlag.reset()
        lastPrompt = prompt
        defer { isRunning = false }
        do {
            let engine = try await loadEngineIfNeeded()
            let stopFlag = self.stopFlag
            statusLine = "generating…"
            let (text, metrics): (String, GenerationMetrics) =
                try await Task.detached(priority: .userInitiated) {
                    let promptIds = engine.tokenizer.encode(prompt)
                    let runner = BenchGenerationRunner(
                        gpuModel: engine.gpuModel,
                        maxContext: engine.contextLimit,
                        eosTokenIds: engine.stopTokenIds)
                    let run = try runner.run(
                        promptIds: promptIds, maxNewTokens: maxNewTokens,
                        shouldStop: { stopFlag.isSet },
                        onToken: { step, _ in self.postProgress(step) })
                    let text = engine.tokenizer.decode(
                        run.tokenIds, skipSpecialTokens: true)
                    return (text, run.metrics)
                }.value
            outputText = text
            statusLine = Self.summaryLine(metrics)
        } catch {
            show(error)
        }
    }

    // MARK: - Benchmark screen (pinned protocol via the engine harness)

    func runBurst(
        prompt: BundledPrompt, batteryNote: String, coldWarmNote: String
    ) async {
        guard !isRunning, !isLoading else { return }
        isRunning = true
        errorMessage = nil
        lastReport = nil
        stopFlag.reset()
        defer { isRunning = false }
        do {
            let engine = try await loadEngineIfNeeded()
            let promptText = try prompt.text()
            let stopFlag = self.stopFlag
            statusLine = "burst run (\(prompt.rawValue))…"
            let metrics: GenerationMetrics =
                try await Task.detached(priority: .userInitiated) {
                    let promptIds = engine.tokenizer.encode(promptText)
                    let runner = BenchGenerationRunner(
                        gpuModel: engine.gpuModel,
                        maxContext: engine.contextLimit,
                        eosTokenIds: engine.stopTokenIds)
                    return try runner.run(
                        promptIds: promptIds,
                        maxNewTokens: BenchDefaults.burstMaxNewTokens,
                        shouldStop: { stopFlag.isSet },
                        onToken: { step, _ in self.postProgress(step) }
                    ).metrics
                }.value
            lastReport = report(
                mode: .burst, promptName: prompt.rawValue,
                promptTokenCount: metrics.promptTokenCount,
                batteryNote: batteryNote, coldWarmNote: coldWarmNote,
                residency: engine.residency,
                weightsFormat: engine.weightsFormat, burst: metrics)
            statusLine = stopFlag.isSet
                ? "burst stopped early — report reflects the partial run"
                : "burst complete"
        } catch {
            show(error)
        }
    }

    /// Sustained rows are pinned to decode-essay (prompts/README role
    /// separation); the regenerate policy is the engine's SustainedLoop.
    func runSustained(batteryNote: String, coldWarmNote: String) async {
        guard !isRunning, !isLoading else { return }
        isRunning = true
        errorMessage = nil
        lastReport = nil
        stopFlag.reset()
        defer { isRunning = false }
        do {
            let engine = try await loadEngineIfNeeded()
            let promptText = try BundledPrompt.decodeEssay.text()
            let stopFlag = self.stopFlag
            statusLine = "sustained loop (≥ 5 min)…"
            let result: SustainedLoopResult =
                try await Task.detached(priority: .userInitiated) {
                    let promptIds = engine.tokenizer.encode(promptText)
                    let runner = BenchGenerationRunner(
                        gpuModel: engine.gpuModel,
                        maxContext: engine.contextLimit,
                        eosTokenIds: engine.stopTokenIds)
                    let loop = SustainedLoop(
                        minDurationSeconds:
                            BenchDefaults.sustainedMinDurationSeconds)
                    return try loop.run { loopStop in
                        if stopFlag.isSet { throw CancellationError() }
                        // Context-fill regenerate policy: each generation may
                        // run until the 4K context fills (or EOS).
                        return try runner.run(
                            promptIds: promptIds,
                            maxNewTokens: engine.contextLimit - promptIds.count,
                            shouldStop: { loopStop() || stopFlag.isSet },
                            onToken: { step, _ in self.postProgress(step) }
                        ).metrics
                    }
                }.value
            lastReport = report(
                mode: .sustained,
                promptName: BundledPrompt.decodeEssay.rawValue,
                promptTokenCount:
                    result.generations.first?.promptTokenCount ?? 0,
                batteryNote: batteryNote, coldWarmNote: coldWarmNote,
                residency: engine.residency,
                weightsFormat: engine.weightsFormat, sustained: result)
            statusLine = "sustained loop complete"
        } catch is CancellationError {
            statusLine = "sustained loop aborted by Stop — no report"
        } catch {
            show(error)
        }
    }

    // MARK: - Internals

    /// Loads (off the main thread) if there is no engine for the selected
    /// residency yet. Errors propagate to the caller's `show(_:)`.
    private func loadEngineIfNeeded() async throws -> LoadedEngine {
        if let engine, engine.residency == residency,
           engine.weightsFormat == weightsFormat { return engine }
        let residency = self.residency
        let weightsFormat = self.weightsFormat
        statusLine = "loading model (weights \(weightsFormat.rawValue), "
            + "residency \(residency.rawValue))…"
        let loaded: LoadedEngine =
            try await Task.detached(priority: .userInitiated) {
                let start = Date()
                let directory = try Self.locateModelDirectory()
                let config = try ModelConfig.load(path: directory.configURL.path)
                let contextLimit = min(
                    Self.contextCap, config.maxPositionEmbeddings)
                // No Metal device → MetalHarnessError.noDevice, a clear
                // error, not a crash (spec edge case 10).
                let metal = try MetalContext()
                let gpu: GPUModel
                switch weightsFormat {
                case .bf16:
                    let checkpoint = try SafetensorsFile(
                        path: directory.checkpointURL.path)
                    gpu = try GPUModel(
                        checkpoint: checkpoint, config: config, context: metal,
                        residency: residency, maxContext: contextLimit)
                case .q4g64:
                    // Missing artifact → the clear noPackedCheckpoint error
                    // (P3-5 edge behavior — never a crash).
                    let packed = try PackedCheckpoint(
                        path: directory.requirePackedCheckpoint().path)
                    gpu = try GPUModel(
                        packed: packed, config: config, context: metal,
                        residency: residency, maxContext: contextLimit)
                }
                let tokenizer = try await TextTokenizer(
                    modelFolder: directory.directoryURL)
                let stops = try directory.stopTokenIds(
                    config: config, tokenizerEOSTokenId: tokenizer.eosTokenId)
                return LoadedEngine(
                    modelDirectoryName:
                        directory.directoryURL.lastPathComponent,
                    residency: residency, weightsFormat: weightsFormat,
                    contextLimit: contextLimit,
                    gpuModel: gpu, tokenizer: tokenizer, stopTokenIds: stops,
                    loadSeconds: Date().timeIntervalSince(start))
            }.value
        engine = loaded
        loadSummary = String(
            format: "%@ — loaded in %.1f s, weights %@, residency %@, context %d",
            loaded.modelDirectoryName, loaded.loadSeconds,
            loaded.weightsFormat.rawValue,
            loaded.residency.rawValue, loaded.contextLimit)
        return loaded
    }

    /// Documents-first search: Documents itself, then each subdirectory —
    /// first one that validates as a ModelDirectory wins.
    nonisolated private static func locateModelDirectory() throws -> ModelDirectory {
        let documents = FileManager.default.urls(
            for: .documentDirectory, in: .userDomainMask)[0]
        var candidates = [documents]
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: documents, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])) ?? []
        candidates += contents.filter {
            (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory)
                == true
        }
        for url in candidates {
            if let directory = try? ModelDirectory(validating: url) {
                return directory
            }
        }
        throw AppError.noModelDirectory(searched: documents.path)
    }

    nonisolated private func postProgress(_ step: Int) {
        guard step.isMultiple(of: 16) else { return }
        Task { @MainActor in
            self.statusLine = "generating… token \(step + 1)"
        }
    }

    private func show(_ error: Error) {
        errorMessage = String(describing: error)
        statusLine = ""
    }

    private static func summaryLine(_ m: GenerationMetrics) -> String {
        var line = String(
            format: "%d tokens in %.1f s (stop: %@)",
            m.generatedTokenCount, m.wallSeconds, m.stopReason.rawValue)
        if let overall = m.overallTokensPerSecond {
            line += String(format: ", overall %.2f tok/s", overall)
        }
        if let t = m.timing {
            line += String(
                format: ", median GPU %.1f ms, %d dispatches/token",
                t.medianGPUSeconds * 1000, t.maxDispatchCount)
        }
        return line
    }

    private func report(
        mode: BenchmarkReport.Mode, promptName: String, promptTokenCount: Int,
        batteryNote: String, coldWarmNote: String,
        residency: WeightsResidency, weightsFormat: WeightsFormat,
        burst: GenerationMetrics? = nil,
        sustained: SustainedLoopResult? = nil
    ) -> String {
        BenchmarkReport(
            dateStamp: Self.dateStamp(),
            deviceLabel: Self.deviceModelIdentifier(),
            osVersion: Self.osVersionString(),
            batteryHealthNote: batteryNote, coldOrWarmNote: coldWarmNote,
            residency: residency, weightsFormat: weightsFormat,
            promptName: promptName,
            promptTokenCount: promptTokenCount, mode: mode,
            burst: burst, sustained: sustained,
            physFootprintBytes: MemoryFootprint.currentPhysFootprintBytes()
        ).exportText()
    }

    nonisolated private static func dateStamp() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate]
        return formatter.string(from: Date())
    }

    /// Hardware identifier (e.g. "iPhone16,1" — the pinned iPhone 15 Pro).
    nonisolated private static func deviceModelIdentifier() -> String {
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafeBytes(of: &systemInfo.machine) { buffer in
            String(decoding: buffer.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }

    nonisolated private static func osVersionString() -> String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }
}
