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
    case prefillAttributionNeedsTiled

    var description: String {
        switch self {
        case .missingBundledPrompt(let name):
            return "bundled prompt '\(name).rendered.txt' missing from the app bundle"
        case .prefillAttributionNeedsTiled:
            return "prefill attribution needs the tiled prefill path — set "
                + "Weights q4g64, Kernels fused, Prefill tiled and reload"
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
    /// P4-4 (phase-4.md D4): fused default; the naive selection exists for
    /// the P4-5 interleaved before/after row. q4g64 only — the bf16 backend
    /// is permanently naive and ignores this.
    @Published var kernelPath: GPUModel.KernelPath = .fused
    /// P5-4 (phase-5.md D5): tiled default; the sequential selection exists
    /// for the P5-5 interleaved sequential-vs-tiled before/after row.
    /// q4g64 + fused only — the bf16 backend keeps sequential prefill
    /// permanently, and the naive kernel arm supports sequential only (the
    /// engine resolves that; the picker hides itself there).
    @Published var prefillPath: GPUModel.PrefillPath = .tiled
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

    /// Same contract for the kernel path (P4-4): pipelines and scratch
    /// buffers are built at load, so a toggle reloads. Only meaningful on
    /// q4g64 (the bf16 model is always naive — no reload needed there).
    func kernelPathChanged() {
        if let engine, engine.weightsFormat == .q4g64,
           engine.gpuModel.kernelPath != kernelPath {
            self.engine = nil
            loadSummary = nil
        }
    }

    /// Same contract for the prefill path (P5-4): the chunk scratch is
    /// preallocated at load, so a toggle reloads. Only meaningful on
    /// q4g64 + fused (elsewhere the engine runs sequential regardless).
    func prefillPathChanged() {
        if let engine, engine.weightsFormat == .q4g64,
           engine.gpuModel.prefillPath != requestedPrefillPath() {
            self.engine = nil
            loadSummary = nil
        }
    }

    /// The prefill path a load with the current toggles runs: the picker's
    /// value on q4g64 + fused, the engine's sequential-only resolution on
    /// the naive kernel arm (mirrors `GPUModel.defaultPrefillPath(for:)`).
    private func requestedPrefillPath() -> GPUModel.PrefillPath {
        kernelPath == .fused
            ? prefillPath : GPUModel.defaultPrefillPath(for: kernelPath)
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
                weightsFormat: engine.weightsFormat,
                kernelPath: engine.gpuModel.kernelPath,
                prefillPath: engine.gpuModel.prefillPath,
                prefillChunkSize: engine.gpuModel.prefillChunkSize,
                burst: metrics)
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
                weightsFormat: engine.weightsFormat,
                kernelPath: engine.gpuModel.kernelPath,
                prefillPath: engine.gpuModel.prefillPath,
                prefillChunkSize: engine.gpuModel.prefillChunkSize,
                sustained: result)
            statusLine = "sustained loop complete"
        } catch is CancellationError {
            statusLine = "sustained loop aborted by Stop — no report"
        } catch {
            show(error)
        }
    }

    /// Which standalone weight-sweep microbench the Benchmark screen runs:
    /// the P3-6 dequant-matvec bench (Phase 3 D7 gate) or the P5-2 tiled
    /// dequant-GEMM M-sweep (Phase 5 D7: M=8 fraction gate + the GB/s and
    /// GFLOPS curve at M ∈ {8, 64, 512}, reported). Both are weights-only
    /// and share the 197-matrix site roster; the export shapes are the
    /// CLI's (`microbench --kernel matvec|gemm`).
    enum MicrobenchKernel: String, CaseIterable, Identifiable {
        case matvec
        case gemm
        var id: String { rawValue }
    }

    /// P3-6 (spec D7) / P5-2 (phase-5.md D7): the standalone dequant
    /// microbenches. Weights-only by construction — they load the packed
    /// artifact directly (no GPUModel, no KV cache) and honor the residency
    /// toggle. The gates (matvec 30.7 GB/s; GEMM M=8 ≥ 30.69 GB/s) are
    /// evaluated by James over the D8 repeats protocol (≥3 same-session
    /// runs of this button, detached, best-of); the app just reports each
    /// run's numbers. GEMM runs the engine's default M list and iteration
    /// counts (the P5-2 protocol shape, identical to the CLI default).
    func runMicrobench(
        kernel: MicrobenchKernel, batteryNote: String, coldWarmNote: String
    ) async {
        guard !isRunning, !isLoading else { return }
        isRunning = true
        errorMessage = nil
        lastReport = nil
        defer { isRunning = false }
        do {
            let residency = self.residency
            statusLine = kernel == .matvec
                ? "microbench (197 packed matvecs, residency "
                    + "\(residency.rawValue))…"
                : "microbench (tiled GEMM M-sweep "
                    + "\(QuantGemmMicrobench.defaultMValues), residency "
                    + "\(residency.rawValue))…"
            let report: String =
                try await Task.detached(priority: .userInitiated) {
                    let directory = try Self.locateModelDirectory()
                    let config = try ModelConfig.load(
                        path: directory.configURL.path)
                    let packed = try PackedCheckpoint(
                        path: directory.requirePackedCheckpoint().path)
                    let context = try MetalContext()
                    switch kernel {
                    case .matvec:
                        let bench = try QuantMatvecMicrobench(
                            packed: packed, config: config,
                            context: context, residency: residency)
                        let result = try bench.run()
                        return result.exportText(
                            dateStamp: Self.dateStamp(),
                            deviceLabel: Self.deviceModelIdentifier(),
                            osVersion: "iOS \(Self.osVersionString())",
                            batteryHealthNote: batteryNote,
                            coldOrWarmNote: coldWarmNote,
                            residency: residency)
                    case .gemm:
                        let bench = try QuantGemmMicrobench(
                            packed: packed, config: config,
                            context: context, residency: residency)
                        let result = try bench.run(
                            mValues: QuantGemmMicrobench.defaultMValues,
                            warmupIterations:
                                QuantGemmMicrobench.defaultWarmupIterations,
                            measuredIterations:
                                QuantGemmMicrobench.defaultMeasuredIterations)
                        return result.exportText(
                            dateStamp: Self.dateStamp(),
                            deviceLabel: Self.deviceModelIdentifier(),
                            osVersion: "iOS \(Self.osVersionString())",
                            batteryHealthNote: batteryNote,
                            coldOrWarmNote: coldWarmNote,
                            residency: residency)
                    }
                }.value
            lastReport = report
            statusLine = "microbench complete"
        } catch {
            show(error)
        }
    }

    /// P4-1 (phase-4.md D1): the diagnostic per-kernel-class attribution
    /// run — decode-essay prompt, interleaved attributed/production
    /// forwards via the engine's AttributionRunner. DIAGNOSTIC ONLY: the
    /// export is never a benchmark row (the P4-5 on-device breakdown James
    /// records comes from this button, labeled as diagnostic).
    /// Which attribution the Benchmark screen runs: the P4-1 per-token
    /// DECODE breakdown (decode-essay, 64 interleaved forwards) or the PF-1
    /// PREFILL breakdown (prefill-summarize, interleaved class-split vs
    /// production tiled prefills). Both DIAGNOSTIC — never rows.
    enum AttributionMode: String, CaseIterable, Identifiable {
        case decode
        case prefill
        var id: String { rawValue }
    }

    func runAttribution(mode: AttributionMode = .decode) async {
        guard !isRunning, !isLoading else { return }
        isRunning = true
        errorMessage = nil
        lastReport = nil
        stopFlag.reset()
        defer { isRunning = false }
        do {
            let engine = try await loadEngineIfNeeded()
            let stopFlag = self.stopFlag
            if mode == .prefill {
                // PF-1: needs the tiled path (q4g64 + fused + Prefill tiled).
                guard engine.gpuModel.prefillPath == .tiled else {
                    throw AppError.prefillAttributionNeedsTiled
                }
                let promptText = try BundledPrompt.prefillSummarize.text()
                statusLine = "prefill attribution run (DIAGNOSTIC, "
                    + "\(BenchDefaults.prefillAttributionRuns) interleaved "
                    + "prefills)…"
                let report: String =
                    try await Task.detached(priority: .userInitiated) {
                        let promptIds = engine.tokenizer.encode(promptText)
                        let runner = PrefillAttributionRunner(
                            gpuModel: engine.gpuModel,
                            maxContext: engine.contextLimit)
                        let result = try runner.run(
                            promptIds: promptIds,
                            runs: BenchDefaults.prefillAttributionRuns,
                            shouldStop: { stopFlag.isSet },
                            onRun: { run in self.postProgress(run * 16) })
                        return result.exportText(
                            dateStamp: Self.dateStamp(),
                            deviceLabel: Self.deviceModelIdentifier(),
                            osVersion: "iOS \(Self.osVersionString())",
                            residency: engine.residency)
                    }.value
                lastReport = report
                statusLine = stopFlag.isSet
                    ? "prefill attribution stopped early — partial diagnostic"
                    : "prefill attribution complete"
                return
            }
            let promptText = try BundledPrompt.decodeEssay.text()
            statusLine = "attribution run (DIAGNOSTIC, "
                + "\(BenchDefaults.attributionDecodeTokens) forwards)…"
            let report: String =
                try await Task.detached(priority: .userInitiated) {
                    let promptIds = engine.tokenizer.encode(promptText)
                    let runner = AttributionRunner(
                        gpuModel: engine.gpuModel,
                        maxContext: engine.contextLimit,
                        eosTokenIds: engine.stopTokenIds)
                    let result = try runner.run(
                        promptIds: promptIds,
                        decodeTokens: BenchDefaults.attributionDecodeTokens,
                        shouldStop: { stopFlag.isSet },
                        onStep: { step in self.postProgress(step) })
                    return result.exportText(
                        dateStamp: Self.dateStamp(),
                        deviceLabel: Self.deviceModelIdentifier(),
                        osVersion: "iOS \(Self.osVersionString())",
                        residency: engine.residency)
                }.value
            lastReport = report
            statusLine = stopFlag.isSet
                ? "attribution stopped early — partial diagnostic"
                : "attribution complete"
        } catch {
            show(error)
        }
    }

    /// OA-1 (seeded by P4-9): the overhead-anatomy diagnostic run —
    /// decode-essay prompt, round-robin production/anatomy/unretained
    /// forwards via the engine's OverheadAnatomyRunner. DIAGNOSTIC ONLY:
    /// the export is never a benchmark row (the P4-11 device span split
    /// James records comes from this button, labeled as diagnostic).
    func runOverheadAnatomy() async {
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
            statusLine = "overhead anatomy run (DIAGNOSTIC, "
                + "\(BenchDefaults.overheadAnatomyDecodeTokens) forwards)…"
            let report: String =
                try await Task.detached(priority: .userInitiated) {
                    let promptIds = engine.tokenizer.encode(promptText)
                    let runner = OverheadAnatomyRunner(
                        gpuModel: engine.gpuModel,
                        maxContext: engine.contextLimit,
                        eosTokenIds: engine.stopTokenIds)
                    let result = try runner.run(
                        promptIds: promptIds,
                        decodeTokens: BenchDefaults.overheadAnatomyDecodeTokens,
                        shouldStop: { stopFlag.isSet },
                        onStep: { step in self.postProgress(step) })
                    return result.exportText(
                        dateStamp: Self.dateStamp(),
                        deviceLabel: Self.deviceModelIdentifier(),
                        osVersion: "iOS \(Self.osVersionString())",
                        residency: engine.residency)
                }.value
            lastReport = report
            statusLine = stopFlag.isSet
                ? "overhead anatomy stopped early — partial diagnostic"
                : "overhead anatomy complete"
        } catch {
            show(error)
        }
    }

    // MARK: - Internals

    /// Loads (off the main thread) if there is no engine for the selected
    /// residency yet. Errors propagate to the caller's `show(_:)`.
    private func loadEngineIfNeeded() async throws -> LoadedEngine {
        if let engine, engine.residency == residency,
           engine.weightsFormat == weightsFormat,
           weightsFormat == .bf16
               || (engine.gpuModel.kernelPath == kernelPath
                   && engine.gpuModel.prefillPath == requestedPrefillPath()) {
            return engine
        }
        let residency = self.residency
        let weightsFormat = self.weightsFormat
        let kernelPath = self.kernelPath
        let prefillPath = requestedPrefillPath()
        statusLine = "loading model (weights \(weightsFormat.rawValue), "
            + "residency \(residency.rawValue)"
            + (weightsFormat == .q4g64
                ? ", kernels \(kernelPath.rawValue), prefill \(prefillPath.rawValue)"
                : "")
            + ")…"
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
                    // P5-4 (spec D5): the prefill toggle; tiled is the
                    // fused default, sequential the P5-5 A/B arm. Chunk
                    // size stays the engine's measured default (C=512).
                    gpu = try GPUModel(
                        packed: packed, config: config, context: metal,
                        residency: residency, maxContext: contextLimit,
                        kernelPath: kernelPath, prefillPath: prefillPath)
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
            format: "%@ — loaded in %.1f s, weights %@, residency %@, "
                + "kernels %@, prefill %@, context %d",
            loaded.modelDirectoryName, loaded.loadSeconds,
            loaded.weightsFormat.rawValue,
            loaded.residency.rawValue,
            loaded.gpuModel.kernelPath.rawValue,
            loaded.gpuModel.prefillPath == .tiled
                ? "tiled (C=\(loaded.gpuModel.prefillChunkSize))" : "sequential",
            loaded.contextLimit)
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
        kernelPath: GPUModel.KernelPath,
        prefillPath: GPUModel.PrefillPath,
        prefillChunkSize: Int,
        burst: GenerationMetrics? = nil,
        sustained: SustainedLoopResult? = nil
    ) -> String {
        BenchmarkReport(
            dateStamp: Self.dateStamp(),
            deviceLabel: Self.deviceModelIdentifier(),
            osVersion: Self.osVersionString(),
            batteryHealthNote: batteryNote, coldOrWarmNote: coldWarmNote,
            residency: residency, weightsFormat: weightsFormat,
            kernelPath: kernelPath,
            prefillPath: prefillPath,
            prefillChunkSize: prefillPath == .tiled ? prefillChunkSize : nil,
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
