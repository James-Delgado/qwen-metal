import Foundation
import QwenMetalEngine

// P4-1 (phase-4.md D1): `qwen-metal-cli attribute` — the diagnostic
// per-kernel-class GPU attribution run. Thin: argument handling and printing
// only; the interleaved harness and report formatting live in
// QwenMetalEngine (AttributionRunner / AttributionRunResult).
// DIAGNOSTIC output — never a benchmark row.

private let attributeUsage = """
usage: qwen-metal-cli attribute --model-dir <dir> --prompt "<text>" \
[--mode decode|prefill] [--tokens N] [--runs N] [--weights bf16|q4g64] \
[--residency mmap|wired] [--kernels naive|fused] [--prefill-chunk C] \
[--prefill-attention query-tiled|per-position]
  --model-dir   directory with the checkpoint(s), config.json,
                tokenizer.json, tokenizer_config.json
  --prompt      non-empty prompt text
  --mode        decode (default — the P4-1 per-token breakdown) or prefill
                (PF-1: per-class GPU time inside the tiled prefill chunks;
                q4g64 + fused only, the tiled path)
  --tokens      decode mode: interleaved decode forwards, half attributed +
                half production reference (default \(BenchDefaults.attributionDecodeTokens))
  --runs        prefill mode: interleaved prefills of the prompt, half
                attributed + half production (default \(BenchDefaults.prefillAttributionRuns))
  --prefill-chunk  prefill mode: tiled chunk size C (default \(GPUModel.defaultPrefillChunkSize))
  --prefill-attention  prefill mode: the chunk's causal SDPA kernel —
                query-tiled (default; PF-2) or per-position (the PF-1
                kernel, the A/B arm)
  --weights     q4g64 (default — the Phase 3+ performance path) or bf16
  --residency   mmap (default) or wired (heap copy)
  --kernels     fused (default on q4g64 — the P4-4 "after" breakdown) or
                naive (the pre-fusion "before" breakdown); fused needs
                q4g64 (the bf16 backend is permanently naive, spec D4)
"""

private func printStderr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func usageError(_ message: String) -> Int32 {
    printStderr("error: \(message)")
    printStderr(attributeUsage)
    return 2
}

func runAttributeCommand(_ arguments: [String]) async -> Int32 {
    var modelDir: String?
    var prompt: String?
    var decodeTokens = BenchDefaults.attributionDecodeTokens
    var mode = "decode"
    var runs = BenchDefaults.prefillAttributionRuns
    var prefillChunk: Int?
    var prefillAttention: GPUModel.PrefillAttention?
    var weightsFormat = WeightsFormat.q4g64
    var residency = WeightsResidency.mmap
    var kernels: GPUModel.KernelPath?

    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard index + 1 < arguments.count else {
            return usageError("flag '\(flag)' needs a value")
        }
        let value = arguments[index + 1]
        switch flag {
        case "--model-dir": modelDir = value
        case "--prompt": prompt = value
        case "--tokens":
            guard let parsed = Int(value), parsed >= 1 else {
                return usageError("--tokens must be a positive integer, got '\(value)'")
            }
            decodeTokens = parsed
        case "--mode":
            guard value == "decode" || value == "prefill" else {
                return usageError("--mode must be 'decode' or 'prefill', got '\(value)'")
            }
            mode = value
        case "--runs":
            guard let parsed = Int(value), parsed >= 2 else {
                return usageError("--runs must be an integer >= 2, got '\(value)'")
            }
            runs = parsed
        case "--prefill-chunk":
            guard let parsed = Int(value), parsed >= 1 else {
                return usageError("--prefill-chunk must be a positive integer, got '\(value)'")
            }
            prefillChunk = parsed
        case "--prefill-attention":
            guard let parsed = GPUModel.PrefillAttention(rawValue: value) else {
                return usageError(
                    "--prefill-attention must be 'query-tiled' or 'per-position', "
                    + "got '\(value)'")
            }
            prefillAttention = parsed
        case "--weights":
            guard let parsed = WeightsFormat(rawValue: value) else {
                return usageError("--weights must be 'bf16' or 'q4g64', got '\(value)'")
            }
            weightsFormat = parsed
        case "--residency":
            switch value {
            case "mmap": residency = .mmap
            case "wired": residency = .wiredCopy
            default:
                return usageError("--residency must be 'mmap' or 'wired', got '\(value)'")
            }
        case "--kernels":
            guard let parsed = GPUModel.KernelPath(rawValue: value) else {
                return usageError("--kernels must be 'naive' or 'fused', got '\(value)'")
            }
            kernels = parsed
        default:
            return usageError("unknown flag '\(flag)'")
        }
        index += 2
    }
    guard let modelDir else { return usageError("--model-dir is required") }
    guard let prompt, !prompt.isEmpty else {
        return usageError("--prompt is required and must not be empty")
    }
    if kernels == .fused, weightsFormat == .bf16 {
        return usageError(
            "--kernels fused needs --weights q4g64 (the bf16 backend runs "
            + "the naive structure permanently, phase-4.md D4)")
    }
    if mode == "prefill" {
        // PF-1: the tiled prefill exists on the packed + fused pipeline only.
        if weightsFormat == .bf16 {
            return usageError(
                "--mode prefill needs --weights q4g64 (tiled prefill is "
                + "packed-pipeline only, phase-5.md D5)")
        }
        if kernels == .naive {
            return usageError(
                "--mode prefill needs --kernels fused (the naive arm runs "
                + "sequential prefill only, phase-5.md D5)")
        }
    } else if prefillChunk != nil || prefillAttention != nil {
        return usageError(
            "--prefill-chunk/--prefill-attention apply to --mode prefill only")
    }

    do {
        let directory = try ModelDirectory(
            validating: URL(fileURLWithPath: modelDir, isDirectory: true))
        let config = try ModelConfig.load(path: directory.configURL.path)
        let contextLimit = min(4096, config.maxPositionEmbeddings)
        let metal = try MetalContext()

        let model: GPUModel
        let loadStart = Date()
        switch weightsFormat {
        case .bf16:
            printStderr("loading checkpoint \(directory.checkpointURL.lastPathComponent) ...")
            let checkpoint = try SafetensorsFile(path: directory.checkpointURL.path)
            model = try GPUModel(
                checkpoint: checkpoint, config: config, context: metal,
                residency: residency, maxContext: contextLimit)
        case .q4g64:
            let packedURL = try directory.requirePackedCheckpoint()
            printStderr("loading packed checkpoint \(packedURL.lastPathComponent) ...")
            let packed = try PackedCheckpoint(path: packedURL.path)
            // Decode attribution keeps sequential prefill (its prompt runs
            // through production `step`s — the P4-1 shape); prefill
            // attribution needs the tiled path (the engine default).
            model = try GPUModel(
                packed: packed, config: config, context: metal,
                residency: residency, maxContext: contextLimit,
                kernelPath: kernels ?? .fused,
                prefillPath: mode == "prefill" ? .tiled : .sequential,
                prefillChunkSize: prefillChunk,
                prefillAttention: prefillAttention)
        }
        let tokenizer = try await TextTokenizer(modelFolder: directory.directoryURL)
        printStderr(String(
            format: "loaded in %.1fs (weights %@, residency %@, kernels %@, prefill %@%@)",
            Date().timeIntervalSince(loadStart), weightsFormat.rawValue,
            residency.rawValue, model.kernelPath.rawValue,
            model.prefillPath.rawValue,
            model.prefillPath == .tiled
                ? ", attention \(model.prefillAttention.rawValue)" : ""))

        let promptIds = tokenizer.encode(prompt)
        let eosTokenIds = try directory.stopTokenIds(
            config: config, tokenizerEOSTokenId: tokenizer.eosTokenId)
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate]

        if mode == "prefill" {
            printStderr(
                "prefill attribution run: \(promptIds.count) prompt tokens × "
                + "\(runs) interleaved prefills, C=\(model.prefillChunkSize) "
                + "(DIAGNOSTIC)…")
            let result = try PrefillAttributionRunner(
                gpuModel: model, maxContext: contextLimit
            ).run(promptIds: promptIds, runs: runs)
            print(result.exportText(
                dateStamp: formatter.string(from: Date()),
                deviceLabel: metal.device.name,
                osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
                residency: residency))
            return 0
        }

        printStderr(
            "attribution run: \(promptIds.count) prompt tokens + "
            + "\(decodeTokens) interleaved decode forwards (DIAGNOSTIC)…")
        let runner = AttributionRunner(
            gpuModel: model, maxContext: contextLimit, eosTokenIds: eosTokenIds)
        let result = try runner.run(promptIds: promptIds, decodeTokens: decodeTokens)

        print(result.exportText(
            dateStamp: formatter.string(from: Date()),
            deviceLabel: metal.device.name,
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            residency: residency))
        return 0
    } catch {
        printStderr("attribute failed: \(error)")
        return 1
    }
}
