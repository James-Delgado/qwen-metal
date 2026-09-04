import Foundation
import QwenMetalEngine

// Phase 1 exit CLI, extended by P2-4 with `--backend gpu`: `qwen-metal-cli
// generate` (phase-0-1.md exit criteria; phase-2.md spec edge cases 9-10).
// Thin: argument handling and printing only — model, tokenizer, stop set,
// and decode logic all live in QwenMetalEngine.

/// Context cap for both backends: the CPU reference's full-attention
/// re-forward is quadratic in sequence length, and the GPU backend's KV cache
/// is preallocated at this size (phase-2.md D3 pins 4096); the model's own
/// max_position_embeddings still wins if smaller.
private let contextCap = 4096

private let generateUsage = """
usage: qwen-metal-cli generate --model-dir <dir> --prompt "<text>" [--max-tokens N] [--backend cpu|gpu] [--weights bf16|q4g64]
  --model-dir   directory with exactly one .safetensors checkpoint,
                config.json, tokenizer.json, tokenizer_config.json
  --prompt      non-empty prompt text
  --max-tokens  max new tokens to generate (default 64)
  --backend     cpu (fp32 reference, default) or gpu (Metal fp16 + KV cache)
  --weights     bf16 (default) or q4g64 (the packed 4-bit artifact — needs a
                *-q4g64.safetensors file beside the checkpoint)
"""

private func printStderr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func usageError(_ message: String) -> Int32 {
    printStderr("error: \(message)")
    printStderr(generateUsage)
    return 2
}

private enum Backend: String {
    case cpu
    case gpu
}

func runGenerateCommand(_ arguments: [String]) async -> Int32 {
    var modelDir: String?
    var prompt: String?
    var maxTokens = 64
    var backend = Backend.cpu
    var weightsFormat = WeightsFormat.bf16

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
        case "--max-tokens":
            guard let parsed = Int(value), parsed >= 1 else {
                return usageError("--max-tokens must be a positive integer, got '\(value)'")
            }
            maxTokens = parsed
        case "--backend":
            guard let parsed = Backend(rawValue: value) else {
                return usageError("--backend must be 'cpu' or 'gpu', got '\(value)'")
            }
            backend = parsed
        case "--weights":
            guard let parsed = WeightsFormat(rawValue: value) else {
                return usageError("--weights must be 'bf16' or 'q4g64', got '\(value)'")
            }
            weightsFormat = parsed
        default:
            return usageError("unknown flag '\(flag)'")
        }
        index += 2
    }

    guard let modelDir else { return usageError("--model-dir is required") }
    guard let prompt else { return usageError("--prompt is required") }
    guard !prompt.isEmpty else { return usageError("--prompt must not be empty") }

    do {
        let directory = try ModelDirectory(
            validating: URL(fileURLWithPath: modelDir, isDirectory: true))
        let config = try ModelConfig.load(path: directory.configURL.path)
        let contextLimit = min(contextCap, config.maxPositionEmbeddings)

        // P3-5: q4g64 resolves the packed artifact (clear missing-file error
        // from requirePackedCheckpoint); bf16 keeps the Phase 2 path and its
        // byte-stable output.
        let model: any NextTokenLogitsSource
        var gpuModel: GPUModel?
        let loadStart: Date
        switch weightsFormat {
        case .bf16:
            printStderr("loading checkpoint \(directory.checkpointURL.lastPathComponent) ...")
            loadStart = Date()
            let checkpoint = try SafetensorsFile(path: directory.checkpointURL.path)
            switch backend {
            case .cpu:
                model = try QwenModel(
                    checkpoint: checkpoint, config: config, maxSequenceLength: contextLimit)
            case .gpu:
                // A machine without Metal fails here with MetalHarnessError
                // .noDevice — a clear error, not a crash (spec edge case 10).
                let metal = try MetalContext()
                let gpu = try GPUModel(
                    checkpoint: checkpoint, config: config, context: metal,
                    maxContext: contextLimit)
                model = gpu
                gpuModel = gpu
            }
        case .q4g64:
            let packedURL = try directory.requirePackedCheckpoint()
            printStderr("loading packed checkpoint \(packedURL.lastPathComponent) ...")
            loadStart = Date()
            let packed = try PackedCheckpoint(path: packedURL.path)
            switch backend {
            case .cpu:
                // The CPU-quant reference (phase-3.md D3): fp32 dequant
                // materialization through the frozen CPU model.
                model = try QwenModel(
                    weights: packed, config: config, maxSequenceLength: contextLimit)
            case .gpu:
                let metal = try MetalContext()
                let gpu = try GPUModel(
                    packed: packed, config: config, context: metal,
                    maxContext: contextLimit)
                model = gpu
                gpuModel = gpu
            }
        }
        let tokenizer = try await TextTokenizer(modelFolder: directory.directoryURL)
        printStderr(String(
            format: "loaded in %.1fs (backend: %@%@)",
            Date().timeIntervalSince(loadStart), backend.rawValue,
            weightsFormat == .bf16 ? "" : ", weights: q4g64"))

        let promptIds = tokenizer.encode(prompt)
        // Engine-owned stop set (phase-2.md D7): config.json ∪ tokenizer ∪
        // generation_config.json — {151645, 151643} on the pinned checkpoint.
        let eosTokenIds = try directory.stopTokenIds(
            config: config, tokenizerEOSTokenId: tokenizer.eosTokenId)

        // P2-5 instrumentation (gpu backend): one TokenStepRecord per
        // generated token, straight from the model's per-step dual timing +
        // dispatch count. Engine-side aggregation (DecodeTimingCollector) so
        // the Phase 2 app reports the same numbers.
        var collector = DecodeTimingCollector()
        let onStep: ((Int, [Float], Int) -> Void)? = gpuModel.map { gpu in
            { _, _, _ in
                if let timing = gpu.lastStepTiming,
                   let dispatches = gpu.lastStepDispatchCount {
                    collector.append(TokenStepRecord(
                        timing: timing, dispatchCount: dispatches))
                }
            }
        }

        let decodeStart = Date()
        let generated = try DecodeLoop(model: model, maxContext: contextLimit)
            .generate(
                promptIds: promptIds, maxNewTokens: maxTokens,
                eosTokenIds: eosTokenIds, onStep: onStep)
        let decodeSeconds = Date().timeIntervalSince(decodeStart)

        print(tokenizer.decode(generated, skipSpecialTokens: true))
        let backendNote: String
        switch (backend, weightsFormat) {
        case (.cpu, .bf16): backendNote = "CPU reference — no KV cache"
        case (.cpu, .q4g64): backendNote = "CPU-quant reference — no KV cache"
        case (.gpu, .bf16): backendNote = "GPU fp16 + KV cache, naive kernels"
        case (.gpu, .q4g64): backendNote = "GPU q4g64 + KV cache, fused dequant kernels"
        }
        printStderr(String(
            format: "%d prompt tokens, %d generated in %.1fs (%.2f tok/s, %@)",
            promptIds.count, generated.count, decodeSeconds,
            Double(generated.count) / max(decodeSeconds, 1e-9), backendNote))

        // P2-5 instrumentation block (gpu backend only): medians + the
        // wall−GPU dispatch-overhead metric (hard rule 7) and the canonical
        // 128–512 window rate (benchmark protocol pin).
        if let summary = collector.summary() {
            let dispatches = summary.minDispatchCount == summary.maxDispatchCount
                ? "\(summary.minDispatchCount)"
                : "UNSTABLE \(summary.minDispatchCount)-\(summary.maxDispatchCount)"
            printStderr(String(
                format: "per-token (%d tokens): median GPU %.2f ms, median wall "
                    + "%.2f ms, median wall-GPU %.3f ms, %@ dispatches/token",
                summary.tokenCount, summary.medianGPUSeconds * 1000,
                summary.medianWallSeconds * 1000,
                summary.medianOverheadSeconds * 1000, dispatches))
            let windowed = collector.canonicalWindowTokensPerSecond().map {
                String(format: "%.2f tok/s", $0)
            } ?? String(
                format: "n/a (needs >= %d generated tokens, got %d)",
                CanonicalDecodeWindow.lastToken, summary.tokenCount)
            let overall = collector.overallTokensPerSecond().map {
                String(format: "%.2f tok/s", $0)
            } ?? "n/a"
            printStderr(
                "decode rate: overall \(overall), canonical window (tokens "
                + "\(CanonicalDecodeWindow.firstToken)-"
                + "\(CanonicalDecodeWindow.lastToken)) \(windowed)")
        }
        return 0
    } catch {
        printStderr("error: \(error)")
        return 1
    }
}
