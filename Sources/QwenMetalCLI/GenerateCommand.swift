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
usage: qwen-metal-cli generate --model-dir <dir> --prompt "<text>" [--max-tokens N] [--backend cpu|gpu]
  --model-dir   directory with exactly one .safetensors checkpoint,
                config.json, tokenizer.json, tokenizer_config.json
  --prompt      non-empty prompt text
  --max-tokens  max new tokens to generate (default 64)
  --backend     cpu (fp32 reference, default) or gpu (Metal fp16 + KV cache)
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

        printStderr("loading checkpoint \(directory.checkpointURL.lastPathComponent) ...")
        let loadStart = Date()
        let checkpoint = try SafetensorsFile(path: directory.checkpointURL.path)
        let model: any NextTokenLogitsSource
        switch backend {
        case .cpu:
            model = try QwenModel(
                checkpoint: checkpoint, config: config, maxSequenceLength: contextLimit)
        case .gpu:
            // A machine without Metal fails here with MetalHarnessError
            // .noDevice — a clear error, not a crash (spec edge case 10).
            let metal = try MetalContext()
            model = try GPUModel(
                checkpoint: checkpoint, config: config, context: metal,
                maxContext: contextLimit)
        }
        let tokenizer = try await TextTokenizer(modelFolder: directory.directoryURL)
        printStderr(String(
            format: "loaded in %.1fs (backend: %@)",
            Date().timeIntervalSince(loadStart), backend.rawValue))

        let promptIds = tokenizer.encode(prompt)
        // Engine-owned stop set (phase-2.md D7): config.json ∪ tokenizer ∪
        // generation_config.json — {151645, 151643} on the pinned checkpoint.
        let eosTokenIds = try directory.stopTokenIds(
            config: config, tokenizerEOSTokenId: tokenizer.eosTokenId)

        let decodeStart = Date()
        let generated = try DecodeLoop(model: model, maxContext: contextLimit)
            .generate(
                promptIds: promptIds, maxNewTokens: maxTokens,
                eosTokenIds: eosTokenIds)
        let decodeSeconds = Date().timeIntervalSince(decodeStart)

        print(tokenizer.decode(generated, skipSpecialTokens: true))
        let backendNote = backend == .cpu
            ? "CPU reference — no KV cache"
            : "GPU fp16 + KV cache, naive kernels"
        printStderr(String(
            format: "%d prompt tokens, %d generated in %.1fs (%.2f tok/s, %@)",
            promptIds.count, generated.count, decodeSeconds,
            Double(generated.count) / max(decodeSeconds, 1e-9), backendNote))
        return 0
    } catch {
        printStderr("error: \(error)")
        return 1
    }
}
