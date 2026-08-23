import Foundation
import QwenMetalEngine

// Phase 1 exit CLI: `qwen-metal-cli generate` (phase-0-1.md exit criteria).
// Thin: argument handling and printing only — model, tokenizer, and decode
// logic all live in QwenMetalEngine.

/// Phase 1 context cap: full-attention re-forward decode is quadratic in
/// sequence length, so the CLI bounds context at 4K (the spec's CLI edge
/// case); the model's own max_position_embeddings still wins if smaller.
private let phase1ContextCap = 4096

private let generateUsage = """
usage: qwen-metal-cli generate --model-dir <dir> --prompt "<text>" [--max-tokens N]
  --model-dir   directory with exactly one .safetensors checkpoint,
                config.json, tokenizer.json, tokenizer_config.json
  --prompt      non-empty prompt text
  --max-tokens  max new tokens to generate (default 64)
"""

private func printStderr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func usageError(_ message: String) -> Int32 {
    printStderr("error: \(message)")
    printStderr(generateUsage)
    return 2
}

func runGenerateCommand(_ arguments: [String]) async -> Int32 {
    var modelDir: String?
    var prompt: String?
    var maxTokens = 64

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
        let contextLimit = min(phase1ContextCap, config.maxPositionEmbeddings)

        printStderr("loading checkpoint \(directory.checkpointURL.lastPathComponent) ...")
        let loadStart = Date()
        let checkpoint = try SafetensorsFile(path: directory.checkpointURL.path)
        let model = try QwenModel(
            checkpoint: checkpoint, config: config, maxSequenceLength: contextLimit)
        let tokenizer = try await TextTokenizer(modelFolder: directory.directoryURL)
        printStderr(String(
            format: "loaded in %.1fs", Date().timeIntervalSince(loadStart)))

        let promptIds = tokenizer.encode(prompt)
        var eosTokenIds = Set(config.eosTokenIds)
        if let eos = tokenizer.eosTokenId { eosTokenIds.insert(eos) }

        let decodeStart = Date()
        let generated = try DecodeLoop(model: model, maxContext: contextLimit)
            .generate(
                promptIds: promptIds, maxNewTokens: maxTokens,
                eosTokenIds: eosTokenIds)
        let decodeSeconds = Date().timeIntervalSince(decodeStart)

        print(tokenizer.decode(generated, skipSpecialTokens: true))
        printStderr(String(
            format: "%d prompt tokens, %d generated in %.1fs (%.2f tok/s, "
                + "CPU reference — no KV cache)",
            promptIds.count, generated.count, decodeSeconds,
            Double(generated.count) / max(decodeSeconds, 1e-9)))
        return 0
    } catch {
        printStderr("error: \(error)")
        return 1
    }
}
