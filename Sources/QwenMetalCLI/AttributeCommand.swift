import Foundation
import QwenMetalEngine

// P4-1 (phase-4.md D1): `qwen-metal-cli attribute` — the diagnostic
// per-kernel-class GPU attribution run. Thin: argument handling and printing
// only; the interleaved harness and report formatting live in
// QwenMetalEngine (AttributionRunner / AttributionRunResult).
// DIAGNOSTIC output — never a benchmark row.

private let attributeUsage = """
usage: qwen-metal-cli attribute --model-dir <dir> --prompt "<text>" \
[--tokens N] [--weights bf16|q4g64] [--residency mmap|wired]
  --model-dir   directory with the checkpoint(s), config.json,
                tokenizer.json, tokenizer_config.json
  --prompt      non-empty prompt text
  --tokens      interleaved decode forwards, half attributed + half
                production reference (default \(BenchDefaults.attributionDecodeTokens))
  --weights     q4g64 (default — the Phase 3+ performance path) or bf16
  --residency   mmap (default) or wired (heap copy)
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
    var weightsFormat = WeightsFormat.q4g64
    var residency = WeightsResidency.mmap

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
        default:
            return usageError("unknown flag '\(flag)'")
        }
        index += 2
    }
    guard let modelDir else { return usageError("--model-dir is required") }
    guard let prompt, !prompt.isEmpty else {
        return usageError("--prompt is required and must not be empty")
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
            model = try GPUModel(
                packed: packed, config: config, context: metal,
                residency: residency, maxContext: contextLimit)
        }
        let tokenizer = try await TextTokenizer(modelFolder: directory.directoryURL)
        printStderr(String(
            format: "loaded in %.1fs (weights %@, residency %@)",
            Date().timeIntervalSince(loadStart), weightsFormat.rawValue,
            residency.rawValue))

        let promptIds = tokenizer.encode(prompt)
        let eosTokenIds = try directory.stopTokenIds(
            config: config, tokenizerEOSTokenId: tokenizer.eosTokenId)

        printStderr(
            "attribution run: \(promptIds.count) prompt tokens + "
            + "\(decodeTokens) interleaved decode forwards (DIAGNOSTIC)…")
        let runner = AttributionRunner(
            gpuModel: model, maxContext: contextLimit, eosTokenIds: eosTokenIds)
        let result = try runner.run(promptIds: promptIds, decodeTokens: decodeTokens)

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate]
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
