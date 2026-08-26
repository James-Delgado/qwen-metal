import Foundation
import QwenMetalEngine

// P3-1: `qwen-metal-cli pack` — offline q4g64 packing (phase-3.md D2).
// Thin: argument handling and printing only; the packer, schema, and all
// validation live in QwenMetalEngine.

private let packUsage = """
usage: qwen-metal-cli pack --input <model.safetensors> --output <model-q4g64.safetensors>
  --input   consolidated single-file bf16 checkpoint (source_revision metadata required)
  --output  destination for the q4g64 packed checkpoint (overwritten if present)
"""

private func printStderr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func usageError(_ message: String) -> Int32 {
    printStderr("error: \(message)")
    printStderr(packUsage)
    return 2
}

func runPackCommand(_ arguments: [String]) -> Int32 {
    var input: String?
    var output: String?

    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard index + 1 < arguments.count else {
            return usageError("flag '\(flag)' needs a value")
        }
        let value = arguments[index + 1]
        switch flag {
        case "--input": input = value
        case "--output": output = value
        default:
            return usageError("unknown flag '\(flag)'")
        }
        index += 2
    }
    guard let input else { return usageError("--input is required") }
    guard let output else { return usageError("--output is required") }

    do {
        let start = Date()
        let summary = try Q4Packer.pack(inputPath: input, outputPath: output) {
            name, current, total in
            print("[\(current)/\(total)] \(name)")
        }
        let elapsed = Date().timeIntervalSince(start)
        print("packed \(summary.packedMatrices) matrices "
            + "(+\(summary.passthroughTensors) bf16 pass-through norms) "
            + "from source_revision \(summary.sourceRevision)")
        if summary.tiedLmHeadOmitted {
            print("tied lm_head.weight was byte-identical to the embedding "
                + "and is stored once (schema D1)")
        }
        print(String(format: "output: %@ (%.3f GB) in %.1f s",
                     output, Double(summary.outputByteCount) / 1e9, elapsed))
        return 0
    } catch {
        printStderr("error: \(error)")
        return 1
    }
}
