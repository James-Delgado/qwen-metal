import Foundation
import QwenMetalEngine

// P3-6 (phase-3.md D7): `qwen-metal-cli microbench` — the standalone
// dequant-matvec bandwidth microbench over the real packed artifact.
// P5-2 (phase-5.md D7) adds `--kernel gemm`: the tiled dequant-GEMM
// M-sweep (GB/s + GFLOPS per M). Thin: argument handling and printing
// only; the harnesses (site resolution, timing, byte accounting, spot
// check) live in QwenMetalEngine.

private let microbenchUsage = """
usage: qwen-metal-cli microbench --model-dir <dir> [--kernel matvec|gemm] \
[--m-list 8,64,512] [--residency mmap|wired] [--warmup N] [--iterations N]
  --model-dir   directory with the *-q4g64.safetensors packed artifact and
                config.json (the generate-command layout)
  --kernel      matvec (default, the P3-6 bench) or gemm (the P5-2 tiled
                dequant-GEMM M-sweep)
  --m-list      gemm only: comma-separated batch sizes (default \
\(QuantGemmMicrobench.defaultMValues.map(String.init).joined(separator: ",")))
  --residency   mmap (default) or wired (heap copy)
  --warmup      warmup iterations, discarded (default \(QuantMatvecMicrobench.defaultWarmupIterations))
  --iterations  measured iterations (default \(QuantMatvecMicrobench.defaultMeasuredIterations))
"""

private func printStderr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func usageError(_ message: String) -> Int32 {
    printStderr("error: \(message)")
    printStderr(microbenchUsage)
    return 2
}

private enum MicrobenchKernel: String {
    case matvec
    case gemm
}

func runMicrobenchCommand(_ arguments: [String]) -> Int32 {
    var modelDir: String?
    var kernel = MicrobenchKernel.matvec
    var mValues = QuantGemmMicrobench.defaultMValues
    var mListGiven = false
    var residency = WeightsResidency.mmap
    var warmup = QuantMatvecMicrobench.defaultWarmupIterations
    var iterations = QuantMatvecMicrobench.defaultMeasuredIterations

    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard index + 1 < arguments.count else {
            return usageError("flag '\(flag)' needs a value")
        }
        let value = arguments[index + 1]
        switch flag {
        case "--model-dir": modelDir = value
        case "--kernel":
            guard let parsed = MicrobenchKernel(rawValue: value) else {
                return usageError("--kernel must be 'matvec' or 'gemm', got '\(value)'")
            }
            kernel = parsed
        case "--m-list":
            let parts = value.split(separator: ",").map {
                Int($0.trimmingCharacters(in: .whitespaces))
            }
            guard !parts.isEmpty, parts.allSatisfy({ ($0 ?? 0) >= 1 }) else {
                return usageError(
                    "--m-list must be comma-separated positive integers, got '\(value)'")
            }
            mValues = parts.compactMap { $0 }
            mListGiven = true
        case "--residency":
            switch value {
            case "mmap": residency = .mmap
            case "wired": residency = .wiredCopy
            default:
                return usageError("--residency must be 'mmap' or 'wired', got '\(value)'")
            }
        case "--warmup":
            guard let parsed = Int(value), parsed >= 0 else {
                return usageError("--warmup must be a non-negative integer, got '\(value)'")
            }
            warmup = parsed
        case "--iterations":
            guard let parsed = Int(value), parsed >= 1 else {
                return usageError("--iterations must be a positive integer, got '\(value)'")
            }
            iterations = parsed
        default:
            return usageError("unknown flag '\(flag)'")
        }
        index += 2
    }
    guard let modelDir else { return usageError("--model-dir is required") }
    if mListGiven && kernel != .gemm {
        return usageError("--m-list applies only with --kernel gemm")
    }

    do {
        let directory = try ModelDirectory(
            validating: URL(fileURLWithPath: modelDir))
        let config = try ModelConfig.load(path: directory.configURL.path)
        let packed = try PackedCheckpoint(
            path: directory.requirePackedCheckpoint().path)
        let context = try MetalContext()

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate]
        let os = ProcessInfo.processInfo.operatingSystemVersionString

        switch kernel {
        case .matvec:
            printStderr("running matvec microbench (\(warmup) warmup + "
                + "\(iterations) measured, residency \(residency.rawValue))…")
            let bench = try QuantMatvecMicrobench(
                packed: packed, config: config, context: context,
                residency: residency)
            let result = try bench.run(
                warmupIterations: warmup, measuredIterations: iterations)
            print(result.exportText(
                dateStamp: formatter.string(from: Date()),
                deviceLabel: context.device.name,
                osVersion: os,
                residency: residency))
        case .gemm:
            printStderr("running tiled GEMM microbench (M ∈ \(mValues), "
                + "\(warmup) warmup + \(iterations) measured, residency "
                + "\(residency.rawValue))…")
            let bench = try QuantGemmMicrobench(
                packed: packed, config: config, context: context,
                residency: residency)
            let result = try bench.run(
                mValues: mValues, warmupIterations: warmup,
                measuredIterations: iterations)
            print(result.exportText(
                dateStamp: formatter.string(from: Date()),
                deviceLabel: context.device.name,
                osVersion: os,
                residency: residency))
        }
        return 0
    } catch {
        printStderr("microbench failed: \(error)")
        return 1
    }
}
