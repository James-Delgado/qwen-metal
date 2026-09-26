import Foundation
import QwenMetalEngine

/// `qwen-metal-cli bandwidth [--kernel triad|read-only]` — runs the pinned
/// streaming-bandwidth protocol (DECISIONS.md 2026-08-21 P0B-4; 2026-09-26
/// BW-1: 1.125 GiB per iteration, 2 warmup + 10 measured, sustained = median)
/// and prints everything a benchmarks/results.md row needs. Thin: the gate
/// (refuse-to-report on a failed sample) and the formatting live in the
/// engine's BandwidthMicrobench, shared with the app.
private let bandwidthUsage = """
usage: qwen-metal-cli bandwidth [--kernel triad|read-only] [--float4s-per-thread N]
  --kernel              triad (default; P0B-4, 2R+1W) or read-only (BW-1, reads only)
  --float4s-per-thread  read-only only, DIAGNOSTIC: per-thread depth in 1...64
                        (pin = 64; any other value is labeled and never a row)
"""

func runBandwidthCommand(_ arguments: [String]) -> Int32 {
    var kernel = BandwidthKernel.triad
    var float4sPerThread = ReadOnlyBandwidthKernel.float4sPerThreadPerBuffer
    var index = 0
    while index < arguments.count {
        let flag = arguments[index]
        guard index + 1 < arguments.count else {
            FileHandle.standardError.write(Data("error: flag '\(flag)' needs a value\n\(bandwidthUsage)\n".utf8))
            return 2
        }
        let value = arguments[index + 1]
        switch flag {
        case "--kernel":
            guard let parsed = BandwidthKernel(rawValue: value) else {
                FileHandle.standardError.write(Data(
                    "error: --kernel must be 'triad' or 'read-only'\n\(bandwidthUsage)\n".utf8))
                return 2
            }
            kernel = parsed
        case "--float4s-per-thread":
            guard let parsed = Int(value),
                  ReadOnlyBandwidthKernel.allowedFloat4sPerThread.contains(parsed) else {
                FileHandle.standardError.write(Data(
                    "error: --float4s-per-thread must be an integer in 1...64\n\(bandwidthUsage)\n".utf8))
                return 2
            }
            float4sPerThread = parsed
        default:
            FileHandle.standardError.write(Data("error: unknown flag '\(flag)'\n\(bandwidthUsage)\n".utf8))
            return 2
        }
        index += 2
    }
    if kernel == .triad, float4sPerThread != ReadOnlyBandwidthKernel.float4sPerThreadPerBuffer {
        FileHandle.standardError.write(Data(
            "error: --float4s-per-thread applies to --kernel read-only only\n".utf8))
        return 2
    }

    do {
        let context = try MetalContext()
        let bench = BandwidthMicrobench(context: context)
        let report = try bench.run(kernel: kernel, float4sPerThread: float4sPerThread)
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate]
        print(report.exportText(
            dateStamp: formatter.string(from: Date()),
            deviceLabel: context.device.name,
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString))
        return 0
    } catch {
        FileHandle.standardError.write(Data("bandwidth bench failed: \(error)\n".utf8))
        return 1
    }
}
