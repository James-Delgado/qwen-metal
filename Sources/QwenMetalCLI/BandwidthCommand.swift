import Foundation
import QwenMetalEngine

/// `qwen-metal-cli bandwidth` — runs the P0B-4 triad microbench at the pinned
/// official protocol (DECISIONS.md 2026-08-21: 1.125 GiB streamed/iteration,
/// 2 warmup + 10 measured, sustained = median) and prints everything a
/// benchmarks/results.md row needs. Refuses to report a number if the sampled
/// outputs fail the pre-committed correctness gate.
func runBandwidthCommand() -> Int32 {
    /// Pre-committed correctness gate — DECISIONS.md 2026-08-21 (P0B-4 entry).
    let tolerance: Float = 1e-6

    do {
        let context = try MetalContext()
        let kernel = try TriadBandwidthKernel(context: context)

        let n = TriadBandwidthKernel.benchmarkElementCount
        let sampleIndices = [0, n / 2, n - 1]
        let os = ProcessInfo.processInfo.operatingSystemVersionString

        print("triad bandwidth microbench (P0B-4, pinned protocol)")
        print("device: \(context.device.name)")
        print("os: \(os)")
        print("buffers: 3 x \(n) fp32 (\(String(format: "%.3f", Double(3 * n * 4) / Double(1 << 30))) GiB streamed/iteration)")
        print("iterations: \(TriadBandwidthKernel.benchmarkWarmupIterations) warmup (discarded) + \(TriadBandwidthKernel.benchmarkMeasuredIterations) measured")

        let result = try kernel.run(
            elementCount: n,
            warmupIterations: TriadBandwidthKernel.benchmarkWarmupIterations,
            measuredIterations: TriadBandwidthKernel.benchmarkMeasuredIterations,
            sampleIndices: sampleIndices
        )

        for index in sampleIndices {
            guard let gpu = result.outputSamples[index] else {
                FileHandle.standardError.write(Data("missing output sample at index \(index)\n".utf8))
                return 1
            }
            let reference = TriadBandwidthKernel.bValue(at: index)
                + TriadBandwidthKernel.scalar * TriadBandwidthKernel.cValue(at: index)
            guard abs(gpu - reference) <= tolerance else {
                FileHandle.standardError.write(Data(
                    "correctness gate FAILED at index \(index): gpu \(gpu) vs cpu \(reference) — bandwidth figure withheld\n".utf8
                ))
                return 1
            }
        }
        print("correctness: \(sampleIndices.count) sampled elements within pre-committed gate (max |Δ| <= 1e-6)")

        print("per-iteration GB/s (GPU time; wall recorded alongside):")
        for (i, timing) in result.measuredTimings.enumerated() {
            let gbps = result.measuredGBps[i]
            print(String(
                format: "  iter %2d: %7.2f GB/s  (gpu %.4fs, wall %.4fs, overhead %.4fs)",
                i + 1, gbps, timing.gpuDuration, timing.wallDuration, timing.dispatchOverhead
            ))
        }
        print(String(format: "sustained (median): %.2f GB/s", result.sustainedGBps))
        print(String(format: "spread: min %.2f / max %.2f GB/s", result.minGBps, result.maxGBps))
        return 0
    } catch {
        FileHandle.standardError.write(Data("bandwidth bench failed: \(error)\n".utf8))
        return 1
    }
}
