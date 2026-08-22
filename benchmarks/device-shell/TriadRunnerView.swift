// TriadRunnerView.swift — disposable device shell for the P0A-1 iPhone triad run.
//
// NOT part of any repo target. Paste into a scratch iOS App project that
// depends on the local QwenMetalEngine package (see README.md in this
// directory). Runs the P0B-4 pinned protocol (DECISIONS.md 2026-08-21):
// 3 × 384 MiB buffers = 1.125 GiB streamed/iteration, 2 warmup + 10 measured,
// sustained = median of measured GPU-time rates, wall clock recorded
// alongside (hard rule 7). Refuses to show a number if the pre-committed
// 1e-6 correctness gate fails on the sampled elements.

import SwiftUI
import QwenMetalEngine

struct TriadRunnerView: View {
    @State private var log: String = "Ready. Charge >50%, note battery health & temperature, then run."
    @State private var running = false

    var body: some View {
        VStack(spacing: 16) {
            ScrollView {
                Text(log)
                    .font(.system(.footnote, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding()
            }
            Button(running ? "Running…" : "Run triad microbench") {
                run()
            }
            .disabled(running)
            .buttonStyle(.borderedProminent)
            .padding(.bottom)
        }
    }

    private func run() {
        running = true
        log = "Running pinned protocol (1.125 GiB/iter, 2+10 iters)…\n"
        Task.detached(priority: .userInitiated) {
            let output: String
            do {
                let tolerance: Float = 1e-6  // pre-committed gate, P0B-4 entry
                let context = try MetalContext()
                let kernel = try TriadBandwidthKernel(context: context)
                let n = TriadBandwidthKernel.benchmarkElementCount
                let sampleIndices = [0, n / 2, n - 1]

                let result = try kernel.run(
                    elementCount: n,
                    warmupIterations: TriadBandwidthKernel.benchmarkWarmupIterations,
                    measuredIterations: TriadBandwidthKernel.benchmarkMeasuredIterations,
                    sampleIndices: sampleIndices
                )

                var lines: [String] = []
                lines.append("device: \(context.device.name)")
                lines.append("os: \(ProcessInfo.processInfo.operatingSystemVersionString)")

                var gateFailed = false
                for index in sampleIndices {
                    let reference = TriadBandwidthKernel.bValue(at: index)
                        + TriadBandwidthKernel.scalar * TriadBandwidthKernel.cValue(at: index)
                    let gpu = result.outputSamples[index] ?? .nan
                    if abs(gpu - reference) > tolerance {
                        lines.append("CORRECTNESS GATE FAILED at \(index): gpu \(gpu) vs cpu \(reference)")
                        gateFailed = true
                    }
                }

                if gateFailed {
                    lines.append("bandwidth figure WITHHELD (gate failure)")
                } else {
                    lines.append("correctness: \(sampleIndices.count) samples within 1e-6 gate")
                    for (i, timing) in result.measuredTimings.enumerated() {
                        lines.append(String(
                            format: "iter %2d: %7.2f GB/s (gpu %.4fs, wall %.4fs, ovh %.4fs)",
                            i + 1, result.measuredGBps[i],
                            timing.gpuDuration, timing.wallDuration, timing.dispatchOverhead
                        ))
                    }
                    lines.append(String(format: "SUSTAINED (median): %.2f GB/s", result.sustainedGBps))
                    lines.append(String(
                        format: "spread: min %.2f / max %.2f GB/s", result.minGBps, result.maxGBps
                    ))
                    lines.append("Record the sustained figure + spread in benchmarks/results.md")
                    lines.append("and DECISIONS.md (roofline denominator).")
                }
                output = lines.joined(separator: "\n")
            } catch {
                output = "FAILED: \(error)"
            }
            await MainActor.run {
                log += output
                running = false
            }
        }
    }
}

#Preview {
    TriadRunnerView()
}
