import Foundation

/// Which streaming-bandwidth kernel to run. Raw values are the CLI/app
/// vocabulary (`bandwidth --kernel triad|read-only`).
public enum BandwidthKernel: String, CaseIterable, Sendable, Identifiable {
    /// P0B-4: STREAM triad a = b + s·c, 2 reads + 1 write, the roofline
    /// denominator of record (PLAN.md invariant 1).
    case triad
    /// BW-1: read-only streaming reduce over the same three buffers — the
    /// read-dominated (decode-shaped) bound, cited alongside the triad.
    case readOnly = "read-only"

    public var id: String { rawValue }
}

/// Errors from the bandwidth microbench harness.
public enum BandwidthMicrobenchError: Error, CustomStringConvertible, Equatable {
    /// The pre-committed correctness gate failed on a sampled output — the
    /// figure is withheld (P0B-4 refuse-to-report precedent).
    case gateFailed(kernel: BandwidthKernel, sampleIndex: Int, maxAbsDelta: Float, tolerance: Float)

    public var description: String {
        switch self {
        case .gateFailed(let kernel, let sampleIndex, let maxAbsDelta, let tolerance):
            return "\(kernel.rawValue) bandwidth correctness gate FAILED at sample "
                + "\(sampleIndex): max |Δ| \(maxAbsDelta) exceeds the pre-committed "
                + "gate \(tolerance) — bandwidth figure WITHHELD"
        }
    }
}

/// One bandwidth microbench run that passed its gate: the figures a
/// benchmarks/results.md row needs, plus the export text both the CLI and
/// the app print.
public struct BandwidthMicrobenchReport: Sendable {
    public let kernel: BandwidthKernel
    public let deviceName: String
    public let elementCount: Int
    /// The GB/s numerator: bytes moved (triad, 2R+1W) or bytes READ (read-only).
    public let bytesCountedPerIteration: Int
    /// Bytes written per iteration; part of the numerator only for the triad.
    public let bytesWrittenPerIteration: Int
    public let writesCounted: Bool
    /// Read-only only: float4 per thread per buffer. Pinned 64; anything else
    /// is a DIAGNOSTIC geometry and the export says so. nil for the triad.
    public let float4sPerThread: Int?
    public let warmupCount: Int
    public let measuredTimings: [DispatchTiming]
    public let gateSampleCount: Int
    public let gateMaxAbsDelta: Float
    public let gateTolerance: Float

    public var measuredGBps: [Double] {
        measuredTimings.map { Double(bytesCountedPerIteration) / $0.gpuDuration / 1e9 }
    }
    public var sustainedGBps: Double { BenchMath.median(measuredGBps) }
    public var minGBps: Double { measuredGBps.min() ?? .nan }
    public var maxGBps: Double { measuredGBps.max() ?? .nan }

    /// True when the run used a non-pinned geometry — never a row.
    public var isDiagnosticGeometry: Bool {
        guard let depth = float4sPerThread else { return false }
        return depth != ReadOnlyBandwidthKernel.float4sPerThreadPerBuffer
    }

    /// Row export (engine-side formatting; callers supply operator context).
    public func exportText(
        dateStamp: String, deviceLabel: String, osVersion: String,
        batteryHealthNote: String = "", coldOrWarmNote: String = ""
    ) -> String {
        var lines: [String] = []
        switch kernel {
        case .triad:
            lines.append("qwen-metal bandwidth microbench — STREAM triad (P0B-4)")
        case .readOnly:
            lines.append("qwen-metal bandwidth microbench — read-only streaming reduce (BW-1)")
        }
        if isDiagnosticGeometry, let depth = float4sPerThread {
            lines.append("DIAGNOSTIC — non-pinned geometry (\(depth) float4/thread/buffer; "
                + "pin is \(ReadOnlyBandwidthKernel.float4sPerThreadPerBuffer)) — NEVER a row")
        }
        lines.append("date: \(dateStamp)")
        lines.append("device: \(deviceLabel) (\(osVersion))")
        lines.append("gpu: \(deviceName)")
        if !batteryHealthNote.isEmpty { lines.append("battery health: \(batteryHealthNote)") }
        if !coldOrWarmNote.isEmpty { lines.append("cold/warm: \(coldOrWarmNote)") }
        let gib = Double(bytesCountedPerIteration) / Double(1 << 30)
        lines.append(String(
            format: "protocol: 3 × %d fp32 buffers (%.3f GiB %@/iteration), %d warmup discarded + "
                + "%d measured, sustained = median (GPU timestamps; wall alongside)",
            elementCount, gib, kernel == .triad ? "moved" : "read",
            warmupCount, measuredTimings.count))
        switch kernel {
        case .triad:
            lines.append("bytes moved/iteration (2R+1W): \(bytesCountedPerIteration)")
        case .readOnly:
            if let depth = float4sPerThread {
                lines.append("geometry: \(depth) float4/thread/buffer, 256-thread "
                    + "threadgroups, one float4 partial per simdgroup")
            }
            lines.append("bytes read/iteration: \(bytesCountedPerIteration)")
            lines.append(String(
                format: "bytes written/iteration: %d (%.3f%% of read; excluded from GB/s)",
                bytesWrittenPerIteration,
                100 * Double(bytesWrittenPerIteration) / Double(bytesCountedPerIteration)))
        }
        lines.append(String(
            format: "correctness: %d sampled %@ within pre-committed gate (max |Δ| %.3g <= %.3g)",
            gateSampleCount, kernel == .triad ? "elements" : "partials",
            gateMaxAbsDelta, gateTolerance))
        lines.append("per-iteration GB/s (GPU time; wall recorded alongside):")
        for (i, timing) in measuredTimings.enumerated() {
            lines.append(String(
                format: "  iter %2d: %7.2f GB/s  (gpu %.4fs, wall %.4fs, overhead %.4fs)",
                i + 1, measuredGBps[i], timing.gpuDuration, timing.wallDuration,
                timing.dispatchOverhead))
        }
        lines.append(String(format: "sustained (median): %.2f GB/s", sustainedGBps))
        lines.append(String(format: "spread: min %.2f / max %.2f GB/s", minGBps, maxGBps))
        return lines.joined(separator: "\n")
    }
}

/// The shared harness behind `qwen-metal-cli bandwidth` and the app's
/// microbench picker: runs either bandwidth kernel at the pinned protocol,
/// enforces the pre-committed 1e-6 gate on sampled outputs BEFORE any figure
/// exists, and formats the row export.
public struct BandwidthMicrobench {
    /// Pre-committed correctness gate — DECISIONS.md 2026-08-21 (P0B-4) and
    /// 2026-09-26 (BW-1). Never loosens.
    public static let tolerance: Float = 1e-6
    /// Pinned working set (both kernels): 1.125 GiB per iteration.
    public static let defaultElementCount = ReadOnlyBandwidthKernel.benchmarkElementCount
    public static let defaultWarmupIterations = ReadOnlyBandwidthKernel.benchmarkWarmupIterations
    public static let defaultMeasuredIterations = ReadOnlyBandwidthKernel.benchmarkMeasuredIterations

    /// One sampled output and its CPU reference (one entry per component).
    public struct GateSample: Sendable {
        public let index: Int
        public let gpu: [Float]
        public let reference: [Float]
        public init(index: Int, gpu: [Float], reference: [Float]) {
            self.index = index
            self.gpu = gpu
            self.reference = reference
        }
    }

    private let context: MetalContext

    public init(context: MetalContext) {
        self.context = context
    }

    /// Pure gate evaluation: returns the max |Δ| over every component of
    /// every sample, or throws `gateFailed` naming the first offending
    /// sample. A NaN or a component-count mismatch fails (never passes by
    /// accident).
    @discardableResult
    public static func checkGate(
        kernel: BandwidthKernel, samples: [GateSample], tolerance: Float
    ) throws -> Float {
        var maxAbsDelta: Float = 0
        for sample in samples {
            guard sample.gpu.count == sample.reference.count else {
                throw BandwidthMicrobenchError.gateFailed(
                    kernel: kernel, sampleIndex: sample.index,
                    maxAbsDelta: .infinity, tolerance: tolerance)
            }
            var sampleMax: Float = 0
            for (gpu, reference) in zip(sample.gpu, sample.reference) {
                let delta = abs(gpu - reference)
                if !(delta <= tolerance) {  // NaN fails
                    throw BandwidthMicrobenchError.gateFailed(
                        kernel: kernel, sampleIndex: sample.index,
                        maxAbsDelta: delta, tolerance: tolerance)
                }
                sampleMax = max(sampleMax, delta)
            }
            maxAbsDelta = max(maxAbsDelta, sampleMax)
        }
        return maxAbsDelta
    }

    /// Runs `kernel` and returns the gated report. Defaults are the pinned
    /// protocol; smaller `elementCount`s exist for tests only, and a
    /// non-pinned `float4sPerThread` (read-only only) is the diagnostic sweep.
    public func run(
        kernel: BandwidthKernel,
        elementCount: Int = BandwidthMicrobench.defaultElementCount,
        warmupIterations: Int = BandwidthMicrobench.defaultWarmupIterations,
        measuredIterations: Int = BandwidthMicrobench.defaultMeasuredIterations,
        float4sPerThread: Int = ReadOnlyBandwidthKernel.float4sPerThreadPerBuffer
    ) throws -> BandwidthMicrobenchReport {
        switch kernel {
        case .triad:
            let triad = try TriadBandwidthKernel(context: context)
            let sampleIndices = [0, elementCount / 2, elementCount - 1]
            let result = try triad.run(
                elementCount: elementCount, warmupIterations: warmupIterations,
                measuredIterations: measuredIterations, sampleIndices: sampleIndices)
            let samples = sampleIndices.map { index in
                GateSample(
                    index: index,
                    gpu: [result.outputSamples[index] ?? .nan],
                    reference: [TriadBandwidthKernel.bValue(at: index)
                        + TriadBandwidthKernel.scalar * TriadBandwidthKernel.cValue(at: index)])
            }
            let maxAbsDelta = try Self.checkGate(
                kernel: kernel, samples: samples, tolerance: Self.tolerance)
            return BandwidthMicrobenchReport(
                kernel: kernel, deviceName: context.device.name,
                elementCount: elementCount,
                bytesCountedPerIteration: result.bytesPerIteration,
                bytesWrittenPerIteration: elementCount * MemoryLayout<Float>.stride,
                writesCounted: true,
                float4sPerThread: nil,
                warmupCount: result.warmupTimings.count,
                measuredTimings: result.measuredTimings,
                gateSampleCount: samples.count, gateMaxAbsDelta: maxAbsDelta,
                gateTolerance: Self.tolerance)
        case .readOnly:
            let readOnly = try ReadOnlyBandwidthKernel(
                context: context, float4sPerThread: float4sPerThread)
            let partials = ReadOnlyBandwidthKernel.partialCount(
                elementCount: elementCount, float4sPerThread: float4sPerThread)
            let sampleIndices = partials > 0 ? [0, partials / 2, partials - 1] : []
            let result = try readOnly.run(
                elementCount: elementCount, warmupIterations: warmupIterations,
                measuredIterations: measuredIterations, samplePartialIndices: sampleIndices)
            let samples = sampleIndices.map { index -> GateSample in
                let gpu = result.partialSamples[index] ?? SIMD4<Float>(repeating: .nan)
                let cpu = ReadOnlyBandwidthKernel.cpuPartial(
                    index: index, elementCount: elementCount, float4sPerThread: float4sPerThread)
                return GateSample(
                    index: index, gpu: (0..<4).map { gpu[$0] }, reference: (0..<4).map { cpu[$0] })
            }
            let maxAbsDelta = try Self.checkGate(
                kernel: kernel, samples: samples, tolerance: Self.tolerance)
            return BandwidthMicrobenchReport(
                kernel: kernel, deviceName: context.device.name,
                elementCount: elementCount,
                bytesCountedPerIteration: result.bytesReadPerIteration,
                bytesWrittenPerIteration: result.bytesWrittenPerIteration,
                writesCounted: false,
                float4sPerThread: result.float4sPerThread,
                warmupCount: result.warmupTimings.count,
                measuredTimings: result.measuredTimings,
                gateSampleCount: samples.count, gateMaxAbsDelta: maxAbsDelta,
                gateTolerance: Self.tolerance)
        }
    }
}
