import Foundation
import Metal

/// One M-point of the P5-2 GEMM sweep: all 197 packed matrices dispatched
/// through the tiled dequant-GEMM kernel at batch M, timed per iteration in
/// one command buffer (the P3-6 aggregate protocol).
public struct QuantGemmSweepPoint: Sendable {
    public let m: Int
    /// Packed bytes (q + scales + biases) one iteration's dispatches read —
    /// M-independent: every threadgroup row reads the whole matrix once.
    public let totalPackedBytes: Int
    /// FLOP accounting for the GFLOPS report: 2·M·Σ(outDim·inDim) over the
    /// sweep's sites (multiply + add per weight element per batch row).
    public let totalFlops: Double
    /// Measured at the dispatch call sites (DispatchCounter, P2-5: measured,
    /// never derived) — 197 at the pinned dims.
    public let dispatchesPerIteration: Int
    public let warmupTimings: [DispatchTiming]
    public let measuredTimings: [DispatchTiming]
    /// Spot-check record for this M (passed — a failing check throws).
    public let spotCheckSite: String
    public let spotCheckMaxAbsDelta: Float
    public let spotCheckTolerance: Float

    /// Effective weight-stream rate per measured iteration (the D7 pinned
    /// definition, GB = 10⁹ bytes, GPU-timestamp basis; wall rides alongside
    /// in `measuredTimings` per hard rule 7).
    public var effectiveGBps: [Double] {
        measuredTimings.map { Double(totalPackedBytes) / $0.gpuDuration / 1e9 }
    }

    /// Compute throughput per measured iteration — the phase's REPORTED
    /// compute denominator (never gated; feeds the Phase 6 roofline).
    public var measuredGFlops: [Double] {
        measuredTimings.map { totalFlops / $0.gpuDuration / 1e9 }
    }

    public var medianGBps: Double { BenchMath.median(effectiveGBps) }
    /// The on-device M=8 gate (P5-5) consumes the BEST across the D8
    /// repeats protocol; within one run this is the best iteration.
    public var bestGBps: Double { effectiveGBps.max() ?? .nan }
    public var minGBps: Double { effectiveGBps.min() ?? .nan }
    public var maxGBps: Double { effectiveGBps.max() ?? .nan }
    public var medianGFlops: Double { BenchMath.median(measuredGFlops) }
    public var bestGFlops: Double { measuredGFlops.max() ?? .nan }
}

/// Result of one GEMM microbench run: the M-sweep points in run order.
public struct QuantGemmMicrobenchResult: Sendable {
    public let points: [QuantGemmSweepPoint]

    /// Row-field export shared by the CLI and the app (P2-6 principle).
    /// Operator context is passed in — the harness never guesses it.
    public func exportText(
        dateStamp: String, deviceLabel: String, osVersion: String,
        batteryHealthNote: String = "", coldOrWarmNote: String = "",
        residency: WeightsResidency
    ) -> String {
        var lines: [String] = []
        lines.append("qwen-metal Phase 5 tiled dequant-GEMM microbench (D7, P5-2)")
        lines.append("date: \(dateStamp)")
        lines.append("device: \(deviceLabel) (\(osVersion))")
        if !batteryHealthNote.isEmpty {
            lines.append("battery health: \(batteryHealthNote)")
        }
        if !coldOrWarmNote.isEmpty {
            lines.append("cold/warm: \(coldOrWarmNote)")
        }
        if let first = points.first {
            lines.append(
                "protocol: per M, one command buffer × "
                    + "\(first.dispatchesPerIteration) real packed GEMMs "
                    + "(weights-only), residency \(residency.rawValue), "
                    + "\(first.warmupTimings.count) warmup discarded + "
                    + "\(first.measuredTimings.count) measured")
            lines.append(String(
                format: "packed bytes/iteration: %d (%.3f GB), M-independent",
                first.totalPackedBytes, Double(first.totalPackedBytes) / 1e9))
        }
        for point in points {
            lines.append(String(format: "M = %d:", point.m))
            lines.append(String(
                format: "  spot check (%@): max |Δ| %.3g within Tier K gate %.3g",
                point.spotCheckSite, point.spotCheckMaxAbsDelta,
                point.spotCheckTolerance))
            for (i, timing) in point.measuredTimings.enumerated() {
                lines.append(String(
                    format: "  iter %2d: %7.2f GB/s  %8.2f GFLOPS  "
                        + "(gpu %.4fs, wall %.4fs, overhead %.4fs)",
                    i + 1, point.effectiveGBps[i], point.measuredGFlops[i],
                    timing.gpuDuration, timing.wallDuration,
                    timing.dispatchOverhead))
            }
            lines.append(String(
                format: "  aggregate: median %.2f GB/s (best %.2f, min-max "
                    + "%.2f-%.2f), median %.2f GFLOPS (best %.2f)",
                point.medianGBps, point.bestGBps, point.minGBps, point.maxGBps,
                point.medianGFlops, point.bestGFlops))
        }
        lines.append(
            "gate: M=8 effective weight-stream ≥ 0.70 × 43.84 = 30.7 GB/s "
                + "applies to the pinned iPhone ONLY (best across the D8 "
                + "repeats protocol; P5-5). GFLOPS are reported, never gated. "
                + "Mac rows are PROVISIONAL dev-loop sanity.")
        return lines.joined(separator: "\n")
    }
}

/// P5-2B: one role's slice of the GEMM sweep (all `siteCount` matrices of
/// that role — e.g. the 28 k_proj triplets — dispatched alone in one command
/// buffer at batch M), so the aggregate can be attributed to the shapes
/// that drag it. DIAGNOSTIC: the D7 gate reads the 197-site aggregate,
/// never a per-role figure.
public struct QuantGemmRoleAttribution: Sendable {
    public let role: String
    public let outDim: Int
    public let inDim: Int
    public let siteCount: Int
    /// Packed bytes the role's `siteCount` dispatches read per iteration.
    public let packedBytes: Int
    /// 2·M·outDim·inDim·siteCount.
    public let flops: Double
    /// Measured at the dispatch call sites (== siteCount when wired right).
    public let dispatchesPerIteration: Int
    public let measuredTimings: [DispatchTiming]

    public var effectiveGBps: [Double] {
        measuredTimings.map { Double(packedBytes) / $0.gpuDuration / 1e9 }
    }
    public var measuredGFlops: [Double] {
        measuredTimings.map { flops / $0.gpuDuration / 1e9 }
    }
    public var medianGBps: Double { BenchMath.median(effectiveGBps) }
    public var bestGBps: Double { effectiveGBps.max() ?? .nan }
    public var medianGFlops: Double { BenchMath.median(measuredGFlops) }
    public var medianGpuSeconds: Double {
        BenchMath.median(measuredTimings.map(\.gpuDuration))
    }
}

/// The per-role attribution of one M-point: every role of the shared roster
/// timed alone, in roster order.
public struct QuantGemmRoleAttributionResult: Sendable {
    public let m: Int
    /// The sweep aggregate's byte total (Σ role packedBytes — pinned equal).
    public let totalPackedBytes: Int
    public let roles: [QuantGemmRoleAttribution]

    /// Σ over roles of the median per-role GPU time: the aggregate the
    /// per-role medians imply (differs from the one-buffer sweep by the
    /// per-buffer gaps + run-to-run variance — a sanity cross-check, not a
    /// figure of record).
    public var impliedAggregateGpuSeconds: Double {
        roles.reduce(0) { $0 + $1.medianGpuSeconds }
    }
    public var impliedAggregateGBps: Double {
        Double(totalPackedBytes) / impliedAggregateGpuSeconds / 1e9
    }

    public func exportText() -> String {
        var lines: [String] = []
        lines.append(String(
            format: "per-role attribution @ M = %d (DIAGNOSTIC — each role "
                + "alone in its own command buffer; the gate reads the "
                + "197-site aggregate only):", m))
        lines.append(
            "  role        shape [out, in]  x n   bytes%   median GB/s  "
                + "(best)   median GFLOPS   median ms   dispatches")
        for role in roles {
            lines.append(String(
                format: "  %-10@  [%6d, %5d] x %2d  %5.1f%%   %7.2f  (%7.2f)   %9.2f   %8.3f   %d",
                role.role, role.outDim, role.inDim, role.siteCount,
                100.0 * Double(role.packedBytes) / Double(totalPackedBytes),
                role.medianGBps, role.bestGBps, role.medianGFlops,
                role.medianGpuSeconds * 1000, role.dispatchesPerIteration))
        }
        lines.append(String(
            format: "  Σ role median GPU %.3f ms ⇒ implied aggregate %.2f GB/s "
                + "(cross-check vs the one-buffer sweep above)",
            impliedAggregateGpuSeconds * 1000, impliedAggregateGBps))
        return lines.joined(separator: "\n")
    }
}

/// P5-2 (docs/phases/phase-5.md D7): the tiled dequant-GEMM M-sweep
/// microbench — the phase's kernel-quality judgment plus the project's
/// first measured compute-throughput denominator. Per M-point, one command
/// buffer runs the SAME 197-matrix sweep as the P3-6 matvec bench (shared
/// site roster — `QuantMatvecMicrobench.resolveSites`) through
/// `QuantGemmKernel.encodeGemm`, batch M rows of deterministic fp16
/// activations against every packed matrix. All sites store fp16 (the D3
/// kernel contract; the real prefill pipeline's lm_head is the
/// last-position-only matvec, P5-3 — its presence here keeps the 197-site
/// / 0.968 GB byte protocol identical across both benches).
///
/// Metrics (pinned in the gates entry): effective weight-stream rate =
/// total packed bytes ÷ the command buffer's GPU time (gated at M=8,
/// ON-DEVICE only, ≥ 0.70 × 43.84 = 30.69 GB/s), and GFLOPS =
/// 2·M·Σ(N·K) ÷ GPU time (REPORTED at every M, never gated). Dual-timed
/// per hard rule 7. Before any M-point reports, one site's GPU output is
/// diffed against the CPU-quant oracle (sgemm over `dequantMatrix`, hard
/// rule 8) at the reused Tier K gate — a wiring bug withholds the number
/// (P0B-4 precedent).
public final class QuantGemmMicrobench {
    /// The gates-entry minimum sweep: the M=8 gate point plus the reported
    /// compute-curve points.
    public static let defaultMValues = [8, 64, 512]
    public static let defaultWarmupIterations = 2
    public static let defaultMeasuredIterations = 10

    /// FLOPs one sweep iteration performs at batch M: multiply + add per
    /// weight element per batch row (pure config arithmetic — tests pin it
    /// without an artifact).
    public static func totalFlops(config: ModelConfig, m: Int) -> Double {
        let weightElements = QuantMatvecMicrobench.siteSpecs(config: config)
            .reduce(0) { $0 + $1.outDim * $1.inDim * $1.count }
        return 2.0 * Double(m) * Double(weightElements)
    }

    private let context: MetalContext
    private let kernel: QuantGemmKernel
    private let packed: PackedCheckpoint
    private let config: ModelConfig
    private let weights: GPUWeights
    private let sites: [QuantMatvecMicrobench.Site]
    private let counter = DispatchCounter()

    public init(
        packed: PackedCheckpoint, config: ModelConfig, context: MetalContext,
        residency: WeightsResidency = .mmap
    ) throws {
        let kernel = try QuantGemmKernel(context: context)
        let weights = try GPUWeights(
            file: packed.file, context: context, residency: residency)
        self.sites = try QuantMatvecMicrobench.resolveSites(
            packed: packed, config: config, weights: weights)
        self.context = context
        self.packed = packed
        self.config = config
        self.kernel = kernel
        self.weights = weights
        kernel.dispatchCounter = counter
    }

    /// Runs the sweep: per M (in the given order) a spot check, then
    /// `warmupIterations + measuredIterations` aggregate passes. Scratch
    /// buffers are allocated per M-point and released before the next
    /// (the M=512 outputs alone are ~0.7 GB at the pinned dims).
    public func run(
        mValues: [Int] = QuantGemmMicrobench.defaultMValues,
        warmupIterations: Int = QuantGemmMicrobench.defaultWarmupIterations,
        measuredIterations: Int = QuantGemmMicrobench.defaultMeasuredIterations
    ) throws -> QuantGemmMicrobenchResult {
        guard warmupIterations >= 0, measuredIterations >= 1 else {
            throw KernelInputError.invalidIterations(
                warmup: warmupIterations, measured: measuredIterations)
        }
        guard !mValues.isEmpty, mValues.allSatisfy({ $0 >= 1 }) else {
            throw QuantKernelError.nonPositiveDimension(
                name: "mValues", value: mValues.min() ?? 0)
        }

        var points: [QuantGemmSweepPoint] = []
        for m in mValues {
            try autoreleasepool {
                points.append(try runPoint(
                    m: m, warmupIterations: warmupIterations,
                    measuredIterations: measuredIterations))
            }
        }
        return QuantGemmMicrobenchResult(points: points)
    }

    // MARK: - Internals

    /// Per-point scratch: one input buffer per distinct inDim (read-only —
    /// no hazards; deterministic fp16-exact values, a pure function of the
    /// flat index so the CPU oracle recomputes them without a copy) and one
    /// output buffer per site (distinct destinations, like the real
    /// pipeline, so hazard tracking cannot serialize independent GEMMs).
    private struct Scratch {
        let inputs: [Int: MTLBuffer]
        let outputs: [MTLBuffer]
    }

    private func makeScratch(
        m: Int, sites: [QuantMatvecMicrobench.Site]
    ) throws -> Scratch {
        var inputs: [Int: MTLBuffer] = [:]
        for inDim in Set(sites.map(\.inDim)) {
            let count = m * inDim
            guard let buffer = context.device.makeBuffer(
                length: count * 2, options: .storageModeShared) else {
                throw MetalHarnessError.bufferAllocationFailed(length: count * 2)
            }
            let pointer = buffer.contents()
                .bindMemory(to: Float16.self, capacity: count)
            for i in 0..<count {
                pointer[i] = Float16(QuantMatvecMicrobench.inputValue(at: i))
            }
            inputs[inDim] = buffer
        }
        let outputs = try sites.map { site -> MTLBuffer in
            let length = m * site.outDim * 2
            guard let buffer = context.device.makeBuffer(
                length: length, options: .storageModeShared) else {
                throw MetalHarnessError.bufferAllocationFailed(length: length)
            }
            return buffer
        }
        return Scratch(inputs: inputs, outputs: outputs)
    }

    private func encode(
        _ site: QuantMatvecMicrobench.Site, m: Int, scratch: Scratch,
        output: MTLBuffer, into encoder: MTLComputeCommandEncoder
    ) throws {
        guard let input = scratch.inputs[site.inDim] else {
            // Structurally impossible: inputs were built from the sites.
            throw QuantKernelError.bufferTooSmall(
                buffer: "input(\(site.inDim))",
                requiredBytes: m * site.inDim * 2, actualBytes: 0)
        }
        try kernel.encodeGemm(
            into: encoder,
            q: weights.buffer, qByteOffset: site.qByteOffset,
            scales: weights.buffer, scalesByteOffset: site.scalesByteOffset,
            biases: weights.buffer, biasesByteOffset: site.biasesByteOffset,
            input: input, batchM: m,
            outDim: site.outDim, inDim: site.inDim, output: output)
    }

    /// P5-2B: the per-role attribution of one M-point — every role of the
    /// roster dispatched ALONE (its `count` sites in one command buffer),
    /// `warmupIterations + measuredIterations` passes each, roster order.
    /// Diagnostic companion to `run`: same kernel, same sites, same byte
    /// accounting (Σ role bytes == the sweep's totalPackedBytes), so a
    /// role's GB/s says how that shape streams when nothing else is in
    /// flight. Scratch is per-role and released between roles.
    public func runRoleAttribution(
        m: Int,
        warmupIterations: Int = QuantGemmMicrobench.defaultWarmupIterations,
        measuredIterations: Int = QuantGemmMicrobench.defaultMeasuredIterations
    ) throws -> QuantGemmRoleAttributionResult {
        guard warmupIterations >= 0, measuredIterations >= 1 else {
            throw KernelInputError.invalidIterations(
                warmup: warmupIterations, measured: measuredIterations)
        }
        guard m >= 1 else {
            throw QuantKernelError.nonPositiveDimension(name: "m", value: m)
        }
        var roles: [QuantGemmRoleAttribution] = []
        for spec in QuantMatvecMicrobench.siteSpecs(config: config) {
            let roleSites = sites.filter { $0.role == spec.role }
            try autoreleasepool {
                let scratch = try makeScratch(m: m, sites: roleSites)
                var measured: [DispatchTiming] = []
                var dispatches = 0
                for iteration in 0..<(warmupIterations + measuredIterations) {
                    counter.reset()
                    let timing = try context.timedDispatch { encoder in
                        for (index, site) in roleSites.enumerated() {
                            try encode(
                                site, m: m, scratch: scratch,
                                output: scratch.outputs[index], into: encoder)
                        }
                    }
                    dispatches = counter.count
                    if iteration >= warmupIterations { measured.append(timing) }
                }
                roles.append(QuantGemmRoleAttribution(
                    role: spec.role, outDim: spec.outDim, inDim: spec.inDim,
                    siteCount: roleSites.count,
                    packedBytes: roleSites.reduce(0) {
                        $0 + QuantMatvecMicrobench.packedBytes(
                            outDim: $1.outDim, inDim: $1.inDim)
                    },
                    flops: 2.0 * Double(m) * Double(spec.outDim)
                        * Double(spec.inDim) * Double(roleSites.count),
                    dispatchesPerIteration: dispatches,
                    measuredTimings: measured))
            }
        }
        return QuantGemmRoleAttributionResult(
            m: m,
            totalPackedBytes: sites.reduce(0) {
                $0 + QuantMatvecMicrobench.packedBytes(
                    outDim: $1.outDim, inDim: $1.inDim)
            },
            roles: roles)
    }

    private func runPoint(
        m: Int, warmupIterations: Int, measuredIterations: Int
    ) throws -> QuantGemmSweepPoint {
        let scratch = try makeScratch(m: m, sites: sites)
        let outputs = scratch.outputs
        func encode(
            _ site: QuantMatvecMicrobench.Site, output: MTLBuffer,
            into encoder: MTLComputeCommandEncoder
        ) throws {
            try self.encode(
                site, m: m, scratch: scratch, output: output, into: encoder)
        }

        // Spot check (site 0) BEFORE any timing this point reports.
        let spotSite = sites[0]
        try context.timedDispatch { encoder in
            try encode(spotSite, output: outputs[0], into: encoder)
        }
        let spotCount = m * spotSite.outDim
        let spotPointer = outputs[0].contents()
            .bindMemory(to: Float16.self, capacity: spotCount)
        let gpu = (0..<spotCount).map { Float(spotPointer[$0]) }
        let a = (0..<(m * spotSite.inDim)).map {
            Float(Float16(QuantMatvecMicrobench.inputValue(at: $0)))
        }
        let reference = try BLAS.sgemm(
            a: a, b: try packed.dequantMatrix(spotSite.name),
            m: m, k: spotSite.inDim, n: spotSite.outDim, transposeB: true)
        let check = QuantMatvecMicrobench.spotCheckDelta(
            gpu: gpu, reference: reference)
        guard check.passed else {
            throw QuantMatvecMicrobenchError.spotCheckFailed(
                site: "\(spotSite.name) @ M=\(m)",
                maxAbsDelta: check.maxAbsDelta, tolerance: check.tolerance)
        }

        var warmupTimings: [DispatchTiming] = []
        var measuredTimings: [DispatchTiming] = []
        var dispatchesPerIteration = 0
        for iteration in 0..<(warmupIterations + measuredIterations) {
            counter.reset()
            let timing = try context.timedDispatch { encoder in
                for (index, site) in sites.enumerated() {
                    try encode(site, output: outputs[index], into: encoder)
                }
            }
            dispatchesPerIteration = counter.count
            if iteration < warmupIterations {
                warmupTimings.append(timing)
            } else {
                measuredTimings.append(timing)
            }
        }

        return QuantGemmSweepPoint(
            m: m,
            totalPackedBytes: sites.reduce(0) {
                $0 + QuantMatvecMicrobench.packedBytes(
                    outDim: $1.outDim, inDim: $1.inDim)
            },
            totalFlops: Self.totalFlops(config: config, m: m),
            dispatchesPerIteration: dispatchesPerIteration,
            warmupTimings: warmupTimings,
            measuredTimings: measuredTimings,
            spotCheckSite: "\(spotSite.name) @ M=\(m)",
            spotCheckMaxAbsDelta: check.maxAbsDelta,
            spotCheckTolerance: check.tolerance)
    }
}
