import Foundation
import Metal

/// Errors from the P3-6 microbench harness.
public enum QuantMatvecMicrobenchError: Error, CustomStringConvertible, Equatable {
    /// The pre-report correctness spot check (Tier K vs the CPU-quant oracle)
    /// failed — the bandwidth figure is withheld (P0B-4 precedent: a bench
    /// never reports a number its own outputs contradict).
    case spotCheckFailed(site: String, maxAbsDelta: Float, tolerance: Float)

    public var description: String {
        switch self {
        case .spotCheckFailed(let site, let maxAbsDelta, let tolerance):
            return "microbench spot check FAILED at \(site): max |Δ| "
                + "\(maxAbsDelta) exceeds the Tier K gate \(tolerance) — "
                + "bandwidth figure withheld"
        }
    }
}

/// Per-role slice of the microbench: all dispatches of one matvec shape
/// (28 per decoder-layer role, 1 for lm_head), timed in their own command
/// buffers. Reported alongside the aggregate, never gated (gates entry:
/// small per-layer matvecs individually underperform; the aggregate is the
/// roofline-relevant number).
public struct QuantMatvecShapeResult: Sendable {
    public let role: String
    public let outDim: Int
    public let inDim: Int
    public let dispatchesPerIteration: Int
    /// Packed bytes (q + scales + biases) this role's dispatches read per
    /// iteration.
    public let packedBytesPerIteration: Int
    public let warmupTimings: [DispatchTiming]
    public let measuredTimings: [DispatchTiming]

    /// Weight-stream GB/s per measured iteration (GB = 10^9 bytes,
    /// GPU-timestamp basis; wall rides alongside in `measuredTimings` per
    /// hard rule 7).
    public var measuredGBps: [Double] {
        measuredTimings.map {
            Double(packedBytesPerIteration) / $0.gpuDuration / 1e9
        }
    }

    public var medianGBps: Double { BenchMath.median(measuredGBps) }
    public var bestGBps: Double { measuredGBps.max() ?? .nan }
}

/// Result of one microbench run: aggregate dual timings over the full
/// one-command-buffer sweep, per-shape slices, byte accounting, the
/// measured dispatch count, and the spot-check record.
public struct QuantMatvecMicrobenchResult: Sendable {
    /// Total packed bytes (q + scales + biases) one aggregate iteration
    /// reads — ≈ 0.967 GB at the pinned dims (gates entry).
    public let totalPackedBytes: Int
    /// Measured at the dispatchThreads call sites (DispatchCounter, P2-5:
    /// measured, never derived) — 197 at the pinned dims.
    public let dispatchesPerIteration: Int
    public let warmupTimings: [DispatchTiming]
    public let measuredTimings: [DispatchTiming]
    public let shapes: [QuantMatvecShapeResult]
    /// Spot-check record (passed — a failing check throws instead).
    public let spotCheckSite: String
    public let spotCheckMaxAbsDelta: Float
    public let spotCheckTolerance: Float

    /// Aggregate weight-stream rate per measured iteration: total packed
    /// bytes ÷ the command buffer's GPU time (the D7 pinned definition;
    /// GB = 10^9 bytes). The single command buffer's GPU duration IS the
    /// summed kernel time — back-to-back dispatches, any inter-dispatch
    /// gap counts against us (strict direction).
    public var measuredGBps: [Double] {
        measuredTimings.map { Double(totalPackedBytes) / $0.gpuDuration / 1e9 }
    }

    public var medianGBps: Double { BenchMath.median(measuredGBps) }
    /// The on-device gate (P3-7) consumes the BEST across the D8 repeats
    /// protocol; within one run this is the best iteration.
    public var bestGBps: Double { measuredGBps.max() ?? .nan }
    public var minGBps: Double { measuredGBps.min() ?? .nan }
    public var maxGBps: Double { measuredGBps.max() ?? .nan }

    /// Row-field export shared by the CLI and the app (P2-6 principle: both
    /// surfaces print identical numbers). Operator context is passed in —
    /// the harness never guesses it.
    public func exportText(
        dateStamp: String, deviceLabel: String, osVersion: String,
        batteryHealthNote: String = "", coldOrWarmNote: String = "",
        residency: WeightsResidency
    ) -> String {
        var lines: [String] = []
        lines.append("qwen-metal Phase 3 dequant-matvec microbench (D7)")
        lines.append("date: \(dateStamp)")
        lines.append("device: \(deviceLabel) (\(osVersion))")
        if !batteryHealthNote.isEmpty {
            lines.append("battery health: \(batteryHealthNote)")
        }
        if !coldOrWarmNote.isEmpty {
            lines.append("cold/warm: \(coldOrWarmNote)")
        }
        lines.append(
            "protocol: one command buffer × \(dispatchesPerIteration) real "
                + "packed matvecs (weights-only), residency \(residency.rawValue), "
                + "\(warmupTimings.count) warmup discarded + "
                + "\(measuredTimings.count) measured")
        lines.append(String(
            format: "packed bytes/iteration: %d (%.3f GB)",
            totalPackedBytes, Double(totalPackedBytes) / 1e9))
        lines.append(String(
            format: "spot check (%@): max |Δ| %.3g within Tier K gate %.3g",
            spotCheckSite, spotCheckMaxAbsDelta, spotCheckTolerance))
        lines.append("per-iteration aggregate GB/s (GPU basis; wall alongside):")
        for (i, timing) in measuredTimings.enumerated() {
            lines.append(String(
                format: "  iter %2d: %7.2f GB/s  (gpu %.4fs, wall %.4fs, overhead %.4fs)",
                i + 1, measuredGBps[i], timing.gpuDuration,
                timing.wallDuration, timing.dispatchOverhead))
        }
        lines.append(String(
            format: "aggregate: median %.2f GB/s, best %.2f, min-max %.2f-%.2f",
            medianGBps, bestGBps, minGBps, maxGBps))
        if !shapes.isEmpty {
            lines.append("per-shape (reported, never gated):")
            for shape in shapes {
                lines.append(String(
                    format: "  %-10@ %6d×%-5d ×%-2d  median %6.2f GB/s, best %6.2f",
                    shape.role as NSString, shape.outDim, shape.inDim,
                    shape.dispatchesPerIteration, shape.medianGBps,
                    shape.bestGBps))
            }
        }
        lines.append(
            "gate: aggregate ≥ 0.70 × 43.84 = 30.7 GB/s applies to the pinned "
                + "iPhone ONLY (best across the D8 repeats protocol; P3-7). "
                + "Mac rows are PROVISIONAL dev-loop sanity.")
        return lines.joined(separator: "\n")
    }
}

/// Small shared statistics helpers for bench result types.
enum BenchMath {
    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return .nan }
        let mid = sorted.count / 2
        return sorted.count % 2 == 1
            ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }
}

/// P3-6 (docs/phases/phase-3.md D7, OV#11): the standalone dequant-matvec
/// bandwidth microbenchmark — the phase's kernel-quality judgment,
/// independent of whole-model behavior. One command buffer runs exactly one
/// token's worth of REAL packed-weight matvecs (28 layers × {q,k,v,o,gate,
/// up,down} + lm_head = 197 dispatches at the pinned dims; no attention/
/// norm/elementwise, no KV — weights-only by construction) through the
/// P3-4 `QuantKernels.encodeMatvec` exactly as the decode pipeline binds
/// them (fp16-store projections, fp32-store lm_head, the tied embedding
/// triplet consumed as [out, in] with no transpose).
///
/// Metric (pinned in the gates entry): aggregate weight-stream rate =
/// total packed bytes (q + scales + biases) ÷ the command buffer's GPU
/// time, dual-timed per hard rule 7. Per-shape rates are measured in
/// separate per-role command buffers and reported alongside, never gated.
/// The 0.70 × 43.84 GB/s gate applies to the ON-DEVICE run only (P3-7,
/// James); Mac runs are PROVISIONAL sanity.
///
/// Inputs are synthetic deterministic fp16 activations (weight traffic is
/// the metric; inputs/outputs are KBs against ~0.97 GB of weights). Each
/// site writes its own output buffer so Metal's hazard tracking introduces
/// no false serialization the real pipeline doesn't have. Before any figure
/// is reported, one site's GPU output is diffed against the CPU-quant
/// oracle (sgemm over `dequantMatrix`, hard rule 8) at the reused
/// pre-committed Tier K gate — a wiring bug withholds the number.
public final class QuantMatvecMicrobench {
    /// Default in-run protocol, mirroring the pinned triad shape (P0B-4):
    /// warmups absorb first-touch page faults and pipeline warm-up. These
    /// are run-shape conventions, not gates — the on-device gate is stated
    /// over the D8 repeats protocol regardless of per-run iteration count.
    public static let defaultWarmupIterations = 2
    public static let defaultMeasuredIterations = 10

    /// One matvec dispatch site: a packed triplet plus its dims and store
    /// precision, in pipeline order.
    struct Site {
        let role: String
        let name: String
        let qByteOffset: Int
        let scalesByteOffset: Int
        let biasesByteOffset: Int
        let outDim: Int
        let inDim: Int
        let fp32Output: Bool
    }

    /// The per-role shape roster of one token's matvec sweep, derived from
    /// the config (pure — tests pin the byte accounting against it without
    /// an artifact). Order matches the pipeline.
    public static func siteSpecs(
        config: ModelConfig
    ) -> [(role: String, outDim: Int, inDim: Int, count: Int)] {
        let hidden = config.hiddenSize
        let qOut = config.numAttentionHeads * config.headDim
        let kvOut = config.numKeyValueHeads * config.headDim
        let intermediate = config.intermediateSize
        let layers = config.numHiddenLayers
        return [
            ("q_proj", qOut, hidden, layers),
            ("k_proj", kvOut, hidden, layers),
            ("v_proj", kvOut, hidden, layers),
            ("o_proj", hidden, qOut, layers),
            ("gate_proj", intermediate, hidden, layers),
            ("up_proj", intermediate, hidden, layers),
            ("down_proj", hidden, intermediate, layers),
            ("lm_head", config.vocabSize, hidden, 1),
        ]
    }

    /// Packed bytes one [outDim, inDim] q4g64 triplet occupies (and one
    /// matvec dispatch reads): q u32 words + fp16 scales + fp16 biases
    /// = 0.5625 bytes per weight element.
    public static func packedBytes(outDim: Int, inDim: Int) -> Int {
        let qBytes = outDim * (inDim / Q4G64.codesPerWord) * 4
        let groupBytes = outDim * (inDim / Q4G64.groupSize) * 2
        return qBytes + 2 * groupBytes
    }

    /// Total packed bytes one aggregate iteration reads — 967,753,728
    /// (≈ 0.967 GB, the gates-entry number) at the pinned dims. The tied
    /// lm_head reads the embedding triplet (stored once, P3-1); traffic
    /// counts per read, so tying does not change this number.
    public static func totalPackedBytes(config: ModelConfig) -> Int {
        siteSpecs(config: config).reduce(0) { sum, spec in
            sum + packedBytes(outDim: spec.outDim, inDim: spec.inDim) * spec.count
        }
    }

    /// Deterministic fp16-exact input activations (multiples of 1/64,
    /// |v| ≤ 0.47): pure function of the index so the CPU spot-check oracle
    /// recomputes them without a copy.
    public static func inputValue(at index: Int) -> Float {
        Float((index &* 37 &+ 11) % 61 - 30) * 0.015625
    }

    /// The reused pre-committed Tier K comparison (|Δ| ≤ max(2⁻⁹·M, 2⁻¹¹),
    /// M = max |reference|) as a pure function so its teeth are testable.
    public static func spotCheckDelta(
        gpu: [Float], reference: [Float]
    ) -> (maxAbsDelta: Float, tolerance: Float, passed: Bool) {
        let m = reference.map(abs).max() ?? 0
        let tolerance = max(Float(exp2(-9.0)) * m, Float(exp2(-11.0)))
        var maxDelta: Float = 0
        for i in 0..<min(gpu.count, reference.count) {
            maxDelta = max(maxDelta, abs(gpu[i] - reference[i]))
        }
        let sameCount = gpu.count == reference.count
        return (maxDelta, tolerance, sameCount && maxDelta <= tolerance)
    }

    private let context: MetalContext
    private let kernels: QuantKernels
    private let packed: PackedCheckpoint
    /// Whole-checkpoint residency buffer, bound three times per triplet
    /// (the P3-5 pattern).
    private let weights: GPUWeights
    private let sites: [Site]
    private let counter = DispatchCounter()
    /// One shared input buffer per distinct inDim (read-only — no hazards).
    private let inputBuffers: [Int: MTLBuffer]
    /// One output buffer per site: distinct destinations, like the real
    /// pipeline, so hazard tracking cannot serialize independent matvecs.
    private let outputBuffers: [MTLBuffer]

    /// Resolves the sites (config shapes validated against the packed dims,
    /// the GPUModel species) and allocates the scratch buffers.
    public init(
        packed: PackedCheckpoint, config: ModelConfig, context: MetalContext,
        residency: WeightsResidency = .mmap
    ) throws {
        let kernels = try QuantKernels(context: context)
        let weights = try GPUWeights(
            file: packed.file, context: context, residency: residency)

        // Locals only inside this helper (no `self` capture — stored
        // properties are assigned together at the end of init).
        func site(
            role: String, name: String, outDim: Int, inDim: Int,
            fp32Output: Bool = false
        ) throws -> Site {
            let dims = try packed.dims(for: name)
            guard dims.outDim == outDim, dims.inDim == inDim else {
                throw ModelError.badWeightShape(
                    tensor: name, expected: [outDim, inDim],
                    actual: [dims.outDim, dims.inDim])
            }
            return Site(
                role: role, name: name,
                qByteOffset: try weights.byteOffset(for: name + Q4G64.qSuffix),
                scalesByteOffset: try weights.byteOffset(
                    for: name + Q4G64.scalesSuffix),
                biasesByteOffset: try weights.byteOffset(
                    for: name + Q4G64.biasesSuffix),
                outDim: outDim, inDim: inDim, fp32Output: fp32Output)
        }

        let hidden = config.hiddenSize
        let qOut = config.numAttentionHeads * config.headDim
        let kvOut = config.numKeyValueHeads * config.headDim
        let intermediate = config.intermediateSize
        var resolvedSites: [Site] = []
        for layer in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(layer)."
            resolvedSites.append(try site(
                role: "q_proj", name: prefix + "self_attn.q_proj.weight",
                outDim: qOut, inDim: hidden))
            resolvedSites.append(try site(
                role: "k_proj", name: prefix + "self_attn.k_proj.weight",
                outDim: kvOut, inDim: hidden))
            resolvedSites.append(try site(
                role: "v_proj", name: prefix + "self_attn.v_proj.weight",
                outDim: kvOut, inDim: hidden))
            resolvedSites.append(try site(
                role: "o_proj", name: prefix + "self_attn.o_proj.weight",
                outDim: hidden, inDim: qOut))
            resolvedSites.append(try site(
                role: "gate_proj", name: prefix + "mlp.gate_proj.weight",
                outDim: intermediate, inDim: hidden))
            resolvedSites.append(try site(
                role: "up_proj", name: prefix + "mlp.up_proj.weight",
                outDim: intermediate, inDim: hidden))
            resolvedSites.append(try site(
                role: "down_proj", name: prefix + "mlp.down_proj.weight",
                outDim: hidden, inDim: intermediate))
        }
        // Tied embeddings: the packed artifact stores the tied matrix once
        // (P3-1) and lm_head consumes it as [out, in] directly. An untied
        // config resolves lm_head.weight and fails loudly when absent.
        resolvedSites.append(try site(
            role: "lm_head",
            name: config.tieWordEmbeddings
                ? "model.embed_tokens.weight" : "lm_head.weight",
            outDim: config.vocabSize, inDim: hidden, fp32Output: true))

        var inputs: [Int: MTLBuffer] = [:]
        for inDim in Set(resolvedSites.map(\.inDim)) {
            guard let buffer = context.device.makeBuffer(
                length: inDim * 2, options: .storageModeShared) else {
                throw MetalHarnessError.bufferAllocationFailed(length: inDim * 2)
            }
            let pointer = buffer.contents()
                .bindMemory(to: Float16.self, capacity: inDim)
            for i in 0..<inDim {
                pointer[i] = Float16(Self.inputValue(at: i))
            }
            inputs[inDim] = buffer
        }
        let outputs = try resolvedSites.map { site -> MTLBuffer in
            let length = site.outDim * (site.fp32Output ? 4 : 2)
            guard let buffer = context.device.makeBuffer(
                length: length, options: .storageModeShared) else {
                throw MetalHarnessError.bufferAllocationFailed(length: length)
            }
            return buffer
        }

        self.context = context
        self.packed = packed
        self.kernels = kernels
        self.weights = weights
        self.sites = resolvedSites
        self.inputBuffers = inputs
        self.outputBuffers = outputs
        kernels.dispatchCounter = counter
    }

    /// Runs the spot check, then `warmupIterations + measuredIterations`
    /// aggregate passes (one command buffer each, all sites), then the
    /// per-shape passes (one role per command buffer, same iteration shape).
    public func run(
        warmupIterations: Int = QuantMatvecMicrobench.defaultWarmupIterations,
        measuredIterations: Int = QuantMatvecMicrobench.defaultMeasuredIterations,
        includePerShape: Bool = true
    ) throws -> QuantMatvecMicrobenchResult {
        guard warmupIterations >= 0, measuredIterations >= 1 else {
            throw KernelInputError.invalidIterations(
                warmup: warmupIterations, measured: measuredIterations)
        }

        let spot = try runSpotCheck()

        var warmupTimings: [DispatchTiming] = []
        var measuredTimings: [DispatchTiming] = []
        var dispatchesPerIteration = 0
        for iteration in 0..<(warmupIterations + measuredIterations) {
            counter.reset()
            let timing = try context.timedDispatch { encoder in
                for (index, site) in sites.enumerated() {
                    try encode(site, output: outputBuffers[index], into: encoder)
                }
            }
            dispatchesPerIteration = counter.count
            if iteration < warmupIterations {
                warmupTimings.append(timing)
            } else {
                measuredTimings.append(timing)
            }
        }

        var shapes: [QuantMatvecShapeResult] = []
        if includePerShape {
            var seenRoles: [String] = []
            for site in sites where !seenRoles.contains(site.role) {
                seenRoles.append(site.role)
            }
            for role in seenRoles {
                shapes.append(try runShape(
                    role: role, warmupIterations: warmupIterations,
                    measuredIterations: measuredIterations))
            }
        }

        return QuantMatvecMicrobenchResult(
            totalPackedBytes: sites.reduce(0) {
                $0 + Self.packedBytes(outDim: $1.outDim, inDim: $1.inDim)
            },
            dispatchesPerIteration: dispatchesPerIteration,
            warmupTimings: warmupTimings,
            measuredTimings: measuredTimings,
            shapes: shapes,
            spotCheckSite: spot.site,
            spotCheckMaxAbsDelta: spot.maxAbsDelta,
            spotCheckTolerance: spot.tolerance)
    }

    // MARK: - Internals

    private func encode(
        _ site: Site, output: MTLBuffer, into encoder: MTLComputeCommandEncoder
    ) throws {
        guard let input = inputBuffers[site.inDim] else {
            // Structurally impossible: inputs were built from the site list.
            throw QuantKernelError.bufferTooSmall(
                buffer: "input(\(site.inDim))", requiredBytes: site.inDim * 2,
                actualBytes: 0)
        }
        try kernels.encodeMatvec(
            into: encoder,
            q: weights.buffer, qByteOffset: site.qByteOffset,
            scales: weights.buffer, scalesByteOffset: site.scalesByteOffset,
            biases: weights.buffer, biasesByteOffset: site.biasesByteOffset,
            input: input, outDim: site.outDim, inDim: site.inDim,
            output: output, fp32Output: site.fp32Output)
    }

    private func runShape(
        role: String, warmupIterations: Int, measuredIterations: Int
    ) throws -> QuantMatvecShapeResult {
        let roleSites = sites.indices.filter { sites[$0].role == role }
        let first = sites[roleSites[0]]
        var warmup: [DispatchTiming] = []
        var measured: [DispatchTiming] = []
        for iteration in 0..<(warmupIterations + measuredIterations) {
            let timing = try context.timedDispatch { encoder in
                for index in roleSites {
                    try encode(
                        sites[index], output: outputBuffers[index],
                        into: encoder)
                }
            }
            if iteration < warmupIterations {
                warmup.append(timing)
            } else {
                measured.append(timing)
            }
        }
        return QuantMatvecShapeResult(
            role: role, outDim: first.outDim, inDim: first.inDim,
            dispatchesPerIteration: roleSites.count,
            packedBytesPerIteration: roleSites.reduce(0) {
                $0 + Self.packedBytes(
                    outDim: sites[$1].outDim, inDim: sites[$1].inDim)
            },
            warmupTimings: warmup, measuredTimings: measured)
    }

    /// Dispatches the first site once and diffs its output against the
    /// CPU-quant oracle (sgemm over the dequantized matrix, hard rule 8) at
    /// the reused Tier K gate. A failure throws — the bench never reports a
    /// figure its own output contradicts.
    private func runSpotCheck() throws -> (site: String, maxAbsDelta: Float, tolerance: Float) {
        let site = sites[0]
        let output = outputBuffers[0]
        try context.timedDispatch { encoder in
            try encode(site, output: output, into: encoder)
        }
        let pointer = output.contents()
            .bindMemory(to: Float16.self, capacity: site.outDim)
        let gpu = (0..<site.outDim).map { Float(pointer[$0]) }

        let x = (0..<site.inDim).map {
            Float(Float16(Self.inputValue(at: $0)))
        }
        let reference = try BLAS.sgemm(
            a: x, b: try packed.dequantMatrix(site.name),
            m: 1, k: site.inDim, n: site.outDim, transposeB: true)
        let check = Self.spotCheckDelta(gpu: gpu, reference: reference)
        guard check.passed else {
            throw QuantMatvecMicrobenchError.spotCheckFailed(
                site: site.name, maxAbsDelta: check.maxAbsDelta,
                tolerance: check.tolerance)
        }
        return (site.name, check.maxAbsDelta, check.tolerance)
    }
}
