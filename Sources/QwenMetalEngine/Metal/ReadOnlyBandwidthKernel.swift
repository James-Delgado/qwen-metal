import Metal

/// Result of one read-only streaming bandwidth run (BW-1): per-iteration dual
/// timings (hard rule 7), the byte accounting the GB/s figure is stated over,
/// and sampled simdgroup partials so callers can verify the measured
/// dispatches really reduced the buffers.
public struct ReadOnlyBandwidthResult: Sendable {
    /// fp32 elements per buffer (three read-only buffers).
    public let elementCount: Int
    /// float4 read per thread per buffer (pinned 64; other values are DIAGNOSTIC).
    public let float4sPerThread: Int
    /// Threads in the grid, T = N / (4 × float4sPerThread).
    public let threadCount: Int
    /// One float4 partial per simdgroup, across the whole (uniform) grid.
    public let partialCount: Int
    /// Bytes READ per iteration = 3 × N × 4. This is the GB/s numerator.
    public let bytesReadPerIteration: Int
    /// Bytes written per iteration = partialCount × 16. Reported, never counted.
    public let bytesWrittenPerIteration: Int
    /// Warmup iterations — timed but excluded from every GB/s figure.
    public let warmupTimings: [DispatchTiming]
    public let measuredTimings: [DispatchTiming]
    /// Simdgroup partials read back after the final iteration, keyed by index.
    public let partialSamples: [Int: SIMD4<Float>]

    /// Per measured iteration: bytesReadPerIteration ÷ gpuDuration, GB/s
    /// (GB = 10^9 bytes). GPU time defines the rate; the wall clock rides
    /// alongside in `measuredTimings` per hard rule 7.
    public var measuredGBps: [Double] {
        measuredTimings.map { Double(bytesReadPerIteration) / $0.gpuDuration / 1e9 }
    }

    /// The pinned "sustained" figure (DECISIONS.md 2026-09-26, BW-1 entry):
    /// median of the measured iterations — the P0B-4 statistic.
    public var sustainedGBps: Double { BenchMath.median(measuredGBps) }
    public var minGBps: Double { measuredGBps.min() ?? .nan }
    public var maxGBps: Double { measuredGBps.max() ?? .nan }

    /// Write traffic as a fraction of the bytes read (0.017% at the pinned N).
    public var writeFraction: Double {
        Double(bytesWrittenPerIteration) / Double(bytesReadPerIteration)
    }
}

/// BW-1: read-only streaming reduce over the triad's three buffers, measuring
/// read-mostly achievable DRAM bandwidth — the roofline denominator for
/// decode-shaped (read-dominated) traffic, cited alongside the 2R+1W triad.
/// Geometry, working set, gate, and statistic are pinned in DECISIONS.md
/// 2026-09-26 BEFORE this kernel existed:
///
/// - thread t of T = N/256 reads float4 t + k·T (k in 0..<64) from each of
///   the three buffers (coalesced grid-stride) into a float4 partial;
/// - the 32-lane simdgroup reduces it and lane 0 writes ONE float4, so the
///   write stream is 0.017% of the reads at the pinned N and excluded from
///   the numerator;
/// - every term is a multiple of 2^-11 with |term| < 1 and a partial holds
///   6144 terms per component, so every intermediate is exact in fp32 and the
///   CPU oracle must agree bit-for-bit regardless of summation order.
///
/// Compiled from source at runtime per the P0B-1 convention.
public final class ReadOnlyBandwidthKernel {
    /// Pinned geometry: float4 per thread per buffer. Other values in
    /// `allowedFloat4sPerThread` exist ONLY for the diagnostic sweep (is the
    /// pinned figure kernel-limited?); rows use the pin.
    public static let float4sPerThreadPerBuffer = 64
    /// The exactness argument (6144 terms per component < 2^13) holds for any
    /// depth up to the pin, so the diagnostic range is 1...64.
    public static let allowedFloat4sPerThread = 1...float4sPerThreadPerBuffer
    /// Pinned geometry: uniform 256-thread threadgroups (8 simdgroups).
    public static let threadgroupWidth = 256
    public static let simdgroupWidth = 32
    public static let bufferCount = 3
    /// N must be a positive multiple of this (float4 × 64 per thread) at the pin.
    public static let elementAlignment = 4 * float4sPerThreadPerBuffer

    public static func elementAlignment(float4sPerThread: Int) -> Int {
        4 * float4sPerThread
    }

    /// Pinned official protocol: identical working set to the triad —
    /// 96 × 2^20 elements/buffer = 384 MiB each, 1.125 GiB read/iteration.
    public static let benchmarkElementCount = TriadBandwidthKernel.benchmarkElementCount
    /// Pinned official protocol: 2 warmup (discarded) + 10 measured iterations.
    public static let benchmarkWarmupIterations = 2
    public static let benchmarkMeasuredIterations = 10

    /// Deterministic period-4096 patterns, values k × 2^-11 − 1 in [-1, 1):
    /// buffers 0 and 1 are exactly the triad's b and c; buffer 2 continues the
    /// offset sequence.
    public static let patternOffsets = [0, 1234, 2468]

    private static func makeSource(float4sPerThread: Int) -> String { """
    #include <metal_stdlib>
    using namespace metal;

    kernel void readonly_reduce(device const float4 *b0   [[buffer(0)]],
                                device const float4 *b1   [[buffer(1)]],
                                device const float4 *b2   [[buffer(2)]],
                                device float4 *partials   [[buffer(3)]],
                                constant uint &threadCount [[buffer(4)]],
                                uint gid        [[thread_position_in_grid]],
                                uint lane       [[thread_index_in_simdgroup]],
                                uint simdIndex  [[simdgroup_index_in_threadgroup]],
                                uint simdCount  [[simdgroups_per_threadgroup]],
                                uint groupIndex [[threadgroup_position_in_grid]]) {
        float4 acc = float4(0.0f);
        if (gid < threadCount) {
            uint index = gid;
            for (uint k = 0; k < \(float4sPerThread); ++k) {
                acc += b0[index] + b1[index] + b2[index];
                index += threadCount;
            }
        }
        acc = simd_sum(acc);
        if (lane == 0) {
            partials[groupIndex * simdCount + simdIndex] = acc;
        }
    }
    """ }

    public static func inputValue(buffer: Int, at index: Int) -> Float {
        precondition((0..<bufferCount).contains(buffer), "buffer index out of range")
        return TriadBandwidthKernel.patternValue(at: index, indexOffset: patternOffsets[buffer])
    }

    /// T = N / (4 × float4sPerThread) for a valid N.
    public static func threadCount(
        elementCount: Int, float4sPerThread: Int = float4sPerThreadPerBuffer
    ) -> Int {
        elementCount / elementAlignment(float4sPerThread: float4sPerThread)
    }

    /// Uniform grid: ceil(T / 256) threadgroups × 8 simdgroups.
    public static func partialCount(
        elementCount: Int, float4sPerThread: Int = float4sPerThreadPerBuffer
    ) -> Int {
        let threads = threadCount(elementCount: elementCount, float4sPerThread: float4sPerThread)
        let groups = (threads + threadgroupWidth - 1) / threadgroupWidth
        return groups * (threadgroupWidth / simdgroupWidth)
    }

    /// The CPU oracle for one simdgroup partial — a pure function of the
    /// index pattern (no copy of the buffers). Accumulated in Double and
    /// narrowed once; exact either way (see the class comment).
    public static func cpuPartial(
        index p: Int, elementCount: Int, float4sPerThread: Int = float4sPerThreadPerBuffer
    ) -> SIMD4<Float> {
        let threads = threadCount(elementCount: elementCount, float4sPerThread: float4sPerThread)
        var sum = SIMD4<Double>(repeating: 0)
        for lane in 0..<simdgroupWidth {
            let t = simdgroupWidth * p + lane
            guard t < threads else { break }
            for k in 0..<float4sPerThread {
                let base = 4 * (t + k * threads)
                for component in 0..<4 {
                    for buffer in 0..<bufferCount {
                        sum[component] += Double(inputValue(buffer: buffer, at: base + component))
                    }
                }
            }
        }
        return SIMD4<Float>(sum)
    }

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState
    /// Geometry this instance was compiled for (pinned 64 unless diagnostic).
    public let float4sPerThread: Int

    public init(
        context: MetalContext, float4sPerThread: Int = float4sPerThreadPerBuffer
    ) throws {
        guard Self.allowedFloat4sPerThread.contains(float4sPerThread) else {
            throw KernelInputError.float4sPerThreadOutOfRange(
                value: float4sPerThread, allowed: Self.allowedFloat4sPerThread)
        }
        self.context = context
        self.float4sPerThread = float4sPerThread
        let library = try context.makeLibrary(
            source: Self.makeSource(float4sPerThread: float4sPerThread))
        self.pipeline = try context.makeComputePipeline(library: library, function: "readonly_reduce")
        precondition(
            pipeline.maxTotalThreadsPerThreadgroup >= Self.threadgroupWidth,
            "pinned geometry needs \(Self.threadgroupWidth)-thread threadgroups")
    }

    /// Runs `warmupIterations + measuredIterations` full passes over freshly
    /// allocated buffers and returns per-iteration dual timing plus sampled
    /// partials. `elementCount` must be a positive multiple of
    /// 4 × float4sPerThread (256 at the pin).
    public func run(
        elementCount: Int,
        warmupIterations: Int,
        measuredIterations: Int,
        samplePartialIndices: [Int]
    ) throws -> ReadOnlyBandwidthResult {
        guard elementCount > 0 else {
            throw KernelInputError.nonPositiveElementCount(count: elementCount)
        }
        let alignment = Self.elementAlignment(float4sPerThread: float4sPerThread)
        guard elementCount % alignment == 0 else {
            throw KernelInputError.elementCountNotMultiple(of: alignment, count: elementCount)
        }
        guard warmupIterations >= 0, measuredIterations >= 1 else {
            throw KernelInputError.invalidIterations(
                warmup: warmupIterations, measured: measuredIterations)
        }
        let partialCount = Self.partialCount(
            elementCount: elementCount, float4sPerThread: float4sPerThread)
        if let bad = samplePartialIndices.first(where: { $0 < 0 || $0 >= partialCount }) {
            throw KernelInputError.sampleIndexOutOfRange(index: bad, count: partialCount)
        }

        let byteLength = elementCount * MemoryLayout<Float>.stride
        let partialsByteLength = partialCount * MemoryLayout<SIMD4<Float>>.stride
        let device = context.device
        var inputs: [MTLBuffer] = []
        for offset in Self.patternOffsets {
            guard let buffer = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
                throw MetalHarnessError.bufferAllocationFailed(length: byteLength)
            }
            TriadBandwidthKernel.fillPattern(buffer, count: elementCount, indexOffset: offset)
            inputs.append(buffer)
        }
        guard let partials = device.makeBuffer(
            length: partialsByteLength, options: .storageModeShared)
        else {
            throw MetalHarnessError.bufferAllocationFailed(length: partialsByteLength)
        }

        let threads = Self.threadCount(
            elementCount: elementCount, float4sPerThread: float4sPerThread)
        var threadCount32 = UInt32(threads)
        let groups = (threads + Self.threadgroupWidth - 1) / Self.threadgroupWidth
        let pipeline = self.pipeline
        var warmupTimings: [DispatchTiming] = []
        var measuredTimings: [DispatchTiming] = []
        for iteration in 0..<(warmupIterations + measuredIterations) {
            let timing = try context.timedDispatch { encoder in
                encoder.setComputePipelineState(pipeline)
                for (slot, buffer) in inputs.enumerated() {
                    encoder.setBuffer(buffer, offset: 0, index: slot)
                }
                encoder.setBuffer(partials, offset: 0, index: 3)
                encoder.setBytes(&threadCount32, length: MemoryLayout<UInt32>.stride, index: 4)
                encoder.dispatchThreadgroups(
                    MTLSize(width: groups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth, height: 1, depth: 1)
                )
            }
            if iteration < warmupIterations {
                warmupTimings.append(timing)
            } else {
                measuredTimings.append(timing)
            }
        }

        let partialPointer = partials.contents().bindMemory(
            to: SIMD4<Float>.self, capacity: partialCount)
        var partialSamples: [Int: SIMD4<Float>] = [:]
        for index in samplePartialIndices {
            partialSamples[index] = partialPointer[index]
        }

        return ReadOnlyBandwidthResult(
            elementCount: elementCount,
            float4sPerThread: float4sPerThread,
            threadCount: threads,
            partialCount: partialCount,
            bytesReadPerIteration: Self.bufferCount * byteLength,
            bytesWrittenPerIteration: partialsByteLength,
            warmupTimings: warmupTimings,
            measuredTimings: measuredTimings,
            partialSamples: partialSamples
        )
    }
}
