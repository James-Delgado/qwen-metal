import Metal

/// Result of one triad bandwidth run: per-iteration dual timings (hard rule 7 —
/// GPU + wall always ride together), the derived GB/s figures, and sampled
/// output values so callers can verify the measured dispatches did real work.
public struct TriadBandwidthResult: Sendable {
    /// fp32 elements per buffer (three buffers: a, b, c).
    public let elementCount: Int
    /// Bytes moved per iteration: read b, read c, write a = 3 × N × 4.
    public let bytesPerIteration: Int
    /// Warmup iterations — timed but excluded from every GB/s figure.
    public let warmupTimings: [DispatchTiming]
    public let measuredTimings: [DispatchTiming]
    /// a[i] read back after the final iteration, keyed by requested index.
    public let outputSamples: [Int: Float]

    /// Per measured iteration: bytesPerIteration ÷ gpuDuration, in GB/s
    /// (GB = 10^9 bytes). GPU time defines the rate; the wall clock is
    /// recorded alongside in `measuredTimings` per hard rule 7.
    public var measuredGBps: [Double] {
        measuredTimings.map { Double(bytesPerIteration) / $0.gpuDuration / 1e9 }
    }

    /// The pinned "sustained" figure (DECISIONS.md 2026-08-21, P0B-4 entry):
    /// median of the measured iterations — a rate the decode loop could
    /// actually sustain, not a lucky burst.
    public var sustainedGBps: Double {
        let sorted = measuredGBps.sorted()
        guard !sorted.isEmpty else { return .nan }
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }

    public var minGBps: Double { measuredGBps.min() ?? .nan }
    public var maxGBps: Double { measuredGBps.max() ?? .nan }
}

/// Kernel 3 (PRD 0a.6/0b.5): STREAM triad, a[i] = b[i] + s·c[i], measuring
/// achievable DRAM bandwidth. The iPhone run of this kernel (P0A-1, James) is
/// the roofline denominator for the entire project (PLAN.md invariant 1); the
/// Mac run is dev-loop sanity. float4 loads/stores so the measurement reflects
/// achievable streaming bandwidth, not scalar-issue limits; the official
/// working set is pinned at >= 1 GiB streamed per iteration so the figure is
/// DRAM, not SLC. Compiled from source at runtime per the P0B-1 convention.
public final class TriadBandwidthKernel {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void triad(device const float4 *b [[buffer(0)]],
                      device const float4 *c [[buffer(1)]],
                      device float4 *a       [[buffer(2)]],
                      constant float &s      [[buffer(3)]],
                      constant uint &n4      [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
        if (gid >= n4) return;
        a[gid] = b[gid] + s * c[gid];
    }
    """

    /// Pinned official protocol (DECISIONS.md 2026-08-21, P0B-4 entry):
    /// 96 × 2^20 elements/buffer = 384 MiB each, 1.125 GiB streamed/iteration.
    public static let benchmarkElementCount = 96 << 20
    /// Pinned official protocol: 2 warmup (discarded) + 10 measured iterations.
    public static let benchmarkWarmupIterations = 2
    public static let benchmarkMeasuredIterations = 10

    /// Triad scalar s. Exactly representable so s·c rounds once at most.
    public static let scalar: Float = 0.75

    /// Deterministic input patterns, period 4096, values in [-1, 1) — the
    /// domain the pre-committed correctness gate is stated over. Pure
    /// functions of the index so the CPU oracle needs no copy of the buffers.
    private static let patternPeriod = 4096

    public static func bValue(at index: Int) -> Float {
        Float(index % patternPeriod) * (1.0 / 2048.0) - 1.0
    }

    public static func cValue(at index: Int) -> Float {
        Float((index &+ 1234) % patternPeriod) * (1.0 / 2048.0) - 1.0
    }

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.makeLibrary(source: Self.source)
        self.pipeline = try context.makeComputePipeline(library: library, function: "triad")
    }

    /// Runs `warmupIterations + measuredIterations` full triad passes over
    /// freshly allocated buffers and returns per-iteration dual timing plus
    /// sampled outputs. `elementCount` must be a positive multiple of 4
    /// (float4 kernel). Buffers are initialized from the deterministic
    /// patterns above and live only for the duration of the call.
    public func run(
        elementCount: Int,
        warmupIterations: Int,
        measuredIterations: Int,
        sampleIndices: [Int]
    ) throws -> TriadBandwidthResult {
        guard elementCount > 0 else {
            throw KernelInputError.nonPositiveElementCount(count: elementCount)
        }
        guard elementCount % 4 == 0 else {
            throw KernelInputError.elementCountNotMultiple(of: 4, count: elementCount)
        }
        guard warmupIterations >= 0, measuredIterations >= 1 else {
            throw KernelInputError.invalidIterations(
                warmup: warmupIterations, measured: measuredIterations
            )
        }
        if let bad = sampleIndices.first(where: { $0 < 0 || $0 >= elementCount }) {
            throw KernelInputError.sampleIndexOutOfRange(index: bad, count: elementCount)
        }

        let byteLength = elementCount * MemoryLayout<Float>.stride
        let device = context.device
        guard
            let bBuffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
            let cBuffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
            let aBuffer = device.makeBuffer(length: byteLength, options: .storageModeShared)
        else {
            throw MetalHarnessError.bufferAllocationFailed(length: byteLength)
        }
        Self.fillPattern(bBuffer, count: elementCount, indexOffset: 0)
        Self.fillPattern(cBuffer, count: elementCount, indexOffset: 1234)

        var s = Self.scalar
        var n4 = UInt32(elementCount / 4)
        let pipeline = self.pipeline
        var warmupTimings: [DispatchTiming] = []
        var measuredTimings: [DispatchTiming] = []
        for iteration in 0..<(warmupIterations + measuredIterations) {
            let timing = try context.timedDispatch { encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(bBuffer, offset: 0, index: 0)
                encoder.setBuffer(cBuffer, offset: 0, index: 1)
                encoder.setBuffer(aBuffer, offset: 0, index: 2)
                encoder.setBytes(&s, length: MemoryLayout<Float>.stride, index: 3)
                encoder.setBytes(&n4, length: MemoryLayout<UInt32>.stride, index: 4)
                let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
                encoder.dispatchThreads(
                    MTLSize(width: elementCount / 4, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
                )
            }
            if iteration < warmupIterations {
                warmupTimings.append(timing)
            } else {
                measuredTimings.append(timing)
            }
        }

        let aPointer = aBuffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        var outputSamples: [Int: Float] = [:]
        for index in sampleIndices {
            outputSamples[index] = aPointer[index]
        }

        return TriadBandwidthResult(
            elementCount: elementCount,
            bytesPerIteration: 3 * byteLength,
            warmupTimings: warmupTimings,
            measuredTimings: measuredTimings,
            outputSamples: outputSamples
        )
    }

    /// Fills a buffer with the period-4096 pattern: one period written
    /// directly, then doubled with memcpy so gigabyte fills run at memory
    /// speed instead of one bounds-checked store per element. Every copy's
    /// destination offset is a multiple of the period (the fill grows
    /// period → 2·period → 4·period …), so `pointer[i] == pattern(i)` holds
    /// for the whole buffer including a ragged final copy.
    private static func fillPattern(_ buffer: MTLBuffer, count: Int, indexOffset: Int) {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        let period = min(patternPeriod, count)
        for i in 0..<period {
            pointer[i] = Float((i &+ indexOffset) % patternPeriod) * (1.0 / 2048.0) - 1.0
        }
        var filled = period
        let stride = MemoryLayout<Float>.stride
        while filled < count {
            let copyCount = min(filled, count - filled)
            memcpy(pointer + filled, pointer, copyCount * stride)
            filled += copyCount
        }
    }
}
