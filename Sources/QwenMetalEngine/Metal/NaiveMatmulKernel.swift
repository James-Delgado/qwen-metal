import Metal

/// Result of one matmul run: row-major C values plus the mandatory dual
/// GPU + wall timing (hard rule 7 — timing always rides along, never optional).
public struct MatmulResult: Sendable {
    public let values: [Float16]
    public let timing: DispatchTiming
}

/// Kernel 2 (PRD 0b.4): naive fp16 matmul, C(m×n) = A(m×k) · B(k×n), all
/// row-major, one thread per output element. Deliberately unoptimized — this
/// is the correctness-first baseline (hard rule 3); no tiling, no threadgroup
/// memory, no vectorized loads. Operands are fp16; accumulation is fp32 with a
/// single round to half on store, per the gate pre-committed in DECISIONS.md
/// 2026-08-21 (P0B-3). Compiled from source at runtime per the P0B-1 convention.
public final class NaiveMatmulKernel {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void naive_matmul_fp16(device const half *a  [[buffer(0)]],
                                  device const half *b  [[buffer(1)]],
                                  device half *c        [[buffer(2)]],
                                  constant uint &m      [[buffer(3)]],
                                  constant uint &k      [[buffer(4)]],
                                  constant uint &n      [[buffer(5)]],
                                  uint2 gid [[thread_position_in_grid]]) {
        // gid.x = output column (0..<n), gid.y = output row (0..<m).
        if (gid.x >= n || gid.y >= m) return;
        float acc = 0.0f;
        for (uint p = 0; p < k; ++p) {
            acc += float(a[gid.y * k + p]) * float(b[p * n + gid.x]);
        }
        c[gid.y * n + gid.x] = half(acc);
    }
    """

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.makeLibrary(source: Self.source)
        self.pipeline = try context.makeComputePipeline(
            library: library, function: "naive_matmul_fp16"
        )
    }

    /// Computes C(m×n) = A(m×k) · B(k×n). Inputs are row-major flat arrays and
    /// are not mutated; the output is a separate buffer (engine convention).
    public func run(
        a: [Float16], b: [Float16], m: Int, k: Int, n: Int
    ) throws -> MatmulResult {
        guard m > 0, k > 0, n > 0 else {
            throw KernelInputError.nonPositiveDimensions(m: m, k: k, n: n)
        }
        guard a.count == m * k else {
            throw KernelInputError.elementCountMismatch(
                matrix: "A", expected: m * k, actual: a.count
            )
        }
        guard b.count == k * n else {
            throw KernelInputError.elementCountMismatch(
                matrix: "B", expected: k * n, actual: b.count
            )
        }

        let elementStride = MemoryLayout<Float16>.stride
        let outputCount = m * n
        let device = context.device
        guard
            let aBuffer = a.withUnsafeBytes({
                device.makeBuffer(bytes: $0.baseAddress!, length: a.count * elementStride,
                                  options: .storageModeShared)
            }),
            let bBuffer = b.withUnsafeBytes({
                device.makeBuffer(bytes: $0.baseAddress!, length: b.count * elementStride,
                                  options: .storageModeShared)
            }),
            let cBuffer = device.makeBuffer(length: outputCount * elementStride,
                                            options: .storageModeShared)
        else {
            throw MetalHarnessError.bufferAllocationFailed(
                length: outputCount * elementStride
            )
        }

        var mDim = UInt32(m)
        var kDim = UInt32(k)
        var nDim = UInt32(n)
        let pipeline = self.pipeline
        let timing = try context.timedDispatch { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(aBuffer, offset: 0, index: 0)
            encoder.setBuffer(bBuffer, offset: 0, index: 1)
            encoder.setBuffer(cBuffer, offset: 0, index: 2)
            encoder.setBytes(&mDim, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&kDim, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&nDim, length: MemoryLayout<UInt32>.stride, index: 5)
            // 2D grid over the output; dispatchThreads handles the ragged edge
            // (kernel bounds-check is belt-and-braces for non-uniform support).
            let side = 16
            let width = min(side, pipeline.maxTotalThreadsPerThreadgroup)
            let height = max(1, min(side, pipeline.maxTotalThreadsPerThreadgroup / width))
            encoder.dispatchThreads(
                MTLSize(width: n, height: m, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: height, depth: 1)
            )
        }

        let values = [Float16](
            UnsafeBufferPointer(
                start: cBuffer.contents().bindMemory(to: Float16.self, capacity: outputCount),
                count: outputCount
            )
        )
        return MatmulResult(values: values, timing: timing)
    }
}
