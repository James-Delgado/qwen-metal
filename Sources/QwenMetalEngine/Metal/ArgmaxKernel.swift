import Metal

/// P4-8 (phase-4.md 2026-09-12 addendum): fp32 argmax on the GPU, so the
/// free-running decode loop reads back one token id (4 bytes) instead of the
/// full-vocab logits (~605 KB at the pinned dims).
///
/// EXACT-EQUALITY contract (no tolerance constant, DECISIONS.md 2026-09-12):
/// the selected index equals `Argmax.firstIndex` on the same values — the
/// left-fold `if values[i] > values[best]` scan. That scan's semantics,
/// restated as an order-free property the reduction can implement:
///
/// - Ties resolve to the LOWEST index (`>` is strict).
/// - `+0.0` and `-0.0` compare equal under IEEE `>`, so they tie by index.
/// - A NaN never becomes `best` (nothing compares greater than NaN, and NaN
///   compares greater than nothing) — EXCEPT when `values[0]` is NaN: index 0
///   is `best` by initialization and can never be displaced. So: if
///   `values[0]` is NaN the answer is 0; otherwise NaN behaves as -inf
///   (it ties with real -inf and loses on index, exactly like the scan).
///
/// The kernel maps each element to a 64-bit key — monotonic unsigned image
/// of the fp32 value in the high word (NaN → the -inf image, -0.0
/// canonicalized to +0.0), bitwise-NOT of the index in the low word — and
/// takes the MAX key. Max is associative and commutative, so the result is
/// bitwise deterministic under ANY reduction order (stronger than the P4-2/
/// P4-7 fixed-order-merge argument: order cannot matter at all). The NaN and
/// zero tests are integer comparisons on the raw bits, immune to fast-math.
///
/// One dispatch, one threadgroup: threads grid-stride over the values, then
/// tree-reduce in threadgroup memory; thread 0 writes the winning index as
/// u32. A single threadgroup reads ~600 KB in tens of microseconds — this
/// replaces a ~1.31 ms measured CPU-side readback+scan, not a bandwidth-
/// bound kernel, so multi-threadgroup splitting would be speculative
/// (hard rule: no optimization without measurement).
public final class ArgmaxKernel {
    /// Threadgroup width cap. Must be a power of two (tree reduction) and
    /// bounded by the static `best[]` array in the kernel source.
    static let maxThreadsPerThreadgroup = 1024

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // 64-bit reduction key: high word = monotonic unsigned image of the fp32
    // value, low word = ~index so the LOWER index wins between equal values.
    // NaN/zero classification uses the raw bits (integer compares only —
    // exact under fast-math, which may assume ordered floats).
    inline ulong argmax_key(float v, uint index) {
        uint b = as_type<uint>(v);
        uint magnitude = b & 0x7FFFFFFFu;
        if (magnitude > 0x7F800000u) {
            b = 0xFF800000u;   // NaN -> -inf image (never wins; see header)
        } else if (magnitude == 0u) {
            b = 0u;            // -0.0 -> +0.0 (IEEE > is sign-blind at zero)
        }
        uint mono = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
        return (ulong(mono) << 32) | ulong(~index);
    }

    kernel void argmax_f32(device const float *values [[buffer(0)]],
                           constant uint &count       [[buffer(1)]],
                           device uint *outIndex      [[buffer(2)]],
                           uint tid    [[thread_position_in_threadgroup]],
                           uint tgSize [[threads_per_threadgroup]]) {
        threadgroup ulong best[1024];
        ulong localBest = 0;   // below every real key (min real key > 2^55)
        for (uint i = tid; i < count; i += tgSize) {
            localBest = max(localBest, argmax_key(values[i], i));
        }
        best[tid] = localBest;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = tgSize >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                best[tid] = max(best[tid], best[tid + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            // values[0] NaN latches index 0 in the CPU scan (header note).
            uint firstBits = as_type<uint>(values[0]) & 0x7FFFFFFFu;
            outIndex[0] = firstBits > 0x7F800000u ? 0u : ~uint(best[0]);
        }
    }
    """

    private let pipeline: MTLComputePipelineState
    /// When set, the encoded dispatch increments it at the call site
    /// (P2-5 instrumentation; `GPUModel` attaches its per-step counter).
    var dispatchCounter: DispatchCounter?

    public init(context: MetalContext) throws {
        let library = try context.makeLibrary(source: Self.source)
        pipeline = try context.makeComputePipeline(
            library: library, function: "argmax_f32")
    }

    /// Encodes the reduction: `output[0] = Argmax.firstIndex` of the `count`
    /// fp32 values at the start of `values`, as u32.
    public func encodeArgmax(
        into encoder: MTLComputeCommandEncoder,
        values: MTLBuffer, count: Int, output: MTLBuffer
    ) throws {
        guard count > 0 else {
            throw DecodeKernelError.nonPositiveDimension(name: "count", value: count)
        }
        guard values.length >= count * 4 else {
            throw DecodeKernelError.bufferTooSmall(
                buffer: "values", requiredBytes: count * 4, actualBytes: values.length)
        }
        guard output.length >= 4 else {
            throw DecodeKernelError.bufferTooSmall(
                buffer: "output", requiredBytes: 4, actualBytes: output.length)
        }

        // Largest power of two the pipeline allows, capped by the kernel's
        // static threadgroup array (tree reduction requires a power of two).
        var threads = 1
        while threads * 2 <= min(
            pipeline.maxTotalThreadsPerThreadgroup, Self.maxThreadsPerThreadgroup
        ) {
            threads *= 2
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        var countU32 = UInt32(count)
        encoder.setBytes(&countU32, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        dispatchCounter?.increment()
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }
}
