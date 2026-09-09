import Metal

/// P4-2 (docs/phases/phase-4.md D2): the fused GQA SDPA decode kernel — for
/// the single query position, QK^T over cache[0..p], softmax, and the PV
/// product in ONE dispatch per layer, replacing the naive three-kernel
/// scores → softmax → PV chain. Online (streaming) softmax in fp32: running
/// max + running denominator + rescaled accumulator, so no scores/probs
/// buffer is ever materialized (the Phase 2 256 KB buffers disappear on the
/// fused path). GQA mapping (query head → KV head) lives inside the kernel.
///
/// Parallelization (spec D2, task's choice): one threadgroup per query head;
/// simdgroups stride the cache positions (simdgroup g handles j = g, g+N,
/// ...), each keeping its own online-softmax state; within a simdgroup, lane
/// ln covers dims {ln, ln+simdWidth, ...} and per-position scores reduce
/// with `simd_sum` — broadcast to every lane with no threadgroup barrier in
/// the position loop. The per-simdgroup states merge once at the end through
/// threadgroup memory with the same rescale rule, in fixed simdgroup order,
/// so the kernel is bitwise deterministic across runs.
///
/// Precision per spec D2: fp16 cache/query reads, fp32 arithmetic
/// throughout, fp16 output store — the same boundary semantics as the naive
/// chain. p=0 degenerates to weight exactly 1.0 → output == V row bitwise
/// (edge test 1; the P2-3 exactness carries).
public final class FusedSDPAKernel {
    /// Threads per threadgroup — 4 simdgroups on Apple GPUs (SIMD width 32).
    static let threadsPerThreadgroup = 128
    /// The kernel's per-lane register accumulator covers up to 4 dims per
    /// lane at SIMD width 32 → headDim ≤ 128 (the pinned model's headDim).
    static let maxHeadDim = 128

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    constant uint MAX_SIMDGROUPS = 8;
    constant uint MAX_DIMS_PER_LANE = 4;
    constant uint MAX_HEAD_DIM = 128;

    // One threadgroup per query head. Offsets are ELEMENT offsets into the
    // shared KV cache buffer (AttentionKernels convention).
    kernel void sdpa_decode_f16(
        device const half *q                [[buffer(0)]],
        device const half *cache            [[buffer(1)]],
        constant ulong &keyBaseElemOffset   [[buffer(2)]],
        constant ulong &valueBaseElemOffset [[buffer(3)]],
        constant uint &groupSize            [[buffer(4)]],
        constant uint &maxContext           [[buffer(5)]],
        constant uint &headDim              [[buffer(6)]],
        constant uint &position             [[buffer(7)]],
        constant float &scale               [[buffer(8)]],
        device half *out                    [[buffer(9)]],
        uint head     [[threadgroup_position_in_grid]],
        uint lane     [[thread_index_in_simdgroup]],
        uint sg       [[simdgroup_index_in_threadgroup]],
        uint simdSize [[threads_per_simdgroup]],
        uint tid      [[thread_index_in_threadgroup]],
        uint tptg     [[threads_per_threadgroup]])
    {
        const uint numSg = tptg / simdSize;
        const uint kvHead = head / groupSize;
        const ulong kBase = keyBaseElemOffset
            + ulong(kvHead) * maxContext * headDim;
        const ulong vBase = valueBaseElemOffset
            + ulong(kvHead) * maxContext * headDim;

        // p=0: the softmax weight is exactly 1.0, so single-position
        // attention IS the V row — copy it (edge test 1's bitwise gate;
        // the accumulate form's 0 + (-0) would flip -0 to +0). Uniform
        // branch: `position` is the same for the whole threadgroup.
        if (position == 0) {
            if (tid < headDim) {
                out[ulong(head) * headDim + tid] = cache[vBase + tid];
            }
            return;
        }

        // This lane's strided query dims (the same lane→dim mapping indexes
        // the accumulator and the V rows).
        float qDims[MAX_DIMS_PER_LANE];
        {
            uint i = 0;
            for (uint d = lane; d < headDim; d += simdSize, ++i) {
                qDims[i] = float(q[ulong(head) * headDim + d]);
            }
        }

        // Per-simdgroup online softmax over positions sg, sg+numSg, ...
        // m: running max; l: running denominator; acc: rescaled numerator.
        float m = -INFINITY;
        float l = 0.0f;
        float acc[MAX_DIMS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint j = sg; j <= position; j += numSg) {
            const ulong kRow = kBase + ulong(j) * headDim;
            float partial = 0.0f;
            uint i = 0;
            for (uint d = lane; d < headDim; d += simdSize, ++i) {
                partial += qDims[i] * float(cache[kRow + d]);
            }
            const float s = simd_sum(partial) * scale;
            const float mNew = max(m, s);
            // First position of this simdgroup: m is -inf; the guard (not
            // exp(-inf), which fast math does not promise) zeroes the
            // stale-state correction.
            const float corr = (m == -INFINITY) ? 0.0f : exp(m - mNew);
            const float w = exp(s - mNew);
            const ulong vRow = vBase + ulong(j) * headDim;
            i = 0;
            for (uint d = lane; d < headDim; d += simdSize, ++i) {
                acc[i] = acc[i] * corr + w * float(cache[vRow + d]);
            }
            l = l * corr + w;
            m = mNew;
        }

        // Merge the per-simdgroup states in fixed order (deterministic):
        // global max first, then denominators and numerators rescaled by
        // exp(m_g - mTotal). A simdgroup that processed no position (m
        // still -inf) contributes nothing.
        threadgroup float tgM[MAX_SIMDGROUPS];
        threadgroup float tgL[MAX_SIMDGROUPS];
        threadgroup float tgAcc[MAX_SIMDGROUPS * MAX_HEAD_DIM];
        if (lane == 0) {
            tgM[sg] = m;
            tgL[sg] = l;
        }
        {
            uint i = 0;
            for (uint d = lane; d < headDim; d += simdSize, ++i) {
                tgAcc[sg * headDim + d] = acc[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Every thread recomputes the tiny merge redundantly (numSg ≤ 8
        // terms — the rmsnorm naive-redundancy pattern); thread t stores
        // output dim t.
        float mTotal = -INFINITY;
        for (uint g = 0; g < numSg; ++g) {
            mTotal = max(mTotal, tgM[g]);
        }
        float lTotal = 0.0f;
        for (uint g = 0; g < numSg; ++g) {
            if (tgM[g] != -INFINITY) {
                lTotal += exp(tgM[g] - mTotal) * tgL[g];
            }
        }
        if (tid < headDim) {
            float numerator = 0.0f;
            for (uint g = 0; g < numSg; ++g) {
                if (tgM[g] != -INFINITY) {
                    numerator += exp(tgM[g] - mTotal) * tgAcc[g * headDim + tid];
                }
            }
            out[ulong(head) * headDim + tid] = half(numerator / lTotal);
        }
    }
    """

    private let pipeline: MTLComputePipelineState

    /// When set, every encoded dispatch increments it at the dispatch call
    /// site (P2-5 instrumentation; `GPUModel` attaches its per-step counter).
    var dispatchCounter: DispatchCounter?

    public init(context: MetalContext) throws {
        let library = try context.makeLibrary(source: Self.source)
        pipeline = try context.makeComputePipeline(
            library: library, function: "sdpa_decode_f16")
    }

    /// Encodes the whole attention read for one decode step of `layer`:
    /// output[qHead] = softmax(q[qHead]·K[kvHead, 0...position] / √headDim)
    /// · V[kvHead, 0...position], one dispatch. `query` is `[numHeads,
    /// headDim]` fp16 (post QK-norm + RoPE); `output` is `[numHeads,
    /// headDim]` fp16 head-major (feeds the o_proj matvec directly).
    public func encodeSDPA(
        into encoder: MTLComputeCommandEncoder,
        cache: KVCache, layer: Int, position: Int,
        query: MTLBuffer, numHeads: Int, output: MTLBuffer
    ) throws {
        try requirePositive(numHeads, "numHeads")
        guard numHeads % cache.kvHeads == 0 else {
            throw KVCacheError.gqaMismatch(numHeads: numHeads, kvHeads: cache.kvHeads)
        }
        guard cache.headDim <= Self.maxHeadDim else {
            throw KVCacheError.headDimExceedsFusedLimit(
                headDim: cache.headDim, limit: Self.maxHeadDim)
        }
        // Validates (layer, position) via the cache's own bounds checks; an
        // encode at position == maxContext throws BEFORE any dispatch.
        _ = try cache.elementOffset(
            layer: layer, component: .key, head: 0, position: position)
        let keyBase = try cache.baseElementOffset(layer: layer, component: .key)
        let valueBase = try cache.baseElementOffset(layer: layer, component: .value)
        try requireCapacity(
            query, bytes: numHeads * cache.headDim * 2, name: "query")
        try requireCapacity(
            output, bytes: numHeads * cache.headDim * 2, name: "output")
        let threads = Self.threadsPerThreadgroup
        // The kernel's fixed register/threadgroup budgets assume the Apple
        // GPU shape (SIMD width ≥ 32, ≥ 128 threads/threadgroup); a device
        // violating them would compute garbage, so refuse loudly instead.
        guard pipeline.maxTotalThreadsPerThreadgroup >= threads,
              pipeline.threadExecutionWidth * 4 >= Self.maxHeadDim,
              threads % pipeline.threadExecutionWidth == 0 else {
            throw KVCacheError.fusedKernelUnsupportedDevice(
                threadExecutionWidth: pipeline.threadExecutionWidth,
                maxThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup)
        }
        // Same formula as the CPU reference's scale, for identical rounding.
        let scale = 1 / Float(cache.headDim).squareRoot()

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(cache.buffer, offset: 0, index: 1)
        setScalar(encoder, UInt64(keyBase), index: 2)
        setScalar(encoder, UInt64(valueBase), index: 3)
        setScalar(encoder, UInt32(numHeads / cache.kvHeads), index: 4)
        setScalar(encoder, UInt32(cache.maxContext), index: 5)
        setScalar(encoder, UInt32(cache.headDim), index: 6)
        setScalar(encoder, UInt32(position), index: 7)
        setScalar(encoder, scale, index: 8)
        encoder.setBuffer(output, offset: 0, index: 9)
        dispatchCounter?.increment()
        encoder.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    // MARK: - Validation helpers (AttentionKernels conventions)

    private func requirePositive(_ value: Int, _ name: String) throws {
        guard value > 0 else {
            throw DecodeKernelError.nonPositiveDimension(name: name, value: value)
        }
    }

    private func requireCapacity(
        _ buffer: MTLBuffer, bytes: Int, name: String
    ) throws {
        guard buffer.length >= bytes else {
            throw DecodeKernelError.bufferTooSmall(
                buffer: name, requiredBytes: bytes, actualBytes: buffer.length)
        }
    }

    private func setScalar<T>(
        _ encoder: MTLComputeCommandEncoder, _ value: T, index: Int
    ) {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<T>.stride, index: index)
    }
}
