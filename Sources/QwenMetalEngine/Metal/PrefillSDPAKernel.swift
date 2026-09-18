import Metal

/// PF-1 lever 1 (docs/phases/phase-5.md D4 option 2, taken because the
/// measurement said it pays): the BATCHED causal SDPA kernel for the tiled
/// prefill — one dispatch per layer covers every chunk position, replacing
/// the P5-3 loop of 2·C per-position split-K dispatches (47,712 attention
/// dispatches per 852-token prefill; 21.9% of the Mac prefill span at the
/// PF-1 attribution, serialized through the shared partial-state scratch).
///
/// Structure: one threadgroup per (chunk position p, query head) — a flat
/// grid of batch·numHeads threadgroups (8,192 at C=512 and the pinned 16
/// heads, so no split-K is needed for occupancy) — running the P4-2/P4-7
/// per-simdgroup online softmax over cache positions 0...basePosition+p
/// (causality IS the depth limit: position p never reads a later slot),
/// merging the simdgroup states once through threadgroup memory in fixed
/// simdgroup order, and storing fp16 directly. No pass 2, no partial-state
/// scratch: the chunk's attention output buffer the pipeline already owns
/// is the only write target (memory budget unchanged).
///
/// Precision per spec D2, unchanged: fp16 cache/query reads, fp32
/// arithmetic, fp16 store. The inner loop body is the P4-7 pass-1 body
/// VERBATIM; the reduction PARTITION differs (one chunk of depth instead of
/// NUM_SPLITS chunks), so outputs are not bitwise equal to the per-position
/// split-K path — a reduction-order difference of the species the
/// pre-committed attention gate max(2⁻⁷·M, 2⁻¹¹) covers (spec D6: "either
/// D4 form" gates at the attention constant; the 2026-09-12 decision).
/// Depth 1 (position 0) is the exact V-row copy (edge test 1 exactness
/// carries). Fixed-order merge ⇒ bitwise deterministic across runs.
public final class PrefillSDPAKernel {
    /// Threads per threadgroup — 4 simdgroups on Apple GPUs (SIMD width 32).
    static let threadsPerThreadgroup = 128
    /// Per-lane register accumulator covers 4 dims per lane at SIMD width
    /// 32 → headDim ≤ 128 (the pinned model's headDim).
    static let maxHeadDim = 128

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    constant uint MAX_SIMDGROUPS = 8;
    constant uint MAX_DIMS_PER_LANE = 4;
    constant uint MAX_HEAD_DIM = 128;

    // One threadgroup per (chunk position p, query head): flat 1-D grid of
    // batch·numHeads threadgroups (p = idx / numHeads, head = idx % numHeads
    // — MSL forbids mixing a vector grid attribute with the scalar
    // simdgroup attributes). Query rows come from the batched roped-q
    // scratch [batch, numHeads·headDim]; outputs go to the batched
    // attention scratch with the same layout. Offsets are ELEMENT offsets
    // into the shared KV cache buffer (AttentionKernels convention).
    kernel void sdpa_prefill_causal_f16(
        device const half *q                [[buffer(0)]],
        device const half *cache            [[buffer(1)]],
        constant ulong &keyBaseElemOffset   [[buffer(2)]],
        constant ulong &valueBaseElemOffset [[buffer(3)]],
        constant uint &groupSize            [[buffer(4)]],
        constant uint &maxContext           [[buffer(5)]],
        constant uint &headDim              [[buffer(6)]],
        constant uint &basePosition         [[buffer(7)]],
        constant uint &numHeads             [[buffer(8)]],
        constant float &scale               [[buffer(9)]],
        device half *out                    [[buffer(10)]],
        uint tgIdx    [[threadgroup_position_in_grid]],
        uint lane     [[thread_index_in_simdgroup]],
        uint sg       [[simdgroup_index_in_threadgroup]],
        uint simdSize [[threads_per_simdgroup]],
        uint tid      [[thread_index_in_threadgroup]],
        uint tptg     [[threads_per_threadgroup]])
    {
        const uint p = tgIdx / numHeads;
        const uint head = tgIdx % numHeads;
        const uint numSg = tptg / simdSize;
        const uint position = basePosition + p;
        const uint count = position + 1;

        const uint kvHead = head / groupSize;
        const ulong kBase = keyBaseElemOffset
            + ulong(kvHead) * maxContext * headDim;
        const ulong vBase = valueBaseElemOffset
            + ulong(kvHead) * maxContext * headDim;
        const ulong qRow = ulong(p) * numHeads * headDim + ulong(head) * headDim;
        const ulong outRow = qRow;

        // Depth 1: the softmax weight is exactly 1.0, so attention IS the V
        // row — a bitwise copy preserves every bit pattern (-0.0, NaN
        // payloads); the accumulate form would flip -0 to +0. Uniform
        // branch (position is per-threadgroup), so the early return is
        // barrier-safe.
        if (count == 1) {
            if (tid < headDim) {
                out[outRow + tid] = cache[vBase + tid];
            }
            return;
        }

        // This lane's strided query dims (the same lane→dim mapping indexes
        // the accumulator and the V rows).
        float qDims[MAX_DIMS_PER_LANE];
        {
            uint i = 0;
            for (uint d = lane; d < headDim; d += simdSize, ++i) {
                qDims[i] = float(q[qRow + d]);
            }
        }

        // Per-simdgroup online softmax over positions sg, sg+numSg, ... <
        // count — the P4-7 pass-1 loop body verbatim (start = 0, end =
        // count). m: running max; l: denominator; acc: rescaled numerator.
        float m = -INFINITY;
        float l = 0.0f;
        float acc[MAX_DIMS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint j = sg; j < count; j += numSg) {
            const ulong kRow = kBase + ulong(j) * headDim;
            float partial = 0.0f;
            uint i = 0;
            for (uint d = lane; d < headDim; d += simdSize, ++i) {
                partial += qDims[i] * float(cache[kRow + d]);
            }
            const float s = simd_sum(partial) * scale;
            const float mNew = max(m, s);
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

        // Fixed-order simdgroup merge (deterministic): max first, then
        // denominators and numerators rescaled by exp(m_g - mAll). A
        // simdgroup that processed no position (m still -inf) contributes
        // nothing.
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

        float mAll = -INFINITY;
        for (uint g = 0; g < numSg; ++g) {
            mAll = max(mAll, tgM[g]);
        }
        float lAll = 0.0f;
        for (uint g = 0; g < numSg; ++g) {
            if (tgM[g] != -INFINITY) {
                lAll += exp(tgM[g] - mAll) * tgL[g];
            }
        }
        if (tid < headDim) {
            float numerator = 0.0f;
            for (uint g = 0; g < numSg; ++g) {
                if (tgM[g] != -INFINITY) {
                    numerator += exp(tgM[g] - mAll) * tgAcc[g * headDim + tid];
                }
            }
            out[outRow + tid] = half(numerator / lAll);
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
            library: library, function: "sdpa_prefill_causal_f16")
    }

    /// Encodes causal attention for `batch` chunk positions of `layer` in
    /// ONE dispatch: for p in 0..<batch, output row p = softmax(q_p ·
    /// K[kvHead, 0...basePosition+p] / √headDim) · V[kvHead, 0...basePosition+p]
    /// per query head. `query` and `output` are `[batch, numHeads·headDim]`
    /// fp16 row-major (the batched roped-q and attention scratch). The
    /// chunk's K/V must already be appended for positions
    /// basePosition..<basePosition+batch (the cluster kernel runs first).
    public func encodeCausalSDPABatch(
        into encoder: MTLComputeCommandEncoder,
        cache: KVCache, layer: Int, basePosition: Int, batch: Int,
        query: MTLBuffer, numHeads: Int, output: MTLBuffer
    ) throws {
        try requirePositive(numHeads, "numHeads")
        try requirePositive(batch, "batch")
        guard basePosition >= 0 else {
            throw DecodeKernelError.nonPositiveDimension(
                name: "basePosition", value: basePosition)
        }
        guard numHeads % cache.kvHeads == 0 else {
            throw KVCacheError.gqaMismatch(numHeads: numHeads, kvHeads: cache.kvHeads)
        }
        guard cache.headDim <= Self.maxHeadDim else {
            throw KVCacheError.headDimExceedsFusedLimit(
                headDim: cache.headDim, limit: Self.maxHeadDim)
        }
        // Validates (layer, last position) via the cache's own bounds
        // checks; a chunk reaching maxContext throws BEFORE any dispatch.
        _ = try cache.elementOffset(
            layer: layer, component: .key, head: 0,
            position: basePosition + batch - 1)
        let keyBase = try cache.baseElementOffset(layer: layer, component: .key)
        let valueBase = try cache.baseElementOffset(layer: layer, component: .value)
        let rowBytes = numHeads * cache.headDim * 2
        try requireCapacity(query, bytes: batch * rowBytes, name: "query")
        try requireCapacity(output, bytes: batch * rowBytes, name: "output")
        let threads = Self.threadsPerThreadgroup
        // The kernel's fixed register/threadgroup budgets assume the Apple
        // GPU shape (SIMD width ≥ 32, ≥ 128 threads/threadgroup); refuse
        // loudly on anything else instead of computing garbage.
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
        setScalar(encoder, UInt32(basePosition), index: 7)
        setScalar(encoder, UInt32(numHeads), index: 8)
        setScalar(encoder, scale, index: 9)
        encoder.setBuffer(output, offset: 0, index: 10)
        dispatchCounter?.increment()
        encoder.dispatchThreadgroups(
            MTLSize(width: batch * numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    // MARK: - Validation helpers (FusedSDPAKernel conventions)

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
