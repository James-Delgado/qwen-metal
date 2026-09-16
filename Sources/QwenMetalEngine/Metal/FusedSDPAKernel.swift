import Metal

/// P4-2 (docs/phases/phase-4.md D2) reworked by P4-7 (the 2026-09-12
/// addendum): the fused GQA SDPA decode kernel in split-K / two-pass form —
/// for the single query position, QK^T over cache[0..p], softmax, and the PV
/// product with online (streaming) softmax in fp32, no scores/probs buffer
/// ever materialized. GQA mapping (query head → KV head) lives inside the
/// kernels.
///
/// Why two passes (P4-7): the P4-2 kernel parked at one threadgroup per
/// query head — 16 threadgroups at the pinned dims, an occupancy ceiling on
/// a ~6-core device GPU while attention sat ~5.75 ms at window depth vs a
/// ~1 ms byte floor. Flash-decode / MLX `sdpa_vector_2pass` precedent:
///
/// - **Pass 1** (`numHeads × numSplits` threadgroups): positions 0...p
///   partition into `numSplits` contiguous chunks (ceil division). Each
///   threadgroup runs the P4-2 structure over its chunk — simdgroups stride
///   the chunk's positions, each keeping its own online-softmax state (lane
///   ln covers dims {ln, ln+simdWidth, ...}, per-position scores reduce with
///   `simd_sum`), then the per-simdgroup states merge once through
///   threadgroup memory in fixed simdgroup order — and writes one fp32
///   partial state (running max m, denominator l, rescaled numerator
///   acc[headDim]) to the scratch buffers. An empty chunk (shallow depth)
///   writes the m = -inf sentinel.
/// - **Pass 2** (one threadgroup per query head): merges the `numSplits`
///   partial states **in fixed split order** with the same rescale rule
///   (skipping m = -inf sentinels), divides by the merged denominator, and
///   stores fp16. Both merges are fixed-order, so the kernel pair is bitwise
///   deterministic across runs (the P4-2 contract carries).
///
/// Precision per spec D2, unchanged: fp16 cache/query reads, fp32 arithmetic
/// throughout, fp16 output store — the same boundary semantics as the naive
/// chain. p=0 degenerates to weight exactly 1.0 → pass 2 copies the V row
/// bitwise (edge test 1; the P2-3 exactness carries; pass 1 early-outs).
/// Both passes always encode, so the dispatch count is depth-independent:
/// 2 per `encodeSDPA`, measured at the call sites (P2-5 rule).
public final class FusedSDPAKernel {
    /// Threads per threadgroup — 4 simdgroups on Apple GPUs (SIMD width 32).
    static let threadsPerThreadgroup = 128
    /// The pass-1 per-lane register accumulator covers up to 4 dims per
    /// lane at SIMD width 32 → headDim ≤ 128 (the pinned model's headDim).
    static let maxHeadDim = 128
    /// P4-7 split-K factor: positions 0...p split into this many contiguous
    /// chunks, one pass-1 threadgroup each → numHeads × numSplits pass-1
    /// threadgroups (128 at the pinned dims vs 16 at P4-2). A structural
    /// constant, not a tolerance; interpolated into the Metal source so
    /// Swift and MSL cannot drift.
    static let numSplits = 8

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    constant uint MAX_SIMDGROUPS = 8;
    constant uint MAX_DIMS_PER_LANE = 4;
    constant uint MAX_HEAD_DIM = 128;
    constant uint NUM_SPLITS = \(numSplits);

    // P4-7 pass 1: one threadgroup per (query head, position chunk), flat
    // 1-D grid of numHeads·NUM_SPLITS threadgroups (head = idx/NUM_SPLITS,
    // chunk = idx%NUM_SPLITS — MSL forbids mixing a vector grid attribute
    // with the scalar simdgroup attributes). Writes the chunk's fp32
    // online-softmax partial state (partM/partL/partAcc indexed by
    // head·NUM_SPLITS+chunk). Offsets are ELEMENT offsets into the shared
    // KV cache buffer (AttentionKernels convention).
    kernel void sdpa_decode_split_f16(
        device const half *q                [[buffer(0)]],
        device const half *cache            [[buffer(1)]],
        constant ulong &keyBaseElemOffset   [[buffer(2)]],
        constant ulong &valueBaseElemOffset [[buffer(3)]],
        constant uint &groupSize            [[buffer(4)]],
        constant uint &maxContext           [[buffer(5)]],
        constant uint &headDim              [[buffer(6)]],
        constant uint &position             [[buffer(7)]],
        constant float &scale               [[buffer(8)]],
        device float *partM                 [[buffer(9)]],
        device float *partL                 [[buffer(10)]],
        device float *partAcc               [[buffer(11)]],
        uint tgIdx    [[threadgroup_position_in_grid]],
        uint lane     [[thread_index_in_simdgroup]],
        uint sg       [[simdgroup_index_in_threadgroup]],
        uint simdSize [[threads_per_simdgroup]],
        uint tid      [[thread_index_in_threadgroup]],
        uint tptg     [[threads_per_threadgroup]])
    {
        const uint head = tgIdx / NUM_SPLITS;
        const uint split = tgIdx % NUM_SPLITS;
        const uint numSg = tptg / simdSize;

        // p=0: pass 2 copies the V row bitwise; no partial state exists.
        // Uniform branch: `position` is the same for the whole grid.
        if (position == 0) {
            return;
        }

        const uint count = position + 1;
        const uint chunk = (count + NUM_SPLITS - 1) / NUM_SPLITS;
        const uint start = split * chunk;
        const uint end = min(start + chunk, count);
        const uint part = head * NUM_SPLITS + split;

        // Empty chunk (shallow depth): the m = -inf sentinel tells pass 2
        // to skip it; its acc slot is never read.
        if (start >= end) {
            if (tid == 0) {
                partM[part] = -INFINITY;
                partL[part] = 0.0f;
            }
            return;
        }

        const uint kvHead = head / groupSize;
        const ulong kBase = keyBaseElemOffset
            + ulong(kvHead) * maxContext * headDim;
        const ulong vBase = valueBaseElemOffset
            + ulong(kvHead) * maxContext * headDim;

        // This lane's strided query dims (the same lane→dim mapping indexes
        // the accumulator and the V rows).
        float qDims[MAX_DIMS_PER_LANE];
        {
            uint i = 0;
            for (uint d = lane; d < headDim; d += simdSize, ++i) {
                qDims[i] = float(q[ulong(head) * headDim + d]);
            }
        }

        // Per-simdgroup online softmax over the chunk's positions
        // start+sg, start+sg+numSg, ... — the P4-2 loop body verbatim.
        // m: running max; l: running denominator; acc: rescaled numerator.
        float m = -INFINITY;
        float l = 0.0f;
        float acc[MAX_DIMS_PER_LANE] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (uint j = start + sg; j < end; j += numSg) {
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
        // chunk max first, then denominators and numerators rescaled by
        // exp(m_g - mChunk). A simdgroup that processed no position (m
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
        // terms — the P4-2 redundancy pattern); thread t stores acc dim t.
        float mChunk = -INFINITY;
        for (uint g = 0; g < numSg; ++g) {
            mChunk = max(mChunk, tgM[g]);
        }
        float lChunk = 0.0f;
        for (uint g = 0; g < numSg; ++g) {
            if (tgM[g] != -INFINITY) {
                lChunk += exp(tgM[g] - mChunk) * tgL[g];
            }
        }
        if (tid == 0) {
            partM[part] = mChunk;
            partL[part] = lChunk;
        }
        if (tid < headDim) {
            float numerator = 0.0f;
            for (uint g = 0; g < numSg; ++g) {
                if (tgM[g] != -INFINITY) {
                    numerator += exp(tgM[g] - mChunk) * tgAcc[g * headDim + tid];
                }
            }
            partAcc[ulong(part) * headDim + tid] = numerator;
        }
    }

    // P4-7 pass 2: one threadgroup per query head. Merges the NUM_SPLITS
    // partial states in FIXED split order (deterministic), divides by the
    // merged denominator, stores fp16. p=0 is the exact copy branch: the
    // softmax weight is exactly 1.0, so single-position attention IS the V
    // row — a bitwise copy preserves every bit pattern including -0.0 and
    // NaN payloads (edge test 1; the accumulate form's 0 + (-0) would flip
    // -0 to +0, and a float round-trip could disturb the NaN payload).
    kernel void sdpa_decode_reduce_f16(
        device const half *cache            [[buffer(0)]],
        constant ulong &valueBaseElemOffset [[buffer(1)]],
        constant uint &groupSize            [[buffer(2)]],
        constant uint &maxContext           [[buffer(3)]],
        constant uint &headDim              [[buffer(4)]],
        constant uint &position             [[buffer(5)]],
        device const float *partM           [[buffer(6)]],
        device const float *partL           [[buffer(7)]],
        device const float *partAcc         [[buffer(8)]],
        device half *out                    [[buffer(9)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid  [[thread_index_in_threadgroup]])
    {
        if (position == 0) {
            const uint kvHead = head / groupSize;
            const ulong vBase = valueBaseElemOffset
                + ulong(kvHead) * maxContext * headDim;
            if (tid < headDim) {
                out[ulong(head) * headDim + tid] = cache[vBase + tid];
            }
            return;
        }

        // Every thread recomputes the tiny merge redundantly (NUM_SPLITS
        // terms — the P4-2 redundancy pattern); thread t stores output
        // dim t. Empty chunks carry the m = -inf sentinel and contribute
        // nothing.
        const uint base = head * NUM_SPLITS;
        float mTotal = -INFINITY;
        for (uint c = 0; c < NUM_SPLITS; ++c) {
            mTotal = max(mTotal, partM[base + c]);
        }
        float lTotal = 0.0f;
        for (uint c = 0; c < NUM_SPLITS; ++c) {
            if (partM[base + c] != -INFINITY) {
                lTotal += exp(partM[base + c] - mTotal) * partL[base + c];
            }
        }
        if (tid < headDim) {
            float numerator = 0.0f;
            for (uint c = 0; c < NUM_SPLITS; ++c) {
                if (partM[base + c] != -INFINITY) {
                    numerator += exp(partM[base + c] - mTotal)
                        * partAcc[ulong(base + c) * headDim + tid];
                }
            }
            out[ulong(head) * headDim + tid] = half(numerator / lTotal);
        }
    }
    """

    private let splitPipeline: MTLComputePipelineState
    private let reducePipeline: MTLComputePipelineState
    private let device: MTLDevice

    // Pass-1 → pass-2 partial-state scratch (fp32, GPU-private, lazily
    // sized to the largest numHeads·headDim seen and reused — ~66.5 KB at
    // the pinned dims; the serial encoder's hazard tracking orders the
    // passes and the cross-layer reuse within one command buffer).
    private var partM: MTLBuffer?
    private var partL: MTLBuffer?
    private var partAcc: MTLBuffer?

    /// When set, every encoded dispatch increments it at the dispatch call
    /// site (P2-5 instrumentation; `GPUModel` attaches its per-step counter).
    var dispatchCounter: DispatchCounter?

    public init(context: MetalContext) throws {
        let library = try context.makeLibrary(source: Self.source)
        splitPipeline = try context.makeComputePipeline(
            library: library, function: "sdpa_decode_split_f16")
        reducePipeline = try context.makeComputePipeline(
            library: library, function: "sdpa_decode_reduce_f16")
        device = context.device
    }

    /// Encodes the whole attention read for one decode step of `layer`:
    /// output[qHead] = softmax(q[qHead]·K[kvHead, 0...position] / √headDim)
    /// · V[kvHead, 0...position], as TWO dispatches (split partials +
    /// reduce). `query` is `[numHeads, headDim]` fp16 (post QK-norm + RoPE);
    /// `output` is `[numHeads, headDim]` fp16 head-major (feeds the o_proj
    /// matvec directly).
    ///
    /// P5-3 (phase-5.md D4 option 1): `queryByteOffset`/`outputByteOffset`
    /// let the chunked prefill path reuse this kernel per query position,
    /// binding one row of its batched q/attention scratch ([batch,
    /// numHeads·headDim] fp16, engine-owned). They are `setBuffer` offsets —
    /// the recorded element-offset convention exists because SAFETENSORS
    /// data offsets can violate binding alignment, which cannot happen here:
    /// row strides of engine-allocated scratch are multiples of 4 (headDim
    /// is even), matching Apple GPUs' 4-byte minimum buffer-offset
    /// alignment, and the wrapper rejects anything else loudly. Defaults of
    /// 0 keep the decode path binary-identical (kernel source untouched).
    public func encodeSDPA(
        into encoder: MTLComputeCommandEncoder,
        cache: KVCache, layer: Int, position: Int,
        query: MTLBuffer, queryByteOffset: Int = 0,
        numHeads: Int, output: MTLBuffer, outputByteOffset: Int = 0
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
        for (name, offset) in [("queryByteOffset", queryByteOffset),
                               ("outputByteOffset", outputByteOffset)] {
            guard offset >= 0, offset % 4 == 0 else {
                throw QuantKernelError.misalignedOffset(
                    buffer: name, byteOffset: offset, alignment: 4)
            }
        }
        try requireCapacity(
            query, bytes: queryByteOffset + numHeads * cache.headDim * 2,
            name: "query")
        try requireCapacity(
            output, bytes: outputByteOffset + numHeads * cache.headDim * 2,
            name: "output")
        let threads = Self.threadsPerThreadgroup
        // The kernels' fixed register/threadgroup budgets assume the Apple
        // GPU shape (SIMD width ≥ 32, ≥ 128 threads/threadgroup); a device
        // violating them would compute garbage, so refuse loudly instead.
        for pipeline in [splitPipeline, reducePipeline] {
            guard pipeline.maxTotalThreadsPerThreadgroup >= threads,
                  pipeline.threadExecutionWidth * 4 >= Self.maxHeadDim,
                  threads % pipeline.threadExecutionWidth == 0 else {
                throw KVCacheError.fusedKernelUnsupportedDevice(
                    threadExecutionWidth: pipeline.threadExecutionWidth,
                    maxThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup)
            }
        }
        let (m, l, acc) = try scratchBuffers(
            numHeads: numHeads, headDim: cache.headDim)
        // Same formula as the CPU reference's scale, for identical rounding.
        let scale = 1 / Float(cache.headDim).squareRoot()
        let tgSize = MTLSize(width: threads, height: 1, depth: 1)

        // Pass 1: numHeads × numSplits chunk partials.
        encoder.setComputePipelineState(splitPipeline)
        encoder.setBuffer(query, offset: queryByteOffset, index: 0)
        encoder.setBuffer(cache.buffer, offset: 0, index: 1)
        setScalar(encoder, UInt64(keyBase), index: 2)
        setScalar(encoder, UInt64(valueBase), index: 3)
        setScalar(encoder, UInt32(numHeads / cache.kvHeads), index: 4)
        setScalar(encoder, UInt32(cache.maxContext), index: 5)
        setScalar(encoder, UInt32(cache.headDim), index: 6)
        setScalar(encoder, UInt32(position), index: 7)
        setScalar(encoder, scale, index: 8)
        encoder.setBuffer(m, offset: 0, index: 9)
        encoder.setBuffer(l, offset: 0, index: 10)
        encoder.setBuffer(acc, offset: 0, index: 11)
        dispatchCounter?.increment()
        encoder.dispatchThreadgroups(
            MTLSize(width: numHeads * Self.numSplits, height: 1, depth: 1),
            threadsPerThreadgroup: tgSize)

        // Pass 2: per-head fixed-order merge → fp16 output.
        encoder.setComputePipelineState(reducePipeline)
        encoder.setBuffer(cache.buffer, offset: 0, index: 0)
        setScalar(encoder, UInt64(valueBase), index: 1)
        setScalar(encoder, UInt32(numHeads / cache.kvHeads), index: 2)
        setScalar(encoder, UInt32(cache.maxContext), index: 3)
        setScalar(encoder, UInt32(cache.headDim), index: 4)
        setScalar(encoder, UInt32(position), index: 5)
        encoder.setBuffer(m, offset: 0, index: 6)
        encoder.setBuffer(l, offset: 0, index: 7)
        encoder.setBuffer(acc, offset: 0, index: 8)
        encoder.setBuffer(output, offset: outputByteOffset, index: 9)
        dispatchCounter?.increment()
        encoder.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: tgSize)
    }

    /// The partial-state scratch triple, grown to fit and reused across
    /// encodes (GPU-private: never read by the CPU).
    private func scratchBuffers(
        numHeads: Int, headDim: Int
    ) throws -> (m: MTLBuffer, l: MTLBuffer, acc: MTLBuffer) {
        let states = numHeads * Self.numSplits
        let stateBytes = states * 4
        let accBytes = states * headDim * 4
        if let m = partM, let l = partL, let acc = partAcc,
           m.length >= stateBytes, l.length >= stateBytes,
           acc.length >= accBytes {
            return (m, l, acc)
        }
        func makePrivate(bytes: Int) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: bytes, options: .storageModePrivate) else {
                throw MetalHarnessError.bufferAllocationFailed(length: bytes)
            }
            return buffer
        }
        let m = try makePrivate(bytes: max(stateBytes, partM?.length ?? 0))
        let l = try makePrivate(bytes: max(stateBytes, partL?.length ?? 0))
        let acc = try makePrivate(bytes: max(accBytes, partAcc?.length ?? 0))
        partM = m
        partL = l
        partAcc = acc
        return (m, l, acc)
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
