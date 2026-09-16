import Metal

/// P5-3 (docs/phases/phase-5.md D2): the batched prefill kernels — the three
/// pieces of the chunked prefill pipeline that have no per-token equivalent
/// to reuse. Everything else in the prefill chunk is an existing kernel
/// (`QuantGemmKernel` for the projections/MLP, `DecodeKernels` rmsnorm/
/// swiglu/residual with batched row counts, the P4-7 split-K SDPA per query
/// position — spec D4 option 1).
///
/// Packed-pipeline only: tiled prefill exists exactly on the q4g64 fused
/// path (spec D5 — the bf16 backend keeps sequential prefill permanently),
/// so the batched embedding gather has no bf16 variant.
///
/// Arithmetic is copied VERBATIM from the kernels these batch
/// (`embedding_gather_q4_f16`, `qknorm_rope_append_f16`): the only change is
/// a batch grid dimension and row addressing, so the per-element value sets
/// are identical and the pre-committed gates carry — batched gather EXACT,
/// batched cluster at the norm-species constant, v-side append a bitwise
/// copy (Phase 5 gates entry, DECISIONS.md 2026-09-14).
public final class PrefillKernels {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // The pinned q4g64 dequant (Q4G64.dequant) — QuantKernels verbatim.
    inline float dequant_q4(uint code, half scale, half bias) {
        return float(code) * float(scale) + float(bias);
    }
    inline uint q4_code(uint word, uint lane) {
        return (word >> (4 * lane)) & 0xFu;
    }
    inline float bf16_to_f32(ushort w) {
        return as_type<float>(uint(w) << 16);
    }

    // Batched row gather + register dequant + fp16 store: one chunk of
    // token ids -> [batch, hiddenSize] fp16. Per-element arithmetic is
    // embedding_gather_q4_f16 verbatim (one fp32 dequant, one rounding to
    // fp16) => the pre-committed EXACT gate carries row-for-row.
    kernel void prefill_embedding_gather_q4_f16(
        device const uint *q        [[buffer(0)]],
        constant ulong &qElemOffset [[buffer(1)]],
        device const half *scales   [[buffer(2)]],
        constant ulong &scalesElemOffset [[buffer(3)]],
        device const half *biases   [[buffer(4)]],
        constant ulong &biasesElemOffset [[buffer(5)]],
        device const uint *tokenIds [[buffer(6)]],
        constant uint &batch        [[buffer(7)]],
        constant uint &hiddenSize   [[buffer(8)]],
        device half *out            [[buffer(9)]],
        uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= hiddenSize || gid.y >= batch) return;
        uint tokenId = tokenIds[gid.y];
        uint word = q[qElemOffset + ulong(tokenId) * (hiddenSize / 8) + gid.x / 8];
        uint code = q4_code(word, gid.x % 8);
        ulong group = ulong(tokenId) * (hiddenSize / 64) + gid.x / 64;
        out[ulong(gid.y) * hiddenSize + gid.x] = half(dequant_q4(
            code, scales[scalesElemOffset + group], biases[biasesElemOffset + group]));
    }

    // The P4-3 qknorm_rope_append_f16 cluster batched over a chunk: grid z
    // is the chunk position p (absolute cache position basePosition + p).
    // Inputs are the three batched projection outputs (separate buffers —
    // the GEMM writes [batch, rows*headDim] row-major per matrix, unlike
    // the decode matvec3's concatenated qkv vector). Per-(head, pair)
    // arithmetic — norm row sum, fp16 round at the norm/rope boundary,
    // rotation, k/v cache stores — is the single-position kernel verbatim,
    // so the norm-species gate and the v-side bitwise-copy claim carry.
    kernel void prefill_qknorm_rope_append_f16(
        device const half *qIn       [[buffer(0)]],
        device const half *kIn       [[buffer(1)]],
        device const half *vIn       [[buffer(2)]],
        device const ushort *qNorm   [[buffer(3)]],
        constant ulong &qNormElemOffset [[buffer(4)]],
        device const ushort *kNorm   [[buffer(5)]],
        constant ulong &kNormElemOffset [[buffer(6)]],
        device const float *cosTable [[buffer(7)]],
        device const float *sinTable [[buffer(8)]],
        device half *cache           [[buffer(9)]],
        constant ulong &keyBaseElemOffset [[buffer(10)]],
        constant ulong &valueBaseElemOffset [[buffer(11)]],
        constant uint &basePosition  [[buffer(12)]],
        constant uint &batch         [[buffer(13)]],
        constant uint &maxContext    [[buffer(14)]],
        constant uint &headDim       [[buffer(15)]],
        constant uint &numHeads      [[buffer(16)]],
        constant uint &kvHeads       [[buffer(17)]],
        constant float &eps          [[buffer(18)]],
        device half *qOut            [[buffer(19)]],
        uint3 gid [[thread_position_in_grid]]) {
        uint halfDim = headDim / 2;
        if (gid.x >= halfDim || gid.y >= numHeads + 2 * kvHeads
            || gid.z >= batch) return;
        uint position = basePosition + gid.z;
        bool isQ = gid.y < numHeads;
        bool isK = !isQ && gid.y < numHeads + kvHeads;

        if (!isQ && !isK) {
            // V: pure fp16 copy into the cache slot — EXACT (bit patterns
            // preserved, NaN payloads included).
            uint h = gid.y - numHeads - kvHeads;
            ulong inBase = (ulong(gid.z) * kvHeads + h) * headDim;
            ulong slot = valueBaseElemOffset
                + (ulong(h) * maxContext + position) * headDim;
            cache[slot + gid.x] = vIn[inBase + gid.x];
            cache[slot + halfDim + gid.x] = vIn[inBase + halfDim + gid.x];
            return;
        }

        device const half *src = isQ ? qIn : kIn;
        uint rowHeads = isQ ? numHeads : kvHeads;
        uint h = isQ ? gid.y : gid.y - numHeads;
        ulong inBase = (ulong(gid.z) * rowHeads + h) * headDim;

        float sumOfSquares = 0.0f;
        for (uint j = 0; j < headDim; ++j) {
            float v = float(src[inBase + j]);
            sumOfSquares += v * v;
        }
        float inverseRMS = 1.0f / sqrt(sumOfSquares / float(headDim) + eps);
        device const ushort *w = isQ ? qNorm : kNorm;
        ulong wOff = isQ ? qNormElemOffset : kNormElemOffset;
        // fp16 round at the norm/rope boundary, exactly like the unfused
        // rmsnorm_f16 store the rope_f16 kernel then reads.
        float n1 = float(half(bf16_to_f32(w[wOff + gid.x])
            * (float(src[inBase + gid.x]) * inverseRMS)));
        float n2 = float(half(bf16_to_f32(w[wOff + halfDim + gid.x])
            * (float(src[inBase + halfDim + gid.x]) * inverseRMS)));
        float c = cosTable[ulong(position) * halfDim + gid.x];
        float sn = sinTable[ulong(position) * halfDim + gid.x];
        half r1 = half(n1 * c - n2 * sn);
        half r2 = half(n2 * c + n1 * sn);
        if (isQ) {
            ulong outBase = (ulong(gid.z) * numHeads + h) * headDim;
            qOut[outBase + gid.x] = r1;
            qOut[outBase + halfDim + gid.x] = r2;
        } else {
            ulong slot = keyBaseElemOffset
                + (ulong(h) * maxContext + position) * headDim;
            cache[slot + gid.x] = r1;
            cache[slot + halfDim + gid.x] = r2;
        }
    }

    // Copies row `row` of a [*, count] fp16 buffer into a [count] buffer —
    // the last chunk position's residual lands in the decode-path hidden
    // buffer, so the final-norm + lm_head tail and the Tier-E
    // lastLayerOutput hook read the same buffer on both prefill paths.
    // Pure fp16 move: EXACT, bit patterns preserved.
    kernel void prefill_copy_row_f16(device const half *src [[buffer(0)]],
                                     constant uint &row     [[buffer(1)]],
                                     constant uint &count   [[buffer(2)]],
                                     device half *dst       [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
        if (gid >= count) return;
        dst[gid] = src[ulong(row) * count + gid];
    }
    """

    private let gatherPipeline: MTLComputePipelineState
    private let clusterPipeline: MTLComputePipelineState
    private let copyRowPipeline: MTLComputePipelineState

    /// When set, every encoded dispatch increments it at the dispatch call
    /// site (P2-5 instrumentation convention).
    var dispatchCounter: DispatchCounter?

    public init(context: MetalContext) throws {
        let library = try context.makeLibrary(source: Self.source)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            try context.makeComputePipeline(library: library, function: name)
        }
        gatherPipeline = try pipeline("prefill_embedding_gather_q4_f16")
        clusterPipeline = try pipeline("prefill_qknorm_rope_append_f16")
        copyRowPipeline = try pipeline("prefill_copy_row_f16")
    }

    /// Gathers + register-dequants embedding rows for `batch` token ids
    /// (u32, already range-validated by the caller — GPUModel validates the
    /// whole suffix before any dispatch) into `output` ([batch, hiddenSize]
    /// fp16). Triplet conventions match `QuantKernels.encodeEmbeddingGather`.
    public func encodeEmbeddingGatherBatch(
        into encoder: MTLComputeCommandEncoder,
        q: MTLBuffer, qByteOffset: Int,
        scales: MTLBuffer, scalesByteOffset: Int,
        biases: MTLBuffer, biasesByteOffset: Int,
        tokenIds: MTLBuffer, batch: Int, vocabSize: Int, hiddenSize: Int,
        output: MTLBuffer
    ) throws {
        let offsets = try QuantKernels.tripletElementOffsets(
            q: q, qByteOffset: qByteOffset,
            scales: scales, scalesByteOffset: scalesByteOffset,
            biases: biases, biasesByteOffset: biasesByteOffset,
            outDim: vocabSize, inDim: hiddenSize)
        try requirePositive(batch, "batch")
        try QuantKernels.requireCapacity(
            tokenIds, bytes: batch * 4, name: "tokenIds")
        try QuantKernels.requireCapacity(
            output, bytes: batch * hiddenSize * 2, name: "output")

        encoder.setComputePipelineState(gatherPipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        setScalar(encoder, UInt64(offsets.q), index: 1)
        encoder.setBuffer(scales, offset: 0, index: 2)
        setScalar(encoder, UInt64(offsets.scales), index: 3)
        encoder.setBuffer(biases, offset: 0, index: 4)
        setScalar(encoder, UInt64(offsets.biases), index: 5)
        encoder.setBuffer(tokenIds, offset: 0, index: 6)
        setScalar(encoder, UInt32(batch), index: 7)
        setScalar(encoder, UInt32(hiddenSize), index: 8)
        encoder.setBuffer(output, offset: 0, index: 9)
        dispatch2D(encoder, pipeline: gatherPipeline,
                   width: hiddenSize, height: batch)
    }

    /// The batched post-QKV cluster: per-head Q/K RMSNorm + RoPE for chunk
    /// positions `basePosition..<basePosition+batch`, roped q heads to
    /// `qOut` ([batch, numHeads·headDim]), roped k heads into their cache
    /// slots, v heads copied into theirs. Inputs are the batched projection
    /// outputs ([batch, heads·headDim] row-major each). Appending past
    /// `cache.maxContext` throws `.contextFull` BEFORE any dispatch with
    /// the cache untouched (the encodeKVAppend contract).
    public func encodeQKNormRoPEAppendBatch(
        into encoder: MTLComputeCommandEncoder,
        qIn: MTLBuffer, kIn: MTLBuffer, vIn: MTLBuffer,
        qNormWeight: MTLBuffer, qNormByteOffset: Int,
        kNormWeight: MTLBuffer, kNormByteOffset: Int,
        cosTable: MTLBuffer, sinTable: MTLBuffer,
        basePosition: Int, batch: Int, positions: Int, numHeads: Int,
        eps: Float, cache: KVCache, layer: Int, qOut: MTLBuffer
    ) throws {
        try requirePositive(numHeads, "numHeads")
        try requirePositive(batch, "batch")
        let headDim = cache.headDim
        guard headDim % 2 == 0 else {
            throw DecodeKernelError.oddHeadDim(headDim: headDim)
        }
        try requirePositive(positions, "positions")
        guard basePosition >= 0 else {
            throw DecodeKernelError.positionOutOfRange(
                position: basePosition, positions: positions)
        }
        // Context-full FIRST (the encodeKVAppend contract): the whole
        // chunk must fit the preallocated cache — the reported position is
        // the chunk's last requested slot.
        guard basePosition + batch <= cache.maxContext else {
            throw KVCacheError.contextFull(
                position: basePosition + batch - 1, maxContext: cache.maxContext)
        }
        guard basePosition + batch <= positions else {
            throw DecodeKernelError.positionOutOfRange(
                position: basePosition + batch - 1, positions: positions)
        }
        // Validates `layer` via the cache's own bounds checks.
        _ = try cache.elementOffset(
            layer: layer, component: .key, head: 0, position: basePosition)
        let keyBase = try cache.baseElementOffset(layer: layer, component: .key)
        let valueBase = try cache.baseElementOffset(layer: layer, component: .value)

        let kvHeads = cache.kvHeads
        let halfDim = headDim / 2
        let qNormElemOffset = try normWeightElementOffset(
            byteOffset: qNormByteOffset, dim: headDim,
            buffer: qNormWeight, name: "qNormWeight")
        let kNormElemOffset = try normWeightElementOffset(
            byteOffset: kNormByteOffset, dim: headDim,
            buffer: kNormWeight, name: "kNormWeight")
        try QuantKernels.requireCapacity(
            qIn, bytes: batch * numHeads * headDim * 2, name: "qIn")
        try QuantKernels.requireCapacity(
            kIn, bytes: batch * kvHeads * headDim * 2, name: "kIn")
        try QuantKernels.requireCapacity(
            vIn, bytes: batch * kvHeads * headDim * 2, name: "vIn")
        try QuantKernels.requireCapacity(
            qOut, bytes: batch * numHeads * headDim * 2, name: "qOut")
        try QuantKernels.requireCapacity(
            cosTable, bytes: positions * halfDim * 4, name: "cosTable")
        try QuantKernels.requireCapacity(
            sinTable, bytes: positions * halfDim * 4, name: "sinTable")

        encoder.setComputePipelineState(clusterPipeline)
        encoder.setBuffer(qIn, offset: 0, index: 0)
        encoder.setBuffer(kIn, offset: 0, index: 1)
        encoder.setBuffer(vIn, offset: 0, index: 2)
        encoder.setBuffer(qNormWeight, offset: 0, index: 3)
        setScalar(encoder, UInt64(qNormElemOffset), index: 4)
        encoder.setBuffer(kNormWeight, offset: 0, index: 5)
        setScalar(encoder, UInt64(kNormElemOffset), index: 6)
        encoder.setBuffer(cosTable, offset: 0, index: 7)
        encoder.setBuffer(sinTable, offset: 0, index: 8)
        encoder.setBuffer(cache.buffer, offset: 0, index: 9)
        setScalar(encoder, UInt64(keyBase), index: 10)
        setScalar(encoder, UInt64(valueBase), index: 11)
        setScalar(encoder, UInt32(basePosition), index: 12)
        setScalar(encoder, UInt32(batch), index: 13)
        setScalar(encoder, UInt32(cache.maxContext), index: 14)
        setScalar(encoder, UInt32(headDim), index: 15)
        setScalar(encoder, UInt32(numHeads), index: 16)
        setScalar(encoder, UInt32(kvHeads), index: 17)
        setScalar(encoder, eps, index: 18)
        encoder.setBuffer(qOut, offset: 0, index: 19)

        dispatchCounter?.increment()
        let totalHeads = numHeads + 2 * kvHeads
        encoder.dispatchThreads(
            MTLSize(width: halfDim, height: totalHeads, depth: batch),
            threadsPerThreadgroup: threadgroup2DShape(
                clusterPipeline, width: halfDim, height: totalHeads))
    }

    /// Copies row `row` of `source` ([rows, count] fp16) into `output`
    /// ([count] fp16) — a bitwise move (EXACT).
    public func encodeCopyRow(
        into encoder: MTLComputeCommandEncoder,
        source: MTLBuffer, row: Int, count: Int, output: MTLBuffer
    ) throws {
        try requirePositive(count, "count")
        guard row >= 0 else {
            throw DecodeKernelError.nonPositiveDimension(name: "row", value: row)
        }
        try QuantKernels.requireCapacity(
            source, bytes: (row + 1) * count * 2, name: "source")
        try QuantKernels.requireCapacity(
            output, bytes: count * 2, name: "output")

        encoder.setComputePipelineState(copyRowPipeline)
        encoder.setBuffer(source, offset: 0, index: 0)
        setScalar(encoder, UInt32(row), index: 1)
        setScalar(encoder, UInt32(count), index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        dispatchCounter?.increment()
        let width = min(copyRowPipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    // MARK: - Validation + dispatch helpers (DecodeKernels conventions)

    private func requirePositive(_ value: Int, _ name: String) throws {
        guard value > 0 else {
            throw DecodeKernelError.nonPositiveDimension(name: name, value: value)
        }
    }

    /// bf16 norm-weight vectors need even byte offsets (16-bit loads); the
    /// offset is passed to the kernel in ELEMENTS (GPUWeights convention).
    private func normWeightElementOffset(
        byteOffset: Int, dim: Int, buffer: MTLBuffer, name: String
    ) throws -> Int {
        guard byteOffset >= 0, byteOffset % 2 == 0 else {
            throw DecodeKernelError.misalignedWeightOffset(byteOffset: byteOffset)
        }
        try QuantKernels.requireCapacity(
            buffer, bytes: byteOffset + dim * 2, name: name)
        return byteOffset / 2
    }

    private func setScalar<T>(
        _ encoder: MTLComputeCommandEncoder, _ value: T, index: Int
    ) {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<T>.stride, index: index)
    }

    private func dispatch2D(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState, width: Int, height: Int
    ) {
        dispatchCounter?.increment()
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: threadgroup2DShape(
                pipeline, width: width, height: height))
    }

    /// 16×16(×1) threadgroups like the DecodeKernels 2-D helper;
    /// dispatchThreads handles the ragged edge (kernel bounds checks are
    /// belt-and-braces).
    private func threadgroup2DShape(
        _ pipeline: MTLComputePipelineState, width: Int, height: Int
    ) -> MTLSize {
        let side = 16
        let tgWidth = min(side, pipeline.maxTotalThreadsPerThreadgroup)
        let tgHeight = max(1, min(side, pipeline.maxTotalThreadsPerThreadgroup / tgWidth))
        return MTLSize(width: tgWidth, height: tgHeight, depth: 1)
    }
}
