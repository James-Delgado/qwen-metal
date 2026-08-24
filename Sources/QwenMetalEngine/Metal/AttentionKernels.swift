import Metal

/// P2-3 (docs/phases/phase-2.md D4): the naive attention kernels — kv-append,
/// attn-scores, softmax, attn-pv — over the preallocated `KVCache`. All
/// one-thread-per-output-element, deliberately unoptimized (hard rule 3:
/// correctness before any optimization; speed is Phases 3-5's job).
///
/// Precision per spec D2: KV cache and activations fp16, scores/probabilities
/// fp32 (max-subtracted softmax in fp32), fp32 accumulation everywhere.
/// GQA per the pinned family: `groupSize = numHeads / kvHeads` consecutive
/// Q heads share one KV head (KV head h serves Q heads {2h, 2h+1} at the
/// model's 16/8 split) — computed host-side, validated, passed to kernels.
///
/// Methods ENCODE into a caller-supplied encoder (DecodeKernels convention):
/// P2-4 packs one command buffer per decoded token, and the Tier-K tests
/// drive the same methods through `MetalContext.timedDispatch` (hard rule 7).
public final class AttentionKernels {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // Copies one token's [kvHeads, headDim] fp16 vectors into the cache slot
    // at `position` (head-major slab: each head's positions are contiguous).
    // Pure fp16 copy, no arithmetic => pre-committed EXACT gate.
    kernel void kv_append_f16(device const half *input   [[buffer(0)]],
                              device half *cache         [[buffer(1)]],
                              constant ulong &baseElemOffset [[buffer(2)]],
                              constant uint &kvHeads     [[buffer(3)]],
                              constant uint &maxContext  [[buffer(4)]],
                              constant uint &headDim     [[buffer(5)]],
                              constant uint &position    [[buffer(6)]],
                              uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= headDim || gid.y >= kvHeads) return;
        ulong slot = baseElemOffset
            + (ulong(gid.y) * maxContext + position) * headDim + gid.x;
        cache[slot] = input[ulong(gid.y) * headDim + gid.x];
    }

    // scores[qHead, j] = scale * q[qHead] · K[qHead/groupSize, j] for
    // j in 0...position, fp32 accumulation. One thread per (position, qHead).
    // The scores buffer is [numHeads, maxContext] fp32, row stride maxContext;
    // columns beyond `position` are left untouched.
    kernel void attn_scores_f16(device const half *q      [[buffer(0)]],
                                device const half *cache  [[buffer(1)]],
                                constant ulong &keyBaseElemOffset [[buffer(2)]],
                                constant uint &groupSize  [[buffer(3)]],
                                constant uint &maxContext [[buffer(4)]],
                                constant uint &headDim    [[buffer(5)]],
                                constant uint &position   [[buffer(6)]],
                                constant uint &numHeads   [[buffer(7)]],
                                constant float &scale     [[buffer(8)]],
                                device float *scores      [[buffer(9)]],
                                uint2 gid [[thread_position_in_grid]]) {
        if (gid.x > position || gid.y >= numHeads) return;
        uint kvHead = gid.y / groupSize;
        ulong kBase = keyBaseElemOffset
            + (ulong(kvHead) * maxContext + gid.x) * headDim;
        ulong qBase = ulong(gid.y) * headDim;
        float acc = 0.0f;
        for (uint d = 0; d < headDim; ++d) {
            acc += float(q[qBase + d]) * float(cache[kBase + d]);
        }
        scores[ulong(gid.y) * maxContext + gid.x] = acc * scale;
    }

    // Max-subtracted softmax over `rows` rows of `count` fp32 values at
    // `rowStride`, out of place (in-place would race: every thread reads its
    // whole row). Each thread recomputes its row's max and sum sequentially
    // in the same order — the rmsnorm_f16 naive-redundancy pattern.
    kernel void softmax_rows_f32(device const float *x   [[buffer(0)]],
                                 constant uint &rows     [[buffer(1)]],
                                 constant uint &count    [[buffer(2)]],
                                 constant uint &rowStride [[buffer(3)]],
                                 device float *out       [[buffer(4)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= count || gid.y >= rows) return;
        ulong base = ulong(gid.y) * rowStride;
        float rowMax = -INFINITY;
        for (uint j = 0; j < count; ++j) {
            rowMax = max(rowMax, x[base + j]);
        }
        float sum = 0.0f;
        for (uint j = 0; j < count; ++j) {
            sum += exp(x[base + j] - rowMax);
        }
        out[base + gid.x] = exp(x[base + gid.x] - rowMax) / sum;
    }

    // out[qHead, d] = sum_{j=0...position} probs[qHead, j] *
    // V[qHead/groupSize, j, d], fp32 accumulation, fp16 store. One thread
    // per (d, qHead); probs is [numHeads, maxContext] fp32 like scores.
    kernel void attn_pv_f16(device const float *probs   [[buffer(0)]],
                            device const half *cache    [[buffer(1)]],
                            constant ulong &valueBaseElemOffset [[buffer(2)]],
                            constant uint &groupSize    [[buffer(3)]],
                            constant uint &maxContext   [[buffer(4)]],
                            constant uint &headDim      [[buffer(5)]],
                            constant uint &position     [[buffer(6)]],
                            constant uint &numHeads     [[buffer(7)]],
                            device half *out            [[buffer(8)]],
                            uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= headDim || gid.y >= numHeads) return;
        uint kvHead = gid.y / groupSize;
        ulong vBase = valueBaseElemOffset
            + ulong(kvHead) * maxContext * headDim + gid.x;
        ulong pBase = ulong(gid.y) * maxContext;
        float acc = 0.0f;
        for (uint j = 0; j <= position; ++j) {
            acc += probs[pBase + j] * float(cache[vBase + ulong(j) * headDim]);
        }
        out[ulong(gid.y) * headDim + gid.x] = half(acc);
    }
    """

    private let appendPipeline: MTLComputePipelineState
    private let scoresPipeline: MTLComputePipelineState
    private let softmaxPipeline: MTLComputePipelineState
    private let pvPipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        let library = try context.makeLibrary(source: Self.source)
        appendPipeline = try context.makeComputePipeline(
            library: library, function: "kv_append_f16")
        scoresPipeline = try context.makeComputePipeline(
            library: library, function: "attn_scores_f16")
        softmaxPipeline = try context.makeComputePipeline(
            library: library, function: "softmax_rows_f32")
        pvPipeline = try context.makeComputePipeline(
            library: library, function: "attn_pv_f16")
    }

    // MARK: - Encode methods (one per kernel)

    /// Appends one token's `[kvHeads, headDim]` fp16 vector into
    /// `(layer, component)` at `position`. Appending at `position ==
    /// maxContext` throws `.contextFull` BEFORE any dispatch — the clean
    /// context-limit stop; an out-of-bounds write is impossible.
    public func encodeKVAppend(
        into encoder: MTLComputeCommandEncoder,
        cache: KVCache, layer: Int, component: KVCache.Component,
        position: Int, vector: MTLBuffer
    ) throws {
        guard position < cache.maxContext else {
            throw KVCacheError.contextFull(
                position: position, maxContext: cache.maxContext)
        }
        let base = try baseOffsetValidatingPosition(
            cache: cache, layer: layer, component: component, position: position)
        try requireCapacity(
            vector, bytes: cache.kvHeads * cache.headDim * 2, name: "vector")

        encoder.setComputePipelineState(appendPipeline)
        encoder.setBuffer(vector, offset: 0, index: 0)
        encoder.setBuffer(cache.buffer, offset: 0, index: 1)
        setScalar(encoder, UInt64(base), index: 2)
        setScalar(encoder, UInt32(cache.kvHeads), index: 3)
        setScalar(encoder, UInt32(cache.maxContext), index: 4)
        setScalar(encoder, UInt32(cache.headDim), index: 5)
        setScalar(encoder, UInt32(position), index: 6)
        dispatch2D(encoder, pipeline: appendPipeline, width: cache.headDim, height: cache.kvHeads)
    }

    /// scores[qHead, 0...position] = q[qHead]·K[kvHead, j] / √headDim over the
    /// cached keys of `layer`. `query` is `[numHeads, headDim]` fp16 (post
    /// QK-norm + RoPE); `scores` is `[numHeads, maxContext]` fp32.
    public func encodeAttentionScores(
        into encoder: MTLComputeCommandEncoder,
        cache: KVCache, layer: Int, position: Int,
        query: MTLBuffer, numHeads: Int, scores: MTLBuffer
    ) throws {
        let groupSize = try requireGQA(numHeads: numHeads, cache: cache)
        let base = try baseOffsetValidatingPosition(
            cache: cache, layer: layer, component: .key, position: position)
        try requireCapacity(
            query, bytes: numHeads * cache.headDim * 2, name: "query")
        try requireCapacity(
            scores, bytes: numHeads * cache.maxContext * 4, name: "scores")
        // Same formula as the CPU reference's scale, for identical rounding.
        let scale = 1 / Float(cache.headDim).squareRoot()

        encoder.setComputePipelineState(scoresPipeline)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(cache.buffer, offset: 0, index: 1)
        setScalar(encoder, UInt64(base), index: 2)
        setScalar(encoder, UInt32(groupSize), index: 3)
        setScalar(encoder, UInt32(cache.maxContext), index: 4)
        setScalar(encoder, UInt32(cache.headDim), index: 5)
        setScalar(encoder, UInt32(position), index: 6)
        setScalar(encoder, UInt32(numHeads), index: 7)
        setScalar(encoder, scale, index: 8)
        encoder.setBuffer(scores, offset: 0, index: 9)
        dispatch2D(encoder, pipeline: scoresPipeline, width: position + 1, height: numHeads)
    }

    /// Max-subtracted fp32 softmax over `rows` rows of `count` values at
    /// `rowStride`, `input` → `output` (out of place). For attention:
    /// rows = numHeads, count = position + 1, rowStride = maxContext.
    public func encodeSoftmaxRows(
        into encoder: MTLComputeCommandEncoder,
        input: MTLBuffer, rows: Int, count: Int, rowStride: Int,
        output: MTLBuffer
    ) throws {
        try requirePositive(rows, "rows")
        try requirePositive(count, "count")
        guard count <= rowStride else {
            throw KVCacheError.rowCountExceedsStride(count: count, rowStride: rowStride)
        }
        try requireCapacity(input, bytes: rows * rowStride * 4, name: "input")
        try requireCapacity(output, bytes: rows * rowStride * 4, name: "output")

        encoder.setComputePipelineState(softmaxPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        setScalar(encoder, UInt32(rows), index: 1)
        setScalar(encoder, UInt32(count), index: 2)
        setScalar(encoder, UInt32(rowStride), index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        dispatch2D(encoder, pipeline: softmaxPipeline, width: count, height: rows)
    }

    /// output[qHead] = Σ_j probs[qHead, j] · V[kvHead, j] over the cached
    /// values of `layer`, j in 0...position. `probs` is `[numHeads,
    /// maxContext]` fp32; `output` is `[numHeads, headDim]` fp16 head-major
    /// (feeds the o_proj matvec directly).
    public func encodeAttentionPV(
        into encoder: MTLComputeCommandEncoder,
        cache: KVCache, layer: Int, position: Int,
        probs: MTLBuffer, numHeads: Int, output: MTLBuffer
    ) throws {
        let groupSize = try requireGQA(numHeads: numHeads, cache: cache)
        let base = try baseOffsetValidatingPosition(
            cache: cache, layer: layer, component: .value, position: position)
        try requireCapacity(
            probs, bytes: numHeads * cache.maxContext * 4, name: "probs")
        try requireCapacity(
            output, bytes: numHeads * cache.headDim * 2, name: "output")

        encoder.setComputePipelineState(pvPipeline)
        encoder.setBuffer(probs, offset: 0, index: 0)
        encoder.setBuffer(cache.buffer, offset: 0, index: 1)
        setScalar(encoder, UInt64(base), index: 2)
        setScalar(encoder, UInt32(groupSize), index: 3)
        setScalar(encoder, UInt32(cache.maxContext), index: 4)
        setScalar(encoder, UInt32(cache.headDim), index: 5)
        setScalar(encoder, UInt32(position), index: 6)
        setScalar(encoder, UInt32(numHeads), index: 7)
        encoder.setBuffer(output, offset: 0, index: 8)
        dispatch2D(encoder, pipeline: pvPipeline, width: cache.headDim, height: numHeads)
    }

    // MARK: - Validation + dispatch helpers (DecodeKernels conventions)

    private func requireGQA(numHeads: Int, cache: KVCache) throws -> Int {
        try requirePositive(numHeads, "numHeads")
        guard numHeads % cache.kvHeads == 0 else {
            throw KVCacheError.gqaMismatch(numHeads: numHeads, kvHeads: cache.kvHeads)
        }
        return numHeads / cache.kvHeads
    }

    /// Validates `(layer, position)` via the cache's own bounds checks and
    /// returns the slab's base element offset.
    private func baseOffsetValidatingPosition(
        cache: KVCache, layer: Int, component: KVCache.Component, position: Int
    ) throws -> Int {
        _ = try cache.elementOffset(
            layer: layer, component: component, head: 0, position: position)
        return try cache.baseElementOffset(layer: layer, component: component)
    }

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

    private func dispatch2D(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState, width: Int, height: Int
    ) {
        // 16×16 threadgroups like DecodeKernels; dispatchThreads handles the
        // ragged edge (kernel bounds checks are belt-and-braces).
        let side = 16
        let tgWidth = min(side, pipeline.maxTotalThreadsPerThreadgroup)
        let tgHeight = max(1, min(side, pipeline.maxTotalThreadsPerThreadgroup / tgWidth))
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: tgHeight, depth: 1))
    }
}
