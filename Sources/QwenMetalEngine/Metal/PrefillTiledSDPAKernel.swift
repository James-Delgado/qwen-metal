import Metal

/// The chunk-level causal SDPA contract the tiled prefill pipeline encodes
/// against (phase-5.md D4): one dispatch per layer per chunk, query rows
/// `[batch, numHeads·headDim]` fp16 in, attention rows of the same shape
/// out, K/V read from the shared cache for positions 0...basePosition+p.
/// Two kernels implement it — the PF-1 per-(position, head) batched kernel
/// (`PrefillSDPAKernel`) and the PF-2 query-tiled kernel below — so the
/// pipeline switches by `GPUModel.PrefillAttention` for the on-device A/B
/// (the P4 D4 / P5 D5 toggle precedent).
protocol PrefillCausalSDPAEncoding: AnyObject {
    var dispatchCounter: DispatchCounter? { get set }
    func encodeCausalSDPABatch(
        into encoder: MTLComputeCommandEncoder,
        cache: KVCache, layer: Int, basePosition: Int, batch: Int,
        query: MTLBuffer, numHeads: Int, output: MTLBuffer
    ) throws
}

extension PrefillSDPAKernel: PrefillCausalSDPAEncoding {}

/// PF-2: the QUERY-TILED causal SDPA for the tiled prefill — the
/// flash-attention structure on `simdgroup_matrix`. Engaged because the
/// device prefill attribution (DECISIONS.md 2026-09-19 P5-5B) put the PF-1
/// kernel at 29.2% of the on-device prefill span: that kernel runs one
/// threadgroup per (chunk position, query head), each streaming its head's
/// whole K/V prefix through scalar lanes with a cross-lane `simd_sum` per
/// (query, key) score — the K/V prefix is re-read once per position and
/// the arithmetic never touches the matrix unit.
///
/// Structure: one threadgroup (128 threads = 4 simdgroups) per 32-row
/// query tile. The 32 rows are `headsPerTile` query heads of one GQA group
/// × `32 / headsPerTile` consecutive chunk positions, so every K/V tile
/// loaded into threadgroup memory serves 32 query rows and (when the GQA
/// ratio allows) the whole kv-head group at once. Each simdgroup owns 8
/// rows of ONE head (consecutive positions) and walks the key range in
/// blocks of 32:
///
///   S (8×32)  = Q_sg (8×HD) · Kᵀ (HD×32)   — `simdgroup_multiply_accumulate`,
///                                            half inputs, fp32 accumulation
///   online softmax per row (fp32; running max m, denominator l) with the
///   causal mask applied to S in registers; P = exp(S − m) as half
///   O (8×HD) += P (8×32) · V (32×HD)       — half inputs, fp32 accumulators
///
/// Threadgroup memory holds the K block TRANSPOSED (`[dim][key]`, so both
/// operand loads are plain `simdgroup_load`s — P5-2 measured the
/// transposing load form slow) and the V block plain; the Q tile is
/// staged once and its buffer is then reused for V. Total 16 KiB at
/// headDim 128; no device-memory scratch (spec D2 budget unchanged), no
/// new persistent allocation (hard rule 4).
///
/// Register-level per-row work (mask, row max/sum, rescale) relies on the
/// lane → (row, column) ownership of `thread_elements()`: lane `t` holds
/// row `(t/4 & 4) + (t/2 % 4)`, columns `(t/4 & 2)·2 + (t%2)·2` and +1 —
/// an undocumented but stable Apple-silicon fact (the MLX steel kernels
/// rest on the same mapping) that `PrefillTiledSDPAKernelTests` PROBES on
/// the running device, so any drift fails loudly before the oracle gates
/// do. Row reductions combine the four lanes of a row with
/// `simd_shuffle_xor` 1 and 8 in fixed order ⇒ bitwise deterministic.
///
/// Precision (spec D2/D6): fp16 cache/query reads, fp32 scores and
/// softmax statistics, P rounded to fp16 for the matrix unit (the
/// denominator accumulates the SAME rounded weights, so numerator and
/// denominator stay consistent), fp32 output accumulation, fp16 store.
/// Gates at the attention-span constant max(2⁻⁷·M, 2⁻¹¹) — spec D6 "either
/// D4 form"; nothing new, nothing loosened. Depth 1 (absolute position 0)
/// stays the exact V-row copy (edge test 1 exactness carries from PF-1).
/// Masked keys contribute exactly zero (exp(−∞) = 0 → P = 0 → 0·V), so a
/// position's output is bitwise independent of every later slot's finite
/// contents; slots at or beyond the chunk end are never read at all
/// (zero-filled tiles).
///
/// Instances are specialized per headDim (a compile-time constant in the
/// kernel source, so every fragment array is register-resident):
/// headDim ∈ {8, 16, …, 128}.
public final class PrefillTiledSDPAKernel {
    /// Query rows per threadgroup (BQ) — 4 simdgroups × 8-row fragments.
    public static let rowsPerTile = 32
    /// Keys per threadgroup-memory block (BK).
    public static let keysPerBlock = 32
    static let threadsPerThreadgroup = 128
    /// The per-lane accumulator budget (16 fp32 8×8 fragments at 128).
    public static let maxHeadDim = 128

    /// The headDim this instance's pipeline is specialized for.
    public let headDim: Int

    /// Query heads sharing one tile: the largest of {4, 2, 1} dividing the
    /// GQA ratio, so a tile never crosses a kv-head group and each
    /// simdgroup's 8 rows are consecutive positions of a single head
    /// (32 / headsPerTile ≥ 8). The pinned model (16 q / 8 kv) tiles both
    /// heads of a group over 16 positions.
    public static func headsPerTile(groupSize: Int) -> Int {
        for candidate in [4, 2, 1] where groupSize % candidate == 0 {
            return candidate
        }
        return 1
    }

    /// Chunk positions covered by one tile for a given GQA ratio.
    public static func positionsPerTile(groupSize: Int) -> Int {
        rowsPerTile / headsPerTile(groupSize: groupSize)
    }

    private static func source(headDim: Int) -> String {
        """
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;

        constant uint HD = \(headDim);
        constant uint BQ = 32;           // query rows per threadgroup
        constant uint BK = 32;           // keys per block
        constant uint NSG = 4;           // simdgroups per threadgroup
        constant uint SG_ROWS = 8;       // query rows per simdgroup
        constant uint TG_THREADS = 128;
        constant uint KF = HD / 8;       // fragments along headDim
        constant uint CF = BK / 8;       // fragments along a key block

        // One threadgroup per (query tile, head group): flat 1-D grid of
        // ceil(batch / positionsPerTile) · (numHeads / headsPerTile)
        // threadgroups (MSL forbids mixing a vector grid attribute with
        // the scalar simdgroup attributes). Offsets are ELEMENT offsets
        // into the shared KV cache buffer (AttentionKernels convention).
        kernel void sdpa_prefill_tiled_f16(
            device const half *q                [[buffer(0)]],
            device const half *cache            [[buffer(1)]],
            constant ulong &keyBaseElemOffset   [[buffer(2)]],
            constant ulong &valueBaseElemOffset [[buffer(3)]],
            constant uint &groupSize            [[buffer(4)]],
            constant uint &maxContext           [[buffer(5)]],
            constant uint &basePosition         [[buffer(6)]],
            constant uint &numHeads             [[buffer(7)]],
            constant uint &batch                [[buffer(8)]],
            constant uint &headsPerTile         [[buffer(9)]],
            constant float &scale               [[buffer(10)]],
            device half *out                    [[buffer(11)]],
            uint tgIdx [[threadgroup_position_in_grid]],
            uint lane  [[thread_index_in_simdgroup]],
            uint sg    [[simdgroup_index_in_threadgroup]],
            uint tid   [[thread_index_in_threadgroup]])
        {
            const uint positionsPerTile = BQ / headsPerTile;
            const uint headGroups = numHeads / headsPerTile;
            const uint tile = tgIdx / headGroups;
            const uint headBase = (tgIdx % headGroups) * headsPerTile;
            const uint kvHead = headBase / groupSize;
            const uint tileStart = tile * positionsPerTile;   // chunk-relative
            const uint qStride = numHeads * HD;
            // Exclusive ABSOLUTE bound of the keys this tile may read: the
            // tile's last valid position + 1. Slots at/after it are never
            // loaded (zero-filled below).
            const uint keyEnd = basePosition + min(tileStart + positionsPerTile, batch);

            // This simdgroup's 8 rows: one head, consecutive positions.
            const uint sgRow0 = sg * SG_ROWS;
            const uint sgPosStart = tileStart + sgRow0 % positionsPerTile;
            const bool sgActive = sgPosStart < batch;
            const uint sgKeyEnd = basePosition + min(sgPosStart + SG_ROWS, batch);

            // Q tile (then V block) and the TRANSPOSED K block; half4-aligned.
            threadgroup half4 tileQV4[BQ * HD / 4];
            threadgroup half4 tileKT4[HD * BK / 4];
            threadgroup half *tileQV = reinterpret_cast<threadgroup half *>(tileQV4);
            threadgroup half *tileKT = reinterpret_cast<threadgroup half *>(tileKT4);

            // Stage the Q tile: rows beyond the chunk are zero (their
            // outputs are computed but never stored).
            for (uint idx = tid; idx < BQ * HD; idx += TG_THREADS) {
                const uint r = idx / HD;
                const uint d = idx % HD;
                const uint p = tileStart + r % positionsPerTile;
                const uint head = headBase + r / positionsPerTile;
                tileQV[idx] = (p < batch)
                    ? q[ulong(p) * qStride + ulong(head) * HD + d] : half(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_half8x8 qFrag[KF];
            for (uint k = 0; k < KF; ++k) {
                simdgroup_load(qFrag[k], tileQV + sgRow0 * HD + k * 8, HD);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);   // tileQV → V

            simdgroup_float8x8 oFrag[KF];
            for (uint k = 0; k < KF; ++k) {
                oFrag[k] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }

            // Fragment ownership (probed by the tests): this lane holds
            // row fm, columns fn and fn+1 of every 8×8 fragment.
            const uint qid = lane / 4;
            const uint fm = (qid & 4) + ((lane / 2) % 4);
            const uint fn = (qid & 2) * 2 + (lane % 2) * 2;
            const uint rowPos = basePosition + sgPosStart + fm;   // absolute
            float m = -INFINITY;   // running row max (scaled scores)
            float l = 0.0f;        // running row denominator

            const ulong kBase = keyBaseElemOffset + ulong(kvHead) * maxContext * HD;
            const ulong vBase = valueBaseElemOffset + ulong(kvHead) * maxContext * HD;

            for (uint blockStart = 0; blockStart < keyEnd; blockStart += BK) {
                // K block → tileKT[d][key]: lane ↔ key, simdgroups stride
                // the dims in groups of 4 (half4 device reads; the 32
                // lanes write 32 consecutive halves per dim — conflict-free).
                {
                    const uint key = blockStart + lane;
                    const bool valid = key < keyEnd;
                    device const half *kRow = cache + kBase + ulong(key) * HD;
                    for (uint d = sg * 4; d < HD; d += NSG * 4) {
                        const half4 v = valid
                            ? *reinterpret_cast<device const half4 *>(kRow + d)
                            : half4(0.0h);
                        tileKT[(d + 0) * BK + lane] = v.x;
                        tileKT[(d + 1) * BK + lane] = v.y;
                        tileKT[(d + 2) * BK + lane] = v.z;
                        tileKT[(d + 3) * BK + lane] = v.w;
                    }
                }
                // V block → tileQV[key][d], plain.
                for (uint idx = tid; idx < BK * HD / 4; idx += TG_THREADS) {
                    const uint elem = idx * 4;
                    const uint key = blockStart + elem / HD;
                    const uint d = elem % HD;
                    tileQV4[idx] = (key < keyEnd)
                        ? *reinterpret_cast<device const half4 *>(
                              cache + vBase + ulong(key) * HD + d)
                        : half4(0.0h);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Simdgroup-uniform: skip blocks entirely past this
                // simdgroup's last row (causal), and idle row groups.
                if (sgActive && blockStart < sgKeyEnd) {
                    // S = Q · Kᵀ
                    simdgroup_float8x8 s[CF];
                    for (uint c = 0; c < CF; ++c) {
                        s[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    }
                    for (uint k = 0; k < KF; ++k) {
                        for (uint c = 0; c < CF; ++c) {
                            simdgroup_half8x8 kt;
                            simdgroup_load(kt, tileKT + (k * 8) * BK + c * 8, BK);
                            simdgroup_multiply_accumulate(s[c], qFrag[k], kt, s[c]);
                        }
                    }
                    // Scale + causal mask in registers; block row max.
                    float sv[CF][2];
                    float blockMax = -INFINITY;
                    for (uint c = 0; c < CF; ++c) {
                        const thread float2 &e =
                            reinterpret_cast<const thread float2 &>(s[c].thread_elements());
                        for (uint j = 0; j < 2; ++j) {
                            const uint key = blockStart + c * 8 + fn + j;
                            const float v = (key > rowPos) ? -INFINITY : e[j] * scale;
                            sv[c][j] = v;
                            blockMax = max(blockMax, v);
                        }
                    }
                    blockMax = max(blockMax, simd_shuffle_xor(blockMax, 1));
                    blockMax = max(blockMax, simd_shuffle_xor(blockMax, 8));
                    // Online softmax update (fixed order ⇒ deterministic).
                    // Key 0 is unmasked for every valid row, so m is finite
                    // from block 0 on; an all-masked later block leaves m
                    // unchanged and adds exact zeros.
                    const float mNew = max(m, blockMax);
                    const float corr = (m == -INFINITY) ? 0.0f : exp(m - mNew);
                    float rowSum = 0.0f;
                    simdgroup_half8x8 p[CF];
                    for (uint c = 0; c < CF; ++c) {
                        const half w0 = half(exp(sv[c][0] - mNew));
                        const half w1 = half(exp(sv[c][1] - mNew));
                        rowSum += float(w0) + float(w1);
                        thread half2 &pe =
                            reinterpret_cast<thread half2 &>(p[c].thread_elements());
                        pe = half2(w0, w1);
                    }
                    rowSum += simd_shuffle_xor(rowSum, 1);
                    rowSum += simd_shuffle_xor(rowSum, 8);
                    l = l * corr + rowSum;
                    m = mNew;
                    for (uint k = 0; k < KF; ++k) {
                        thread float2 &oe =
                            reinterpret_cast<thread float2 &>(oFrag[k].thread_elements());
                        oe *= corr;
                    }
                    // O += P · V
                    for (uint c = 0; c < CF; ++c) {
                        for (uint k = 0; k < KF; ++k) {
                            simdgroup_half8x8 v;
                            simdgroup_load(v, tileQV + (c * 8) * HD + k * 8, HD);
                            simdgroup_multiply_accumulate(oFrag[k], p[c], v, oFrag[k]);
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);   // before reloading tiles
            }

            // Normalize and stage fp16 rows through the (now free) V tile,
            // then store only the rows inside the chunk.
            if (sgActive) {
                for (uint k = 0; k < KF; ++k) {
                    const thread float2 &oe =
                        reinterpret_cast<const thread float2 &>(oFrag[k].thread_elements());
                    simdgroup_half8x8 oh;
                    thread half2 &he =
                        reinterpret_cast<thread half2 &>(oh.thread_elements());
                    he = half2(half(oe[0] / l), half(oe[1] / l));
                    simdgroup_store(oh, tileQV + sgRow0 * HD + k * 8, HD);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < BQ * HD; idx += TG_THREADS) {
                const uint r = idx / HD;
                const uint d = idx % HD;
                const uint p = tileStart + r % positionsPerTile;
                if (p >= batch) { continue; }
                const uint head = headBase + r / positionsPerTile;
                // Depth 1 (absolute position 0): the softmax weight is
                // exactly 1.0, so attention IS the V row — a bitwise copy
                // preserves every bit pattern (-0.0, NaN payloads); the
                // matrix-unit form would not. One write per element.
                const half value = (basePosition + p == 0)
                    ? cache[vBase + d] : tileQV[idx];
                out[ulong(p) * qStride + ulong(head) * HD + d] = value;
            }
        }
        """
    }

    private let pipeline: MTLComputePipelineState

    /// When set, every encoded dispatch increments it at the dispatch call
    /// site (P2-5 instrumentation; `GPUModel` attaches its per-step counter).
    var dispatchCounter: DispatchCounter?

    /// Compiles the kernel specialized for `headDim` (a multiple of 8, at
    /// most `maxHeadDim`).
    public init(context: MetalContext, headDim: Int) throws {
        guard headDim > 0, headDim % 8 == 0 else {
            throw DecodeKernelError.unsupportedHeadDim(
                headDim: headDim,
                requirement: "the query-tiled SDPA needs a positive multiple of 8 "
                    + "(8×8 matrix fragments along headDim)")
        }
        guard headDim <= Self.maxHeadDim else {
            throw KVCacheError.headDimExceedsFusedLimit(
                headDim: headDim, limit: Self.maxHeadDim)
        }
        self.headDim = headDim
        let library = try context.makeLibrary(source: Self.source(headDim: headDim))
        pipeline = try context.makeComputePipeline(
            library: library, function: "sdpa_prefill_tiled_f16")
    }

    /// Encodes causal attention for `batch` chunk positions of `layer` in
    /// ONE dispatch: for p in 0..<batch, output row p = softmax(q_p ·
    /// K[kvHead, 0...basePosition+p] / √headDim) · V[kvHead, 0...basePosition+p]
    /// per query head. `query` and `output` are `[batch, numHeads·headDim]`
    /// fp16 row-major (the batched roped-q and attention scratch). The
    /// chunk's K/V must already be appended for positions
    /// basePosition..<basePosition+batch (the cluster kernel runs first).
    /// Same contract as `PrefillSDPAKernel.encodeCausalSDPABatch`.
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
        guard cache.headDim == headDim else {
            throw DecodeKernelError.unsupportedHeadDim(
                headDim: cache.headDim,
                requirement: "this query-tiled SDPA instance is specialized for "
                    + "headDim \(headDim)")
        }
        // Validates (layer, last position) via the cache's own bounds
        // checks; a chunk reaching maxContext throws BEFORE any dispatch.
        _ = try cache.elementOffset(
            layer: layer, component: .key, head: 0,
            position: basePosition + batch - 1)
        let keyBase = try cache.baseElementOffset(layer: layer, component: .key)
        let valueBase = try cache.baseElementOffset(layer: layer, component: .value)
        let rowBytes = numHeads * headDim * 2
        try requireCapacity(query, bytes: batch * rowBytes, name: "query")
        try requireCapacity(output, bytes: batch * rowBytes, name: "output")
        let threads = Self.threadsPerThreadgroup
        // The fragment-ownership map and the fixed 4-simdgroup tile assume
        // the Apple GPU shape (SIMD width exactly 32, ≥ 128 threads per
        // threadgroup); refuse loudly on anything else.
        guard pipeline.maxTotalThreadsPerThreadgroup >= threads,
              pipeline.threadExecutionWidth == 32 else {
            throw KVCacheError.fusedKernelUnsupportedDevice(
                threadExecutionWidth: pipeline.threadExecutionWidth,
                maxThreadsPerThreadgroup: pipeline.maxTotalThreadsPerThreadgroup)
        }
        let groupSize = numHeads / cache.kvHeads
        let headsPerTile = Self.headsPerTile(groupSize: groupSize)
        let positionsPerTile = Self.rowsPerTile / headsPerTile
        let tiles = (batch + positionsPerTile - 1) / positionsPerTile
        let headGroups = numHeads / headsPerTile
        // Same formula as the CPU reference's scale, for identical rounding.
        let scale = 1 / Float(headDim).squareRoot()

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(cache.buffer, offset: 0, index: 1)
        setScalar(encoder, UInt64(keyBase), index: 2)
        setScalar(encoder, UInt64(valueBase), index: 3)
        setScalar(encoder, UInt32(groupSize), index: 4)
        setScalar(encoder, UInt32(cache.maxContext), index: 5)
        setScalar(encoder, UInt32(basePosition), index: 6)
        setScalar(encoder, UInt32(numHeads), index: 7)
        setScalar(encoder, UInt32(batch), index: 8)
        setScalar(encoder, UInt32(headsPerTile), index: 9)
        setScalar(encoder, scale, index: 10)
        encoder.setBuffer(output, offset: 0, index: 11)
        dispatchCounter?.increment()
        encoder.dispatchThreadgroups(
            MTLSize(width: tiles * headGroups, height: 1, depth: 1),
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

extension PrefillTiledSDPAKernel: PrefillCausalSDPAEncoding {}
