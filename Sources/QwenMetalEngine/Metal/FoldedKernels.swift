import Metal

/// P4-3 (docs/phases/phase-4.md D3): the folding set for the fused (q4g64)
/// kernel path — four kernels that take the packed pipeline from 19 to 8
/// dispatches per layer:
///
/// - `matvec3_q4_f16`: the three QKV matvecs as ONE dispatch over the
///   concatenated output rows (q | k | v), writing the `[numHeads + 2·kvHeads,
///   headDim]` qkv buffer the cluster kernel consumes. Same inner loop and
///   rounding as `matvec_q4_f16` — a matvec-only span (Tier K).
/// - `qknorm_rope_append_f16`: the 6-dispatch post-QKV elementwise cluster
///   (q-norm, k-norm, rope-q, rope-k, append-K, append-V) as ONE dispatch.
///   Q heads: per-head RMSNorm + RoPE → the query buffer. K heads: per-head
///   RMSNorm + RoPE stored straight into the cache slot — the k-side cache
///   write becomes a COMPUTED value, so its exactness gate is replaced by the
///   norm-species tolerance (spec D3/D5, approved gates entry 2026-09-05).
///   V heads: pure fp16 copy into the cache slot — no arithmetic touches the
///   bits, so the v-side append stays EXACT. Reads are out-of-place (the qkv
///   buffer is never written), so the per-thread row re-reads race nothing.
/// - `gateup_swiglu_q4_f16`: gate and up projections as ONE dispatch; each
///   thread computes both row dots and applies SwiGLU to its own pair — the
///   standalone SwiGLU dispatch disappears with zero redundant compute.
/// - `matvec_res_q4_f16`: a matvec with the residual add folded into the
///   store (serves o_proj and down_proj) — the standalone residual dispatches
///   disappear.
///
/// Every fold preserves the fp16-boundary/fp32-accumulate semantics of the
/// unfused chain (spec D3): matvec accumulators round to fp16 exactly where
/// the naive kernels stored fp16, norm outputs round to fp16 before RoPE
/// reads them, and the residual add reads two fp16 values and rounds once —
/// so each fused span computes the naive chain's arithmetic, just without
/// the intermediate DRAM round-trips and dispatch boundaries. Dequant stays
/// in registers (hard rule 1); weights never exist dequantized in DRAM.
///
/// These kernels exist ONLY on the fused kernel path of the packed pipeline
/// (spec D4): the bf16 backend keeps the naive Phase 2 structure permanently,
/// and the naive packed path stays selectable for the P4-5 before/after row.
public final class FoldedKernels {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // bf16 -> fp32 exact upcast (DecodeKernels): identical to the CPU
    // reference's, so norm-weight bits contribute zero divergence.
    inline float bf16_to_f32(ushort w) {
        return as_type<float>(uint(w) << 16);
    }

    // One packed row's dot product with x — the matvec_q4_f16 inner loop
    // verbatim (same accumulation order, same single-rounded dequant), so a
    // folded matvec computes the standalone kernel's arithmetic exactly.
    inline float matvec_row_q4(device const uint *q, ulong qElemOffset,
                               device const half *scales, ulong scalesElemOffset,
                               device const half *biases, ulong biasesElemOffset,
                               device const half *x, uint row, uint inDim) {
        uint groupsPerRow = inDim / 64;
        ulong qBase = qElemOffset + ulong(row) * (inDim / 8);
        ulong groupBase = ulong(row) * groupsPerRow;
        float acc = 0.0f;
        uint c = 0;
        for (uint g = 0; g < groupsPerRow; ++g) {
            float s = float(scales[scalesElemOffset + groupBase + g]);
            float b = float(biases[biasesElemOffset + groupBase + g]);
            for (uint w = 0; w < 8; ++w) {
                uint word = q[qBase + ulong(g) * 8 + w];
                for (uint lane = 0; lane < 8; ++lane) {
                    acc += (float(word & 0xFu) * s + b) * float(x[c]);
                    word >>= 4;
                    ++c;
                }
            }
        }
        return acc;
    }

    // Three packed matvecs, one dispatch: out rows [0, outDimA) come from
    // triplet A, [outDimA, outDimA+outDimB) from B, the rest from C. All
    // three share the same input x (the normed hidden state).
    kernel void matvec3_q4_f16(device const uint *qA        [[buffer(0)]],
                               constant ulong &qAOffset     [[buffer(1)]],
                               device const half *sA        [[buffer(2)]],
                               constant ulong &sAOffset     [[buffer(3)]],
                               device const half *bA        [[buffer(4)]],
                               constant ulong &bAOffset     [[buffer(5)]],
                               device const uint *qB        [[buffer(6)]],
                               constant ulong &qBOffset     [[buffer(7)]],
                               device const half *sB        [[buffer(8)]],
                               constant ulong &sBOffset     [[buffer(9)]],
                               device const half *bB        [[buffer(10)]],
                               constant ulong &bBOffset     [[buffer(11)]],
                               device const uint *qC        [[buffer(12)]],
                               constant ulong &qCOffset     [[buffer(13)]],
                               device const half *sC        [[buffer(14)]],
                               constant ulong &sCOffset     [[buffer(15)]],
                               device const half *bC        [[buffer(16)]],
                               constant ulong &bCOffset     [[buffer(17)]],
                               device const half *x         [[buffer(18)]],
                               constant uint &outDimA       [[buffer(19)]],
                               constant uint &outDimB       [[buffer(20)]],
                               constant uint &outDimC       [[buffer(21)]],
                               constant uint &inDim         [[buffer(22)]],
                               device half *out             [[buffer(23)]],
                               uint gid [[thread_position_in_grid]]) {
        if (gid >= outDimA + outDimB + outDimC) return;
        float acc;
        if (gid < outDimA) {
            acc = matvec_row_q4(qA, qAOffset, sA, sAOffset, bA, bAOffset,
                                x, gid, inDim);
        } else if (gid < outDimA + outDimB) {
            acc = matvec_row_q4(qB, qBOffset, sB, sBOffset, bB, bBOffset,
                                x, gid - outDimA, inDim);
        } else {
            acc = matvec_row_q4(qC, qCOffset, sC, sCOffset, bC, bCOffset,
                                x, gid - outDimA - outDimB, inDim);
        }
        out[gid] = half(acc);
    }

    // The post-QKV cluster, one dispatch. Thread (pair, role-head): role
    // heads 0..<numHeads are q (norm + rope → qOut), the next kvHeads are k
    // (norm + rope → cache K slot), the last kvHeads are v (pure copy →
    // cache V slot). The qkv layout is the matvec3 output read as
    // [numHeads + 2·kvHeads, headDim]. Norm rows recompute their sum of
    // squares sequentially per thread (the rmsnorm_f16 naive-redundancy
    // pattern), and every rounding boundary matches the unfused chain: norm
    // output rounds to fp16 BEFORE the rotation reads it.
    kernel void qknorm_rope_append_f16(device const half *qkv       [[buffer(0)]],
                                       device const ushort *qNorm   [[buffer(1)]],
                                       constant ulong &qNormElemOffset [[buffer(2)]],
                                       device const ushort *kNorm   [[buffer(3)]],
                                       constant ulong &kNormElemOffset [[buffer(4)]],
                                       device const float *cosTable [[buffer(5)]],
                                       device const float *sinTable [[buffer(6)]],
                                       device half *cache           [[buffer(7)]],
                                       constant ulong &keyBaseElemOffset [[buffer(8)]],
                                       constant ulong &valueBaseElemOffset [[buffer(9)]],
                                       constant uint &position      [[buffer(10)]],
                                       constant uint &maxContext    [[buffer(11)]],
                                       constant uint &headDim       [[buffer(12)]],
                                       constant uint &numHeads      [[buffer(13)]],
                                       constant uint &kvHeads       [[buffer(14)]],
                                       constant float &eps          [[buffer(15)]],
                                       device half *qOut            [[buffer(16)]],
                                       uint2 gid [[thread_position_in_grid]]) {
        uint halfDim = headDim / 2;
        if (gid.x >= halfDim || gid.y >= numHeads + 2 * kvHeads) return;
        bool isQ = gid.y < numHeads;
        bool isK = !isQ && gid.y < numHeads + kvHeads;
        ulong inBase = ulong(gid.y) * headDim;

        if (!isQ && !isK) {
            // V: pure fp16 copy into the cache slot — EXACT (bit patterns
            // preserved, NaN payloads included; the kv_append_f16 claim).
            uint h = gid.y - numHeads - kvHeads;
            ulong slot = valueBaseElemOffset
                + (ulong(h) * maxContext + position) * headDim;
            cache[slot + gid.x] = qkv[inBase + gid.x];
            cache[slot + halfDim + gid.x] = qkv[inBase + halfDim + gid.x];
            return;
        }

        float sumOfSquares = 0.0f;
        for (uint j = 0; j < headDim; ++j) {
            float v = float(qkv[inBase + j]);
            sumOfSquares += v * v;
        }
        float inverseRMS = 1.0f / sqrt(sumOfSquares / float(headDim) + eps);
        device const ushort *w = isQ ? qNorm : kNorm;
        ulong wOff = isQ ? qNormElemOffset : kNormElemOffset;
        // fp16 round at the norm/rope boundary, exactly like the unfused
        // rmsnorm_f16 store the rope_f16 kernel then read.
        float n1 = float(half(bf16_to_f32(w[wOff + gid.x])
            * (float(qkv[inBase + gid.x]) * inverseRMS)));
        float n2 = float(half(bf16_to_f32(w[wOff + halfDim + gid.x])
            * (float(qkv[inBase + halfDim + gid.x]) * inverseRMS)));
        float c = cosTable[ulong(position) * halfDim + gid.x];
        float sn = sinTable[ulong(position) * halfDim + gid.x];
        half r1 = half(n1 * c - n2 * sn);
        half r2 = half(n2 * c + n1 * sn);
        if (isQ) {
            qOut[inBase + gid.x] = r1;
            qOut[inBase + halfDim + gid.x] = r2;
        } else {
            uint h = gid.y - numHeads;
            ulong slot = keyBaseElemOffset
                + (ulong(h) * maxContext + position) * headDim;
            cache[slot + gid.x] = r1;
            cache[slot + halfDim + gid.x] = r2;
        }
    }

    // Gate and up projections + SwiGLU, one dispatch: thread r computes BOTH
    // row-r dots, rounds each to fp16 (the unfused matvec stores), and
    // applies the SwiGLU formula (swiglu_f16 verbatim) to its own pair.
    kernel void gateup_swiglu_q4_f16(device const uint *qG        [[buffer(0)]],
                                     constant ulong &qGOffset     [[buffer(1)]],
                                     device const half *sG        [[buffer(2)]],
                                     constant ulong &sGOffset     [[buffer(3)]],
                                     device const half *bG        [[buffer(4)]],
                                     constant ulong &bGOffset     [[buffer(5)]],
                                     device const uint *qU        [[buffer(6)]],
                                     constant ulong &qUOffset     [[buffer(7)]],
                                     device const half *sU        [[buffer(8)]],
                                     constant ulong &sUOffset     [[buffer(9)]],
                                     device const half *bU        [[buffer(10)]],
                                     constant ulong &bUOffset     [[buffer(11)]],
                                     device const half *x         [[buffer(12)]],
                                     constant uint &outDim        [[buffer(13)]],
                                     constant uint &inDim         [[buffer(14)]],
                                     device half *out             [[buffer(15)]],
                                     uint gid [[thread_position_in_grid]]) {
        if (gid >= outDim) return;
        float g = float(half(matvec_row_q4(qG, qGOffset, sG, sGOffset,
                                           bG, bGOffset, x, gid, inDim)));
        float u = float(half(matvec_row_q4(qU, qUOffset, sU, sUOffset,
                                           bU, bUOffset, x, gid, inDim)));
        out[gid] = half((g / (1.0f + exp(-g))) * u);
    }

    // Matvec with the residual add folded into the store: the accumulator
    // rounds to fp16 exactly where matvec_q4_f16 stored it, then the
    // residual_add_f16 arithmetic (fp32 add, one round) runs in place.
    kernel void matvec_res_q4_f16(device const uint *q         [[buffer(0)]],
                                  constant ulong &qElemOffset  [[buffer(1)]],
                                  device const half *scales    [[buffer(2)]],
                                  constant ulong &scalesElemOffset [[buffer(3)]],
                                  device const half *biases    [[buffer(4)]],
                                  constant ulong &biasesElemOffset [[buffer(5)]],
                                  device const half *x         [[buffer(6)]],
                                  device const half *res       [[buffer(7)]],
                                  constant uint &outDim        [[buffer(8)]],
                                  constant uint &inDim         [[buffer(9)]],
                                  device half *out             [[buffer(10)]],
                                  uint gid [[thread_position_in_grid]]) {
        if (gid >= outDim) return;
        half projected = half(matvec_row_q4(
            q, qElemOffset, scales, scalesElemOffset, biases, biasesElemOffset,
            x, gid, inDim));
        out[gid] = half(float(res[gid]) + float(projected));
    }
    """

    private let matvec3Pipeline: MTLComputePipelineState
    private let clusterPipeline: MTLComputePipelineState
    private let gateupPipeline: MTLComputePipelineState
    private let matvecResPipeline: MTLComputePipelineState

    /// When set, every encoded dispatch increments it at the dispatch call
    /// site (P2-5 instrumentation; `GPUModel` attaches its per-step counter).
    var dispatchCounter: DispatchCounter?

    public init(context: MetalContext) throws {
        let library = try context.makeLibrary(source: Self.source)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            try context.makeComputePipeline(library: library, function: name)
        }
        matvec3Pipeline = try pipeline("matvec3_q4_f16")
        clusterPipeline = try pipeline("qknorm_rope_append_f16")
        gateupPipeline = try pipeline("gateup_swiglu_q4_f16")
        matvecResPipeline = try pipeline("matvec_res_q4_f16")
    }

    /// One packed triplet's buffers + byte offsets (GPUWeights convention:
    /// offsets convert to elements host-side, never `setBuffer` offsets).
    public struct Triplet {
        public let q: MTLBuffer
        public let qByteOffset: Int
        public let scales: MTLBuffer
        public let scalesByteOffset: Int
        public let biases: MTLBuffer
        public let biasesByteOffset: Int

        public init(
            q: MTLBuffer, qByteOffset: Int,
            scales: MTLBuffer, scalesByteOffset: Int,
            biases: MTLBuffer, biasesByteOffset: Int
        ) {
            self.q = q
            self.qByteOffset = qByteOffset
            self.scales = scales
            self.scalesByteOffset = scalesByteOffset
            self.biases = biases
            self.biasesByteOffset = biasesByteOffset
        }
    }

    // MARK: - Encode methods (one per kernel)

    /// y = concat(A·x, B·x, C·x) in one dispatch, fp16 store — the QKV
    /// concatenation (spec D3). `output` is `[outDimA + outDimB + outDimC]`
    /// fp16, q rows first, then k, then v.
    public func encodeMatvec3(
        into encoder: MTLComputeCommandEncoder,
        a: Triplet, outDimA: Int, b: Triplet, outDimB: Int,
        c: Triplet, outDimC: Int, inDim: Int,
        input: MTLBuffer, output: MTLBuffer
    ) throws {
        let offsetsA = try QuantKernels.tripletElementOffsets(
            a, outDim: outDimA, inDim: inDim)
        let offsetsB = try QuantKernels.tripletElementOffsets(
            b, outDim: outDimB, inDim: inDim)
        let offsetsC = try QuantKernels.tripletElementOffsets(
            c, outDim: outDimC, inDim: inDim)
        try QuantKernels.requireCapacity(input, bytes: inDim * 2, name: "input")
        try QuantKernels.requireCapacity(
            output, bytes: (outDimA + outDimB + outDimC) * 2, name: "output")

        encoder.setComputePipelineState(matvec3Pipeline)
        setTriplet(encoder, a, offsets: offsetsA, baseIndex: 0)
        setTriplet(encoder, b, offsets: offsetsB, baseIndex: 6)
        setTriplet(encoder, c, offsets: offsetsC, baseIndex: 12)
        encoder.setBuffer(input, offset: 0, index: 18)
        setScalar(encoder, UInt32(outDimA), index: 19)
        setScalar(encoder, UInt32(outDimB), index: 20)
        setScalar(encoder, UInt32(outDimC), index: 21)
        setScalar(encoder, UInt32(inDim), index: 22)
        encoder.setBuffer(output, offset: 0, index: 23)
        dispatch1D(encoder, pipeline: matvec3Pipeline,
                   count: outDimA + outDimB + outDimC)
    }

    /// The fused post-QKV cluster (spec D3): per-head Q/K RMSNorm + RoPE at
    /// absolute `position`, the roped q heads stored to `qOut`, the roped k
    /// heads stored INTO the cache slot, and the v heads copied into theirs —
    /// one dispatch replacing q-norm/k-norm/rope-q/rope-k/append-K/append-V.
    ///
    /// `qkv` is the matvec3 output read as `[numHeads + 2·kvHeads, headDim]`
    /// fp16 (q heads, then k, then v); it is never written. Norm weights are
    /// bf16 vectors of `headDim` values (schema D1 pass-through). Appending
    /// at `position == maxContext` throws `.contextFull` BEFORE any dispatch
    /// with the cache untouched (the encodeKVAppend contract carries over).
    public func encodeQKNormRoPEAppend(
        into encoder: MTLComputeCommandEncoder,
        qkv: MTLBuffer,
        qNormWeight: MTLBuffer, qNormByteOffset: Int,
        kNormWeight: MTLBuffer, kNormByteOffset: Int,
        cosTable: MTLBuffer, sinTable: MTLBuffer,
        position: Int, positions: Int, numHeads: Int, eps: Float,
        cache: KVCache, layer: Int, qOut: MTLBuffer
    ) throws {
        try requirePositive(numHeads, "numHeads")
        let headDim = cache.headDim
        guard headDim % 2 == 0 else {
            throw DecodeKernelError.oddHeadDim(headDim: headDim)
        }
        try requirePositive(positions, "positions")
        // Context-full FIRST (the encodeKVAppend contract, spec edge test 3):
        // with the usual positions == maxContext table this check must win
        // over the rope-table range check so callers see `.contextFull`.
        guard position < cache.maxContext else {
            throw KVCacheError.contextFull(
                position: position, maxContext: cache.maxContext)
        }
        guard position >= 0, position < positions else {
            throw DecodeKernelError.positionOutOfRange(
                position: position, positions: positions)
        }
        // Validates `layer` via the cache's own bounds checks.
        _ = try cache.elementOffset(
            layer: layer, component: .key, head: 0, position: position)
        let keyBase = try cache.baseElementOffset(layer: layer, component: .key)
        let valueBase = try cache.baseElementOffset(layer: layer, component: .value)

        let kvHeads = cache.kvHeads
        let halfDim = headDim / 2
        let totalHeads = numHeads + 2 * kvHeads
        let qNormElemOffset = try normWeightElementOffset(
            byteOffset: qNormByteOffset, dim: headDim,
            buffer: qNormWeight, name: "qNormWeight")
        let kNormElemOffset = try normWeightElementOffset(
            byteOffset: kNormByteOffset, dim: headDim,
            buffer: kNormWeight, name: "kNormWeight")
        try QuantKernels.requireCapacity(
            qkv, bytes: totalHeads * headDim * 2, name: "qkv")
        try QuantKernels.requireCapacity(
            qOut, bytes: numHeads * headDim * 2, name: "qOut")
        try QuantKernels.requireCapacity(
            cosTable, bytes: positions * halfDim * 4, name: "cosTable")
        try QuantKernels.requireCapacity(
            sinTable, bytes: positions * halfDim * 4, name: "sinTable")

        encoder.setComputePipelineState(clusterPipeline)
        encoder.setBuffer(qkv, offset: 0, index: 0)
        encoder.setBuffer(qNormWeight, offset: 0, index: 1)
        setScalar(encoder, UInt64(qNormElemOffset), index: 2)
        encoder.setBuffer(kNormWeight, offset: 0, index: 3)
        setScalar(encoder, UInt64(kNormElemOffset), index: 4)
        encoder.setBuffer(cosTable, offset: 0, index: 5)
        encoder.setBuffer(sinTable, offset: 0, index: 6)
        encoder.setBuffer(cache.buffer, offset: 0, index: 7)
        setScalar(encoder, UInt64(keyBase), index: 8)
        setScalar(encoder, UInt64(valueBase), index: 9)
        setScalar(encoder, UInt32(position), index: 10)
        setScalar(encoder, UInt32(cache.maxContext), index: 11)
        setScalar(encoder, UInt32(headDim), index: 12)
        setScalar(encoder, UInt32(numHeads), index: 13)
        setScalar(encoder, UInt32(kvHeads), index: 14)
        setScalar(encoder, eps, index: 15)
        encoder.setBuffer(qOut, offset: 0, index: 16)
        dispatch2D(encoder, pipeline: clusterPipeline,
                   width: halfDim, height: totalHeads)
    }

    /// out = silu(G·x) · (U·x) in one dispatch, fp16 store — gate+up
    /// concatenation with the SwiGLU fold (spec D3). Both triplets share
    /// `[outDim, inDim]`.
    public func encodeGateUpSwiGLU(
        into encoder: MTLComputeCommandEncoder,
        gate: Triplet, up: Triplet, outDim: Int, inDim: Int,
        input: MTLBuffer, output: MTLBuffer
    ) throws {
        let offsetsG = try QuantKernels.tripletElementOffsets(
            gate, outDim: outDim, inDim: inDim)
        let offsetsU = try QuantKernels.tripletElementOffsets(
            up, outDim: outDim, inDim: inDim)
        try QuantKernels.requireCapacity(input, bytes: inDim * 2, name: "input")
        try QuantKernels.requireCapacity(output, bytes: outDim * 2, name: "output")

        encoder.setComputePipelineState(gateupPipeline)
        setTriplet(encoder, gate, offsets: offsetsG, baseIndex: 0)
        setTriplet(encoder, up, offsets: offsetsU, baseIndex: 6)
        encoder.setBuffer(input, offset: 0, index: 12)
        setScalar(encoder, UInt32(outDim), index: 13)
        setScalar(encoder, UInt32(inDim), index: 14)
        encoder.setBuffer(output, offset: 0, index: 15)
        dispatch1D(encoder, pipeline: gateupPipeline, count: outDim)
    }

    /// out = res + W·x in one dispatch, fp16 store — the residual fold
    /// (spec D3) serving o_proj and down_proj. `res` is `[outDim]` fp16.
    public func encodeMatvecResidual(
        into encoder: MTLComputeCommandEncoder,
        triplet: Triplet, outDim: Int, inDim: Int,
        input: MTLBuffer, residual: MTLBuffer, output: MTLBuffer
    ) throws {
        let offsets = try QuantKernels.tripletElementOffsets(
            triplet, outDim: outDim, inDim: inDim)
        try QuantKernels.requireCapacity(input, bytes: inDim * 2, name: "input")
        try QuantKernels.requireCapacity(
            residual, bytes: outDim * 2, name: "residual")
        try QuantKernels.requireCapacity(output, bytes: outDim * 2, name: "output")

        encoder.setComputePipelineState(matvecResPipeline)
        setTriplet(encoder, triplet, offsets: offsets, baseIndex: 0)
        encoder.setBuffer(input, offset: 0, index: 6)
        encoder.setBuffer(residual, offset: 0, index: 7)
        setScalar(encoder, UInt32(outDim), index: 8)
        setScalar(encoder, UInt32(inDim), index: 9)
        encoder.setBuffer(output, offset: 0, index: 10)
        dispatch1D(encoder, pipeline: matvecResPipeline, count: outDim)
    }

    // MARK: - Validation + dispatch helpers (QuantKernels/DecodeKernels conventions)

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

    /// Binds one triplet at indices baseIndex..baseIndex+5 (the QuantKernels
    /// buffer/offset-pair layout).
    private func setTriplet(
        _ encoder: MTLComputeCommandEncoder, _ triplet: Triplet,
        offsets: QuantKernels.TripletElementOffsets, baseIndex: Int
    ) {
        encoder.setBuffer(triplet.q, offset: 0, index: baseIndex)
        setScalar(encoder, UInt64(offsets.q), index: baseIndex + 1)
        encoder.setBuffer(triplet.scales, offset: 0, index: baseIndex + 2)
        setScalar(encoder, UInt64(offsets.scales), index: baseIndex + 3)
        encoder.setBuffer(triplet.biases, offset: 0, index: baseIndex + 4)
        setScalar(encoder, UInt64(offsets.biases), index: baseIndex + 5)
    }

    // Concrete overloads (not generic): sidesteps the DK-1 "T may contain an
    // object reference" warning species by construction.
    private func setScalar(
        _ encoder: MTLComputeCommandEncoder, _ value: UInt32, index: Int
    ) {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<UInt32>.stride, index: index)
    }

    private func setScalar(
        _ encoder: MTLComputeCommandEncoder, _ value: UInt64, index: Int
    ) {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<UInt64>.stride, index: index)
    }

    private func setScalar(
        _ encoder: MTLComputeCommandEncoder, _ value: Float, index: Int
    ) {
        var v = value
        encoder.setBytes(&v, length: MemoryLayout<Float>.stride, index: index)
    }

    private func dispatch1D(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState, count: Int
    ) {
        dispatchCounter?.increment()
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func dispatch2D(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState, width: Int, height: Int
    ) {
        dispatchCounter?.increment()
        // 16×16 threadgroups (DecodeKernels pattern); dispatchThreads handles
        // the ragged edge (kernel bounds checks are belt-and-braces).
        let side = 16
        let tgWidth = min(side, pipeline.maxTotalThreadsPerThreadgroup)
        let tgHeight = max(1, min(side, pipeline.maxTotalThreadsPerThreadgroup / tgWidth))
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: tgHeight, depth: 1))
    }
}
