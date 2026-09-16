import Metal

/// P5-2 (docs/phases/phase-5.md D3): the tiled q4g64 dequant-GEMM kernel —
/// the Phase 5 core. C×K fp16 activations × a packed [N, K] q4g64 triplet →
/// C×N fp16 output, fp32 accumulation, built on threadgroup memory tiles +
/// `simdgroup_matrix` accumulation (the PLAN phase-table pin).
///
/// Dequantization happens inside this consuming kernel (hard rule 1).
/// Dequantized weight TILES are staged in threadgroup (on-chip) memory —
/// permitted by the recorded hard-rule-1 clarification (DECISIONS.md
/// 2026-09-14 Phase 5 gates entry): the tile is transient scratch that dies
/// with the threadgroup. Nothing dequantized ever reaches a device/DRAM
/// buffer.
///
/// Arithmetic: the staged weight value is the pinned `Q4G64.dequant`
/// fp32 result (`float(q)·float(scale)+float(bias)`, one correctly rounded
/// operation), activations upcast fp16→fp32 exactly, and
/// `simdgroup_float8x8` multiply-accumulate keeps every partial sum in
/// fp32 — the same value set the CPU-quant oracle (sgemm over
/// `dequantMatrix`) consumes. Only the reduction ORDER differs from
/// matvec/sgemm, which the pre-committed Tier K gate absorbs (the
/// 2026-09-12 reduction-order decision; tiled-vs-sequential outputs are
/// not required to match bitwise).
///
/// Two kernels serve one API, selected by batchM (spec D3: "the task
/// chooses by measurement"; the optimization-iteration ledger is in the
/// DECISIONS.md P5-2 entry — every iteration re-passed the correctness
/// suite per hard rule 3):
///
///   `gemm_q4_f16` (batchM > 8, the prefill-chunk path, the PLAN pin's
///   structure): BM = BN = BK = 32, 128 threads = 4 simdgroups per
///   threadgroup, grid ⌈N/32⌉ × ⌈M/32⌉. Per K-step, the threadgroup
///   cooperatively stages A (32×32 fp32, zero-padded past M) and the
///   dequantized W tile — TRANSPOSED to [k][n] at dequant time so the
///   inner-loop fragment loads are plain (transposed simdgroup_load
///   measured slow) — then each simdgroup accumulates its 8-row slab via
///   8×8 fragments. K is structurally a multiple of 64 (q4g64 schema), so
///   32-wide K-tiles are always full and never straddle a group mid-word.
///
///   `gemm_q4_f16_m8` (batchM ≤ 8, the microbench gate point): measured
///   faster than every tiled variant at M=8 — see its comment.
///
/// Ragged M and N (M % tile ≠ 0, M = 1, odd N) are handled by zero-padded
/// loads and bounds-guarded stores — spec edge tests 1–2 pin all of it
/// against the CPU-quant oracle.
public final class QuantGemmKernel {
    /// Output tile rows (M), columns (N), and K-step per threadgroup.
    static let tileM = 32
    static let tileN = 32
    static let tileK = 32
    static let threadsPerThreadgroup = 128

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    constant constexpr uint BM = \(tileM);
    constant constexpr uint BN = \(tileN);
    constant constexpr uint BK = \(tileK);

    // out[M, N] = x[M, K] · dequant(W[N, K])ᵀ, fp32 accumulation via
    // simdgroup_float8x8, fp16 store. Triplet layout and element-offset
    // convention are identical to the P3-4 matvec kernels.
    kernel void gemm_q4_f16(device const uint *q        [[buffer(0)]],
                            constant ulong &qElemOffset [[buffer(1)]],
                            device const half *scales   [[buffer(2)]],
                            constant ulong &scalesElemOffset [[buffer(3)]],
                            device const half *biases   [[buffer(4)]],
                            constant ulong &biasesElemOffset [[buffer(5)]],
                            device const half *x        [[buffer(6)]],
                            constant uint &batchM       [[buffer(7)]],
                            constant uint &outDim       [[buffer(8)]],
                            constant uint &inDim        [[buffer(9)]],
                            device half *out            [[buffer(10)]],
                            uint2 tgid [[threadgroup_position_in_grid]],
                            uint tid  [[thread_index_in_threadgroup]],
                            uint sgid [[simdgroup_index_in_threadgroup]]) {
        threadgroup float aTile[BM * BK];   // [m][k], zero-padded past M
        // W tile staged TRANSPOSED ([k][n]) at dequant time: the inner-loop
        // fragment loads are then plain (transposed simdgroup_load measured
        // slow on the 2026-09-15 Mac sanity run), and the staging writes are
        // coalesced across lanes (consecutive n at fixed k).
        threadgroup float wTileT[BK * BN];  // [k][n], dequantized, zero-padded past N
        threadgroup float outTile[BM * BN]; // [m][n] staging for guarded stores

        const uint m0 = tgid.y * BM;
        const uint n0 = tgid.x * BN;
        const uint wordsPerRow = inDim / 8;
        const uint groupsPerRow = inDim / 64;

        // Cooperative-load ownership: thread `tid` owns row r = tid/4 of the
        // tile and 8-element segment c = tid%4 — exactly one packed u32 word
        // on the W side.
        const uint r = tid / 4;
        const uint c = tid % 4;

        simdgroup_float8x8 acc[BN / 8];
        for (uint nn = 0; nn < BN / 8; ++nn) {
            acc[nn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }

        for (uint k0 = 0; k0 < inDim; k0 += BK) {
            // A tile: fp16 → fp32 exact upcast; rows past M stage zeros so
            // ragged-M tiles accumulate garbage-free.
            {
                const uint m = m0 + r;
                for (uint j = 0; j < 8; ++j) {
                    const uint kk = 8 * c + j;
                    aTile[r * BK + kk] = (m < batchM)
                        ? float(x[ulong(m) * inDim + k0 + kk]) : 0.0f;
                }
            }
            // W tile: one u32 word (8 codes) per thread, register-dequanted
            // into the transposed threadgroup stage. The word's 8 codes
            // share one group (words never straddle a group of 64).
            {
                const uint n = n0 + r;
                const uint kBase = 8 * c;
                if (n < outDim) {
                    uint word = q[qElemOffset + ulong(n) * wordsPerRow
                                  + (k0 + kBase) / 8];
                    const ulong group = ulong(n) * groupsPerRow + (k0 + kBase) / 64;
                    const float s = float(scales[scalesElemOffset + group]);
                    const float b = float(biases[biasesElemOffset + group]);
                    for (uint j = 0; j < 8; ++j) {
                        // The pinned Q4G64.dequant arithmetic, verbatim.
                        wTileT[(kBase + j) * BN + r] = float(word & 0xFu) * s + b;
                        word >>= 4;
                    }
                } else {
                    for (uint j = 0; j < 8; ++j) wTileT[(kBase + j) * BN + r] = 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Simdgroup `sgid` accumulates output rows [sgid*8, sgid*8+8).
            const threadgroup float *aBase = aTile + sgid * 8 * BK;
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_float8x8 aFrag;
                simdgroup_load(aFrag, aBase + kk, BK);
                for (uint nn = 0; nn < BN / 8; ++nn) {
                    simdgroup_float8x8 wFrag;
                    simdgroup_load(wFrag, wTileT + kk * BN + nn * 8, BN);
                    simdgroup_multiply_accumulate(acc[nn], aFrag, wFrag, acc[nn]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint nn = 0; nn < BN / 8; ++nn) {
            simdgroup_store(acc[nn], outTile + sgid * 8 * BN + nn * 8, BN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Guarded fp16 store: each accumulated value rounds fp32 → fp16 once.
        {
            const uint m = m0 + r;
            if (m < batchM) {
                for (uint j = 0; j < 8; ++j) {
                    const uint n = n0 + 8 * c + j;
                    if (n < outDim) {
                        out[ulong(m) * outDim + n] =
                            half(outTile[r * BN + 8 * c + j]);
                    }
                }
            }
        }
    }

    // Small-batch specialization (M ≤ 8), chosen by measurement (the D3
    // "task chooses by measurement" license; Mac sanity runs 2026-09-15):
    // at M=8 the threadgroup round-trip (dequant → staged write → fragment
    // read → MAC) costs more lane-work per weight than the 8 useful FLOPs
    // it enables — tiled variants measured 11.6–13.8 GB/s effective vs the
    // matvec bench's 58.8 on the same Mac. So the small-batch path is a
    // register-blocked multi-matvec: one thread streams one W row exactly
    // like the P3-4 matvec (the memory pattern measured at 35.29 GB/s
    // on-device) holding up to 8 fp32 accumulators. Dequant stays in
    // registers; accumulation stays fp32; the tiled threadgroup +
    // simdgroup_matrix kernel above remains the prefill-chunk (large-M)
    // path — the PLAN pin's structure.
    kernel void gemm_q4_f16_m8(device const uint *q        [[buffer(0)]],
                               constant ulong &qElemOffset [[buffer(1)]],
                               device const half *scales   [[buffer(2)]],
                               constant ulong &scalesElemOffset [[buffer(3)]],
                               device const half *biases   [[buffer(4)]],
                               constant ulong &biasesElemOffset [[buffer(5)]],
                               device const half *x        [[buffer(6)]],
                               constant uint &batchM       [[buffer(7)]],
                               constant uint &outDim       [[buffer(8)]],
                               constant uint &inDim        [[buffer(9)]],
                               device half *out            [[buffer(10)]],
                               uint2 tgid [[threadgroup_position_in_grid]],
                               uint tid  [[thread_index_in_threadgroup]]) {
        // K-chunk of the activation slab staged TRANSPOSED ([k][m], the 8
        // batch values contiguous per k): the per-weight inner step is then
        // two half4 vector reads + two float4 FMAs. 1024 × 8 halfs = 16 KB
        // (fewer threadgroup-wide barrier joins than a smaller chunk).
        constexpr uint CHUNK = 1024;
        threadgroup half xT[CHUNK * 8];

        const uint n = tgid.x * 128 + tid;   // this thread's W row
        const bool rowLive = n < outDim;     // no early return — barriers
        const uint groupsPerRow = inDim / 64;
        const ulong qBase = qElemOffset + ulong(n) * (inDim / 8);
        const ulong groupBase = ulong(n) * groupsPerRow;

        float4 accLo = 0.0f;
        float4 accHi = 0.0f;

        for (uint k0 = 0; k0 < inDim; k0 += CHUNK) {
            const uint span = min(CHUNK, inDim - k0);
            // Cooperative transpose-stage, vectorized device reads (span is
            // a multiple of 64, so half4 rows are always full; alignment
            // holds — inDim is a multiple of 64 and k steps by 4). Batch
            // rows past M stage zeros.
            for (uint idx = tid; idx < span * 2; idx += 128) {
                const uint m = idx % 8;
                const uint k4 = idx / 8;
                const half4 v = (m < batchM)
                    ? *(device const half4 *)(x + ulong(m) * inDim + k0 + 4 * k4)
                    : half4(0.0f);
                xT[(4 * k4 + 0) * 8 + m] = v.x;
                xT[(4 * k4 + 1) * 8 + m] = v.y;
                xT[(4 * k4 + 2) * 8 + m] = v.z;
                xT[(4 * k4 + 3) * 8 + m] = v.w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (rowLive) {
                // span is a multiple of 64 (K and CHUNK both are).
                const uint g0 = k0 / 64;
                for (uint g = 0; g < span / 64; ++g) {
                    const float s = float(
                        scales[scalesElemOffset + groupBase + g0 + g]);
                    const float b = float(
                        biases[biasesElemOffset + groupBase + g0 + g]);
                    uint kc = g * 64;
                    for (uint w = 0; w < 8; ++w) {
                        uint word = q[qBase + ulong(g0 + g) * 8 + w];
                        for (uint lane = 0; lane < 8; ++lane) {
                            // The pinned Q4G64.dequant arithmetic, verbatim.
                            const float wt = float(word & 0xFu) * s + b;
                            const threadgroup half4 *xk =
                                (const threadgroup half4 *)(xT + kc * 8);
                            accLo += wt * float4(xk[0]);
                            accHi += wt * float4(xk[1]);
                            word >>= 4;
                            ++kc;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (rowLive) {
            const float accArr[8] = {
                accLo.x, accLo.y, accLo.z, accLo.w,
                accHi.x, accHi.y, accHi.z, accHi.w,
            };
            for (uint m = 0; m < 8; ++m) {
                if (m < batchM) out[ulong(m) * outDim + n] = half(accArr[m]);
            }
        }
    }
    """

    /// batchM at or below which the register-blocked small-batch
    /// specialization dispatches (chosen by measurement — see the m8
    /// kernel's comment).
    static let smallBatchMaxM = 8

    private let pipeline: MTLComputePipelineState
    private let smallBatchPipeline: MTLComputePipelineState

    /// When set, every encoded dispatch increments it at the
    /// dispatch call site (P2-5 instrumentation convention).
    var dispatchCounter: DispatchCounter?

    public init(context: MetalContext) throws {
        let library = try context.makeLibrary(source: Self.source)
        pipeline = try context.makeComputePipeline(
            library: library, function: "gemm_q4_f16")
        smallBatchPipeline = try context.makeComputePipeline(
            library: library, function: "gemm_q4_f16_m8")
        // 128 threads/threadgroup is structural for BOTH kernels (the tiled
        // kernel's 4-simdgroup tile ownership map; the m8 kernel's
        // cooperative-staging stride and barrier participation). A pipeline
        // that cannot grant them would compute undefined tiles — fail
        // loudly instead (never expected on Apple GPUs; belt-and-braces
        // like the wrapper validation).
        for candidate in [pipeline, smallBatchPipeline]
        where candidate.maxTotalThreadsPerThreadgroup < Self.threadsPerThreadgroup {
            throw MetalHarnessError.libraryCompileFailed(
                underlying: QuantKernelError.nonPositiveDimension(
                    name: "maxTotalThreadsPerThreadgroup",
                    value: candidate.maxTotalThreadsPerThreadgroup))
        }
    }

    /// out[batchM, outDim] = input[batchM, inDim] · dequant(W)ᵀ with W the
    /// packed [outDim, inDim] triplet. Buffer/offset conventions match
    /// `QuantKernels.encodeMatvec` (element offsets, never setBuffer
    /// offsets); validation is the shared triplet rule plus the M-side
    /// checks.
    public func encodeGemm(
        into encoder: MTLComputeCommandEncoder,
        q: MTLBuffer, qByteOffset: Int,
        scales: MTLBuffer, scalesByteOffset: Int,
        biases: MTLBuffer, biasesByteOffset: Int,
        input: MTLBuffer, batchM: Int, outDim: Int, inDim: Int,
        output: MTLBuffer
    ) throws {
        let offsets = try QuantKernels.tripletElementOffsets(
            q: q, qByteOffset: qByteOffset,
            scales: scales, scalesByteOffset: scalesByteOffset,
            biases: biases, biasesByteOffset: biasesByteOffset,
            outDim: outDim, inDim: inDim)
        guard batchM > 0 else {
            throw QuantKernelError.nonPositiveDimension(
                name: "batchM", value: batchM)
        }
        try QuantKernels.requireCapacity(
            input, bytes: batchM * inDim * 2, name: "input")
        try QuantKernels.requireCapacity(
            output, bytes: batchM * outDim * 2, name: "output")

        let useSmallBatch = batchM <= Self.smallBatchMaxM
        encoder.setComputePipelineState(
            useSmallBatch ? smallBatchPipeline : pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        setScalar(encoder, UInt64(offsets.q), index: 1)
        encoder.setBuffer(scales, offset: 0, index: 2)
        setScalar(encoder, UInt64(offsets.scales), index: 3)
        encoder.setBuffer(biases, offset: 0, index: 4)
        setScalar(encoder, UInt64(offsets.biases), index: 5)
        encoder.setBuffer(input, offset: 0, index: 6)
        setScalar(encoder, UInt32(batchM), index: 7)
        setScalar(encoder, UInt32(outDim), index: 8)
        setScalar(encoder, UInt32(inDim), index: 9)
        encoder.setBuffer(output, offset: 0, index: 10)

        dispatchCounter?.increment()
        if useSmallBatch {
            // One thread per output row, full 128-thread threadgroups (all
            // threads stay alive for the cooperative-staging barriers; rows
            // past outDim are guarded in-kernel).
            encoder.dispatchThreadgroups(
                MTLSize(width: (outDim + 127) / 128, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        } else {
            encoder.dispatchThreadgroups(
                MTLSize(
                    width: (outDim + Self.tileN - 1) / Self.tileN,
                    height: (batchM + Self.tileM - 1) / Self.tileM,
                    depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: Self.threadsPerThreadgroup, height: 1, depth: 1))
        }
    }

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
}
