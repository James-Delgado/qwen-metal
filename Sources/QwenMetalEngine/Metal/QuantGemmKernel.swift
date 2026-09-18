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
///   `gemm_q4_f16_m8` (batchM ≤ 8, the microbench gate point): a lane-split
///   multi-matvec (K across the 32 lanes of a simdgroup, several rows per
///   simdgroup in flight, fp32 activations in registers) — the P5-2B
///   redesign of the P5-2 thread-per-row path; see its comment.
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

    // Small-batch path (M ≤ 8) — the D7 microbench gate point. P5-2B design,
    // reached by measurement on the per-role attribution (DECISIONS.md
    // 2026-09-18 P5-2B, the iteration ledger): the P5-2 thread-per-row
    // multi-matvec launched only outDim/128 threadgroups (8 for k/v_proj,
    // 16 for q/o/down) and its per-role stream rate tracked that count —
    // parallelism starvation, not a bandwidth limit — while every weight
    // also cost 8 fp16→fp32 converts of the staged activations per thread.
    // Geometry (M8_S = 8 lanes per row, M8_R = 2 rows per lane, 8
    // simdgroups = 256 threads, 16 KB chunk) is the Mac-measured optimum of
    // the ledger's grid; the constants live on the Swift side so a device
    // re-tune is a constant change.
    //
    // Here a simdgroup is 32/M8_S row-groups of M8_S lanes. Within a
    // row-group, K is split across the M8_S lanes: lane slot t owns one
    // packed u32 word (8 codes) at k = step·8·M8_S + 8t, so the group's
    // load of one row is a contiguous 4·M8_S-byte run; each lane
    // accumulates M8_R rows concurrently (independent loads in flight), so
    // a simdgroup streams (32/M8_S)·M8_R rows and threadgroups =
    // ⌈outDim / (M8_SIMDGROUPS·(32/M8_S)·M8_R)⌉ at 32·M8_SIMDGROUPS threads
    // each (the staging cost amortizes over the threadgroup). The activation
    // chunk is staged ONCE per threadgroup as fp32 (16 KB) so the inner
    // loop's x operands are plain float4 register loads (no per-weight
    // converts); the 32/M8_S row-groups read the SAME addresses at the same
    // time (simdgroup broadcast — the P3-4 matvec's memory pattern) and the
    // M8_R rows per lane share each loaded vector, so threadgroup-memory
    // traffic is M8_S/M8_R bytes per weight — 8 B for the pure 32-lane
    // split that measured as the ceiling (DECISIONS.md P5-2B iteration
    // ledger), 4 B here. Dequant
    // stays in registers (hard rule 1), the pinned Q4G64.dequant arithmetic
    // verbatim, fp32 accumulation; per-lane partials merge through a
    // fixed-order xor butterfly across the row-group (deterministic; the
    // reduction ORDER differs from matvec/sgemm, which the Tier K gate
    // absorbs per the 2026-09-12 decision). The body is written with NAMED
    // registers (macros, no arrays) — an array-based draft measured at
    // 19 GB/s because the accumulators landed in private memory.
    // Preprocessor-visible copies (the #if guards below need them).
    #define M8_S_DEF \(smallBatchLanesPerRow)
    #define M8_R_DEF \(smallBatchRowsPerLane)
    constant constexpr uint M8_S = M8_S_DEF;     // lanes splitting K per row-group
    constant constexpr uint M8_R = M8_R_DEF;     // rows per lane in flight
    constant constexpr uint M8_GROUPS = 32 / M8_S;               // row-groups per simdgroup
    constant constexpr uint M8_ROWS_PER_SG = M8_GROUPS * M8_R;
    constant constexpr uint M8_SIMDGROUPS = \(smallBatchSimdgroupsPerThreadgroup);
    constant constexpr uint M8_THREADS = 32 * M8_SIMDGROUPS;
    constant constexpr uint M8_KC = 512;   // fp32 activation chunk, 16 KB
    constant constexpr uint M8_LK = M8_KC / 8;   // words per chunk row (lane-k slots)
    constant constexpr uint M8_STEP = 8 * M8_S;  // k per step per row-group
    // The store maps a row-group's slot to a batch row (slot m stores
    // out[m]), so a row-group needs ≥ 8 lanes; 4 was tried and fails the
    // spot check by construction.
    static_assert(M8_S == 8 || M8_S == 16 || M8_S == 32, "M8_S must cover 8 batch rows");
    static_assert(M8_R >= 1 && M8_R <= 4, "the m8 body below is unrolled for up to 4 rows");
    static_assert(M8_KC % M8_STEP == 0, "chunk must hold whole steps");

    // Named registers: accLo_r / accHi_r hold batch rows 0–3 / 4–7 of this
    // lane's row r; xlo_j / xhi_j hold the same split of the activation
    // column at this lane's code j.
    #define M8_LOAD_ROW(r) { \
        const uint n = min(rowBase + r, outDim - 1u); \
        word##r = q[qElemOffset + ulong(n) * wordsPerRow + wordIndex]; \
        const ulong g = ulong(n) * groupsPerRow + group; \
        s##r = float(scales[scalesElemOffset + g]); \
        b##r = float(biases[biasesElemOffset + g]); }
    // The pinned Q4G64.dequant arithmetic, verbatim, then 8 fp32 FMAs.
    #define M8_CODE(r, j) { \
        const float wt = float(word##r & 0xFu) * s##r + b##r; \
        word##r >>= 4; \
        accLo##r += wt * xlo##j; accHi##r += wt * xhi##j; }
    #define M8_ROW_HALF(r) M8_CODE(r, 0) M8_CODE(r, 1) M8_CODE(r, 2) M8_CODE(r, 3)
    // Plane (h, j, mh) of the staged chunk: float4 over batch rows 4mh..4mh+3
    // at k = 8·lk + 4h + j, indexed by lk — the M8_S slots of a row-group
    // read consecutive 16-byte vectors (bank-conflict-free) and the
    // 32/M8_S row-groups read the same ones (broadcast).
    #define M8_XPLANE(h, j, mh) (xs + (((h) * 8 + (j) * 2 + (mh)) * M8_LK + lk) * 4)
    #define M8_LOAD_X(h) \
        xlo0 = *(const threadgroup float4 *)M8_XPLANE(h, 0, 0); \
        xhi0 = *(const threadgroup float4 *)M8_XPLANE(h, 0, 1); \
        xlo1 = *(const threadgroup float4 *)M8_XPLANE(h, 1, 0); \
        xhi1 = *(const threadgroup float4 *)M8_XPLANE(h, 1, 1); \
        xlo2 = *(const threadgroup float4 *)M8_XPLANE(h, 2, 0); \
        xhi2 = *(const threadgroup float4 *)M8_XPLANE(h, 2, 1); \
        xlo3 = *(const threadgroup float4 *)M8_XPLANE(h, 3, 0); \
        xhi3 = *(const threadgroup float4 *)M8_XPLANE(h, 3, 1);
    // Fixed-order xor butterfly over the row-group's M8_S lanes (every lane
    // of the group ends with the group total); then slot m of the group
    // keeps batch row m of this lane's row r.
    #define M8_BUTTERFLY(v) { \
        if (M8_S >= 2)  v += simd_shuffle_xor(v, 1); \
        if (M8_S >= 4)  v += simd_shuffle_xor(v, 2); \
        if (M8_S >= 8)  v += simd_shuffle_xor(v, 4); \
        if (M8_S >= 16) v += simd_shuffle_xor(v, 8); \
        if (M8_S >= 32) v += simd_shuffle_xor(v, 16); }
    #define M8_MERGE(r) { \
        M8_BUTTERFLY(accLo##r) M8_BUTTERFLY(accHi##r) \
        mine##r = (slot == 0) ? accLo##r.x : mine##r; mine##r = (slot == 1) ? accLo##r.y : mine##r; \
        mine##r = (slot == 2) ? accLo##r.z : mine##r; mine##r = (slot == 3) ? accLo##r.w : mine##r; \
        mine##r = (slot == 4) ? accHi##r.x : mine##r; mine##r = (slot == 5) ? accHi##r.y : mine##r; \
        mine##r = (slot == 6) ? accHi##r.z : mine##r; mine##r = (slot == 7) ? accHi##r.w : mine##r; }
    #define M8_STORE(r) { \
        const uint n = rowBase + r; \
        if (n < outDim && slot < batchM) out[ulong(slot) * outDim + n] = half(mine##r); }
    #if M8_R_DEF >= 2
    #define M8_IF2(x) x
    #else
    #define M8_IF2(x)
    #endif
    #if M8_R_DEF >= 3
    #define M8_IF3(x) x
    #else
    #define M8_IF3(x)
    #endif
    #if M8_R_DEF >= 4
    #define M8_IF4(x) x
    #else
    #define M8_IF4(x)
    #endif

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
                               uint tid  [[thread_index_in_threadgroup]],
                               uint sgid [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
        // Activation chunk as 16 planes of M8_LK float4s (see M8_XPLANE);
        // rows past M are zero.
        threadgroup float xs[8 * M8_KC];

        const uint wordsPerRow = inDim / 8;
        const uint groupsPerRow = inDim / 64;
        const uint slot = lane % M8_S;          // this lane's k-slot in its row-group
        const uint rowGroup = lane / M8_S;
        const uint rowBase = (tgid.x * M8_SIMDGROUPS + sgid) * M8_ROWS_PER_SG
                             + rowGroup * M8_R;

        float4 accLo0 = 0.0f, accHi0 = 0.0f, accLo1 = 0.0f, accHi1 = 0.0f;
        float4 accLo2 = 0.0f, accHi2 = 0.0f, accLo3 = 0.0f, accHi3 = 0.0f;

        for (uint k0 = 0; k0 < inDim; k0 += M8_KC) {
            // span is a multiple of 64 (inDim and M8_KC both are) — every
            // staged half4 is full; alignment holds (row starts are 128-byte
            // multiples, k steps by 4). Batch rows past M stage zeros so
            // their accumulators stay exactly 0 (never stored anyway).
            const uint span = min(M8_KC, inDim - k0);
            const uint vecsPerRow = span / 4;
            for (uint idx = tid; idx < 8 * vecsPerRow; idx += M8_THREADS) {
                const uint m = idx / vecsPerRow;
                const uint k4 = idx - m * vecsPerRow;
                const float4 v = (m < batchM)
                    ? float4(*(device const half4 *)(x + ulong(m) * inDim + k0 + 4 * k4))
                    : float4(0.0f);
                const uint lk = k4 >> 1;
                const uint h = k4 & 1;
                const uint mh = m >> 2;
                const uint c = m & 3;
                xs[((h * 8 + 0 * 2 + mh) * M8_LK + lk) * 4 + c] = v.x;
                xs[((h * 8 + 1 * 2 + mh) * M8_LK + lk) * 4 + c] = v.y;
                xs[((h * 8 + 2 * 2 + mh) * M8_LK + lk) * 4 + c] = v.z;
                xs[((h * 8 + 3 * 2 + mh) * M8_LK + lk) * 4 + c] = v.w;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint step = 0; step < M8_KC; step += M8_STEP) {
                const uint kLocal = step + 8 * slot;
                // Slots past the chunk's span (the ragged tail of a short K)
                // contribute nothing; span is a multiple of 64, so the live
                // slots form a prefix.
                if (kLocal < span) {
                    const uint k = k0 + kLocal;
                    const uint wordIndex = k / 8;
                    const uint group = k / 64;
                    const uint lk = kLocal / 8;
                    // This lane's word + group constants for each in-flight
                    // row (independent loads; rows past outDim are clamped
                    // to a live row for the read and never stored).
                    uint word0, word1, word2, word3;
                    float s0, s1, s2, s3, b0, b1, b2, b3;
                    M8_LOAD_ROW(0) M8_IF2(M8_LOAD_ROW(1)) M8_IF3(M8_LOAD_ROW(2)) M8_IF4(M8_LOAD_ROW(3))
                    float4 xlo0, xhi0, xlo1, xhi1, xlo2, xhi2, xlo3, xhi3;
                    // Codes 0–3 then 4–7: 32 activation registers per half,
                    // shared by all of this lane's rows.
                    M8_LOAD_X(0)
                    M8_ROW_HALF(0) M8_IF2(M8_ROW_HALF(1)) M8_IF3(M8_ROW_HALF(2)) M8_IF4(M8_ROW_HALF(3))
                    M8_LOAD_X(1)
                    M8_ROW_HALF(0) M8_IF2(M8_ROW_HALF(1)) M8_IF3(M8_ROW_HALF(2)) M8_IF4(M8_ROW_HALF(3))
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Cross-lane merge per row-group, then slot m stores out[m][row r].
        float mine0 = 0.0f, mine1 = 0.0f, mine2 = 0.0f, mine3 = 0.0f;
        M8_MERGE(0) M8_IF2(M8_MERGE(1)) M8_IF3(M8_MERGE(2)) M8_IF4(M8_MERGE(3))
        if (slot < 8) {
            M8_STORE(0) M8_IF2(M8_STORE(1)) M8_IF3(M8_STORE(2)) M8_IF4(M8_STORE(3))
        }
    }
    #undef M8_LOAD_ROW
    #undef M8_CODE
    #undef M8_ROW_HALF
    #undef M8_XPLANE
    #undef M8_LOAD_X
    #undef M8_BUTTERFLY
    #undef M8_MERGE
    #undef M8_STORE
    #undef M8_IF2
    #undef M8_IF3
    #undef M8_IF4
    #undef M8_S_DEF
    #undef M8_R_DEF
    """

    /// batchM at or below which the small-batch specialization dispatches
    /// (chosen by measurement — see the m8 kernel's comment).
    static let smallBatchMaxM = 8
    /// m8 kernel geometry (chosen by measurement — see the kernel's comment
    /// and the DECISIONS.md P5-2B ledger): lanes splitting K per row-group
    /// and rows each lane accumulates concurrently. A simdgroup streams
    /// (32 / lanesPerRow) · rowsPerLane rows; a 128-thread threadgroup
    /// (4 simdgroups) covers 4× that.
    static let smallBatchLanesPerRow = 8
    static let smallBatchRowsPerLane = 2
    static let smallBatchSimdgroupsPerThreadgroup = 8
    static var smallBatchThreadsPerThreadgroup: Int {
        32 * smallBatchSimdgroupsPerThreadgroup
    }
    static var smallBatchRowsPerThreadgroup: Int {
        smallBatchSimdgroupsPerThreadgroup * (32 / smallBatchLanesPerRow)
            * smallBatchRowsPerLane
    }

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
        // The threadgroup sizes are structural for BOTH kernels (the tiled
        // kernel's 4-simdgroup tile ownership map at 128; the m8 kernel's
        // cooperative-staging stride and barrier participation at its own
        // count). A pipeline that cannot grant them would compute undefined
        // tiles — fail loudly instead (never expected on Apple GPUs;
        // belt-and-braces like the wrapper validation).
        for (candidate, required) in [
            (pipeline, Self.threadsPerThreadgroup),
            (smallBatchPipeline, Self.smallBatchThreadsPerThreadgroup),
        ] where candidate.maxTotalThreadsPerThreadgroup < required {
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
            // smallBatchRowsPerThreadgroup rows per threadgroup of
            // smallBatchThreadsPerThreadgroup threads (all stay alive for
            // the staging barriers; rows past outDim are guarded in-kernel).
            let rows = Self.smallBatchRowsPerThreadgroup
            encoder.dispatchThreadgroups(
                MTLSize(width: (outDim + rows - 1) / rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: Self.smallBatchThreadsPerThreadgroup, height: 1, depth: 1))
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
