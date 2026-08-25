import Metal

/// Errors from the decode-kernel host wrappers. Every validation failure is
/// explicit and named (METHODOLOGY: clear errors at system boundaries) — a bad
/// dimension or short buffer must never reach a GPU dispatch.
public enum DecodeKernelError: Error, CustomStringConvertible, Equatable {
    case nonPositiveDimension(name: String, value: Int)
    case misalignedWeightOffset(byteOffset: Int)
    case bufferTooSmall(buffer: String, requiredBytes: Int, actualBytes: Int)
    case tokenIdOutOfRange(id: Int, vocabSize: Int)
    case positionOutOfRange(position: Int, positions: Int)
    case oddHeadDim(headDim: Int)

    public var description: String {
        switch self {
        case .nonPositiveDimension(let name, let value):
            return "Kernel dimension \(name) must be positive, got \(value)"
        case .misalignedWeightOffset(let byteOffset):
            return "Weight byte offset \(byteOffset) is not 2-byte aligned; "
                + "16-bit weight loads require even offsets (the pinned "
                + "checkpoint guarantees this — see GPUWeightsTests spot check)"
        case .bufferTooSmall(let buffer, let required, let actual):
            return "Buffer '\(buffer)' holds \(actual) bytes, kernel needs \(required)"
        case .tokenIdOutOfRange(let id, let vocabSize):
            return "Token id \(id) outside embedding table vocab \(vocabSize)"
        case .positionOutOfRange(let position, let positions):
            return "RoPE position \(position) outside the \(positions)-position table"
        case .oddHeadDim(let headDim):
            return "RoPE headDim \(headDim) must be even (half-split rotation)"
        }
    }
}

/// P2-2 (docs/phases/phase-2.md D4): the naive non-attention decode kernels —
/// embedding-lookup, rmsnorm, matvec, rope, swiglu, residual-add. All
/// one-thread-per-output-element, deliberately unoptimized (hard rule 3:
/// correctness before any optimization; speed is Phases 3-5's job).
///
/// Precision per spec D2: fp16 activations between kernels, fp32 accumulation
/// and math inside, bf16 weights read as `ushort` and upcast in registers via
/// integer shift + bitcast (spec D1 — exact, nothing materialized, hard rule 1).
/// Weights are addressed by ELEMENT offset into the whole-checkpoint buffer;
/// wrappers reject odd byte offsets loudly rather than corrupt a load.
///
/// Methods ENCODE into a caller-supplied encoder rather than dispatching —
/// P2-4 packs one command buffer per decoded token (~500 dispatches, spec D5),
/// and the Tier-K tests drive the same methods through
/// `MetalContext.timedDispatch` (hard rule 7: timing always rides along).
public final class DecodeKernels {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // bf16 -> fp32 exact upcast (spec D1): identical to the CPU reference's,
    // so weight bits contribute zero divergence to any gate.
    inline float bf16_to_f32(ushort w) {
        return as_type<float>(uint(w) << 16);
    }

    // Row gather from the bf16 [vocab, hidden] table + fp16 store. Pure
    // convert chain (no arithmetic) => pre-committed EXACT gate.
    kernel void embedding_lookup(device const ushort *table [[buffer(0)]],
                                 constant ulong &tableElemOffset [[buffer(1)]],
                                 constant uint &tokenId     [[buffer(2)]],
                                 constant uint &hiddenSize  [[buffer(3)]],
                                 device half *out           [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
        if (gid >= hiddenSize) return;
        ushort w = table[tableElemOffset + ulong(tokenId) * hiddenSize + gid];
        out[gid] = half(bf16_to_f32(w));
    }

    // RMSNorm rows of `dim` values (Qwen3RMSNorm semantics, matching the CPU
    // reference exactly in structure): rows=1/dim=hidden serves the block
    // norms, rows=heads/dim=headDim serves the per-head QK-norm. Every thread
    // recomputes its row's mean square sequentially in fp32 — same order for
    // all threads, naive by design.
    kernel void rmsnorm_f16(device const half *x           [[buffer(0)]],
                            device const ushort *weight    [[buffer(1)]],
                            constant ulong &weightElemOffset [[buffer(2)]],
                            constant uint &rows            [[buffer(3)]],
                            constant uint &dim             [[buffer(4)]],
                            constant float &eps            [[buffer(5)]],
                            device half *out               [[buffer(6)]],
                            uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dim || gid.y >= rows) return;
        ulong base = ulong(gid.y) * dim;
        float sumOfSquares = 0.0f;
        for (uint j = 0; j < dim; ++j) {
            float v = float(x[base + j]);
            sumOfSquares += v * v;
        }
        float inverseRMS = 1.0f / sqrt(sumOfSquares / float(dim) + eps);
        float w = bf16_to_f32(weight[weightElemOffset + gid.x]);
        out[base + gid.x] = half(w * (float(x[base + gid.x]) * inverseRMS));
    }

    // y[r] = sum_c W[r,c] * x[c], W bf16 [out, in] row-major (HF storage),
    // fp32 accumulation. Serves QKV/o/gate/up/down projections, and lm_head
    // by consuming the tied [vocab, hidden] embedding table directly as
    // [out, in] — no transpose is ever materialized.
    kernel void matvec_bf16_f16(device const ushort *weights [[buffer(0)]],
                                constant ulong &weightElemOffset [[buffer(1)]],
                                device const half *x         [[buffer(2)]],
                                constant uint &outDim        [[buffer(3)]],
                                constant uint &inDim         [[buffer(4)]],
                                device half *out             [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
        if (gid >= outDim) return;
        ulong rowBase = weightElemOffset + ulong(gid) * inDim;
        float acc = 0.0f;
        for (uint c = 0; c < inDim; ++c) {
            acc += bf16_to_f32(weights[rowBase + c]) * float(x[c]);
        }
        out[gid] = half(acc);
    }

    // fp32-store variant: logits stay fp32 (spec D2 — the gate suite and
    // argmax consume fp32 logits).
    kernel void matvec_bf16_f32(device const ushort *weights [[buffer(0)]],
                                constant ulong &weightElemOffset [[buffer(1)]],
                                device const half *x         [[buffer(2)]],
                                constant uint &outDim        [[buffer(3)]],
                                constant uint &inDim         [[buffer(4)]],
                                device float *out            [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
        if (gid >= outDim) return;
        ulong rowBase = weightElemOffset + ulong(gid) * inDim;
        float acc = 0.0f;
        for (uint c = 0; c < inDim; ++c) {
            acc += bf16_to_f32(weights[rowBase + c]) * float(x[c]);
        }
        out[gid] = acc;
    }

    // Half-split rotation for ONE token at absolute `position`, in place over
    // [heads, headDim] fp16. One thread per (head, pair): reads exactly the
    // two elements it writes, so in-place is race-free. cos/sin tables are
    // the CPU RoPE's fp32 tables ([positions, headDim/2]) — bit-identical
    // angles by construction.
    kernel void rope_f16(device half *x               [[buffer(0)]],
                         device const float *cosTable [[buffer(1)]],
                         device const float *sinTable [[buffer(2)]],
                         constant uint &position      [[buffer(3)]],
                         constant uint &heads         [[buffer(4)]],
                         constant uint &headDim       [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
        uint halfDim = headDim / 2;
        if (gid.x >= halfDim || gid.y >= heads) return;
        ulong base = ulong(gid.y) * headDim;
        float c = cosTable[ulong(position) * halfDim + gid.x];
        float sn = sinTable[ulong(position) * halfDim + gid.x];
        float x1 = float(x[base + gid.x]);
        float x2 = float(x[base + halfDim + gid.x]);
        x[base + gid.x] = half(x1 * c - x2 * sn);
        x[base + halfDim + gid.x] = half(x2 * c + x1 * sn);
    }

    // silu(gate) * up elementwise, fp32 math (CPU MLP's exact formula).
    kernel void swiglu_f16(device const half *gate [[buffer(0)]],
                           device const half *up   [[buffer(1)]],
                           constant uint &count    [[buffer(2)]],
                           device half *out        [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
        if (gid >= count) return;
        float g = float(gate[gid]);
        float u = float(up[gid]);
        out[gid] = half((g / (1.0f + exp(-g))) * u);
    }

    // a + b in fp32, single round to fp16 on store.
    kernel void residual_add_f16(device const half *a  [[buffer(0)]],
                                 device const half *b  [[buffer(1)]],
                                 constant uint &count  [[buffer(2)]],
                                 device half *out      [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
        if (gid >= count) return;
        out[gid] = half(float(a[gid]) + float(b[gid]));
    }
    """

    private let embeddingPipeline: MTLComputePipelineState
    private let rmsnormPipeline: MTLComputePipelineState
    private let matvecF16Pipeline: MTLComputePipelineState
    private let matvecF32Pipeline: MTLComputePipelineState
    private let ropePipeline: MTLComputePipelineState
    private let swigluPipeline: MTLComputePipelineState
    private let residualPipeline: MTLComputePipelineState

    /// When set, every encoded dispatch increments it at the dispatchThreads
    /// call site (P2-5 instrumentation; `GPUModel` attaches its per-step
    /// counter). nil (the default) counts nothing.
    var dispatchCounter: DispatchCounter?

    public init(context: MetalContext) throws {
        let library = try context.makeLibrary(source: Self.source)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            try context.makeComputePipeline(library: library, function: name)
        }
        embeddingPipeline = try pipeline("embedding_lookup")
        rmsnormPipeline = try pipeline("rmsnorm_f16")
        matvecF16Pipeline = try pipeline("matvec_bf16_f16")
        matvecF32Pipeline = try pipeline("matvec_bf16_f32")
        ropePipeline = try pipeline("rope_f16")
        swigluPipeline = try pipeline("swiglu_f16")
        residualPipeline = try pipeline("residual_add_f16")
    }

    // MARK: - Encode methods (one per kernel)

    /// Gathers embedding row `tokenId` from the bf16 [vocabSize, hiddenSize]
    /// table at `tableByteOffset` into `output` ([hiddenSize] fp16).
    public func encodeEmbeddingLookup(
        into encoder: MTLComputeCommandEncoder,
        table: MTLBuffer, tableByteOffset: Int, vocabSize: Int, hiddenSize: Int,
        tokenId: Int, output: MTLBuffer
    ) throws {
        try requirePositive(vocabSize, "vocabSize")
        try requirePositive(hiddenSize, "hiddenSize")
        guard tokenId >= 0, tokenId < vocabSize else {
            throw DecodeKernelError.tokenIdOutOfRange(id: tokenId, vocabSize: vocabSize)
        }
        let elemOffset = try weightElementOffset(
            byteOffset: tableByteOffset, elementCount: vocabSize * hiddenSize,
            buffer: table, name: "table")
        try requireCapacity(output, bytes: hiddenSize * 2, name: "output")

        encoder.setComputePipelineState(embeddingPipeline)
        encoder.setBuffer(table, offset: 0, index: 0)
        setScalar(encoder, UInt64(elemOffset), index: 1)
        setScalar(encoder, UInt32(tokenId), index: 2)
        setScalar(encoder, UInt32(hiddenSize), index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        dispatch1D(encoder, pipeline: embeddingPipeline, count: hiddenSize)
    }

    /// Normalizes `rows` consecutive `dim`-length fp16 rows of `input` with
    /// the bf16 weight vector at `weightByteOffset` (Qwen3RMSNorm semantics).
    public func encodeRMSNorm(
        into encoder: MTLComputeCommandEncoder,
        input: MTLBuffer, weight: MTLBuffer, weightByteOffset: Int,
        rows: Int, dim: Int, eps: Float, output: MTLBuffer
    ) throws {
        try requirePositive(rows, "rows")
        try requirePositive(dim, "dim")
        let elemOffset = try weightElementOffset(
            byteOffset: weightByteOffset, elementCount: dim,
            buffer: weight, name: "weight")
        try requireCapacity(input, bytes: rows * dim * 2, name: "input")
        try requireCapacity(output, bytes: rows * dim * 2, name: "output")

        encoder.setComputePipelineState(rmsnormPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        setScalar(encoder, UInt64(elemOffset), index: 2)
        setScalar(encoder, UInt32(rows), index: 3)
        setScalar(encoder, UInt32(dim), index: 4)
        setScalar(encoder, eps, index: 5)
        encoder.setBuffer(output, offset: 0, index: 6)
        dispatch2D(encoder, pipeline: rmsnormPipeline, width: dim, height: rows)
    }

    /// y = W·x with W bf16 [outDim, inDim] row-major at `weightByteOffset`,
    /// x fp16 [inDim]; output fp16 (projections) or fp32 (lm_head logits).
    public func encodeMatvec(
        into encoder: MTLComputeCommandEncoder,
        weights: MTLBuffer, weightByteOffset: Int,
        input: MTLBuffer, outDim: Int, inDim: Int,
        output: MTLBuffer, fp32Output: Bool = false
    ) throws {
        try requirePositive(outDim, "outDim")
        try requirePositive(inDim, "inDim")
        let elemOffset = try weightElementOffset(
            byteOffset: weightByteOffset, elementCount: outDim * inDim,
            buffer: weights, name: "weights")
        try requireCapacity(input, bytes: inDim * 2, name: "input")
        try requireCapacity(
            output, bytes: outDim * (fp32Output ? 4 : 2), name: "output")

        let pipeline = fp32Output ? matvecF32Pipeline : matvecF16Pipeline
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: 0, index: 0)
        setScalar(encoder, UInt64(elemOffset), index: 1)
        encoder.setBuffer(input, offset: 0, index: 2)
        setScalar(encoder, UInt32(outDim), index: 3)
        setScalar(encoder, UInt32(inDim), index: 4)
        encoder.setBuffer(output, offset: 0, index: 5)
        dispatch1D(encoder, pipeline: pipeline, count: outDim)
    }

    /// Rotates one token's [heads, headDim] fp16 vector in place at absolute
    /// `position`, using fp32 cos/sin tables of shape [positions, headDim/2]
    /// (take them from the CPU `RoPE`'s `cosValues`/`sinValues`).
    public func encodeRoPE(
        into encoder: MTLComputeCommandEncoder,
        vector: MTLBuffer, cosTable: MTLBuffer, sinTable: MTLBuffer,
        position: Int, positions: Int, heads: Int, headDim: Int
    ) throws {
        try requirePositive(heads, "heads")
        try requirePositive(headDim, "headDim")
        guard headDim % 2 == 0 else {
            throw DecodeKernelError.oddHeadDim(headDim: headDim)
        }
        try requirePositive(positions, "positions")
        guard position >= 0, position < positions else {
            throw DecodeKernelError.positionOutOfRange(
                position: position, positions: positions)
        }
        let halfDim = headDim / 2
        try requireCapacity(vector, bytes: heads * headDim * 2, name: "vector")
        try requireCapacity(cosTable, bytes: positions * halfDim * 4, name: "cosTable")
        try requireCapacity(sinTable, bytes: positions * halfDim * 4, name: "sinTable")

        encoder.setComputePipelineState(ropePipeline)
        encoder.setBuffer(vector, offset: 0, index: 0)
        encoder.setBuffer(cosTable, offset: 0, index: 1)
        encoder.setBuffer(sinTable, offset: 0, index: 2)
        setScalar(encoder, UInt32(position), index: 3)
        setScalar(encoder, UInt32(heads), index: 4)
        setScalar(encoder, UInt32(headDim), index: 5)
        dispatch2D(encoder, pipeline: ropePipeline, width: halfDim, height: heads)
    }

    /// out = silu(gate) · up elementwise over `count` fp16 values.
    public func encodeSwiGLU(
        into encoder: MTLComputeCommandEncoder,
        gate: MTLBuffer, up: MTLBuffer, count: Int, output: MTLBuffer
    ) throws {
        try requirePositive(count, "count")
        try requireCapacity(gate, bytes: count * 2, name: "gate")
        try requireCapacity(up, bytes: count * 2, name: "up")
        try requireCapacity(output, bytes: count * 2, name: "output")

        encoder.setComputePipelineState(swigluPipeline)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        setScalar(encoder, UInt32(count), index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        dispatch1D(encoder, pipeline: swigluPipeline, count: count)
    }

    /// out = a + b elementwise over `count` fp16 values (fp32 add, one round).
    public func encodeResidualAdd(
        into encoder: MTLComputeCommandEncoder,
        a: MTLBuffer, b: MTLBuffer, count: Int, output: MTLBuffer
    ) throws {
        try requirePositive(count, "count")
        try requireCapacity(a, bytes: count * 2, name: "a")
        try requireCapacity(b, bytes: count * 2, name: "b")
        try requireCapacity(output, bytes: count * 2, name: "output")

        encoder.setComputePipelineState(residualPipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        setScalar(encoder, UInt32(count), index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        dispatch1D(encoder, pipeline: residualPipeline, count: count)
    }

    // MARK: - Validation + dispatch helpers

    private func requirePositive(_ value: Int, _ name: String) throws {
        guard value > 0 else {
            throw DecodeKernelError.nonPositiveDimension(name: name, value: value)
        }
    }

    /// 16-bit weight loads need even byte offsets; the offset is passed to
    /// kernels in ELEMENTS (never as a `setBuffer` offset, whose alignment
    /// rules the safetensors format can violate — GPUWeights convention).
    private func weightElementOffset(
        byteOffset: Int, elementCount: Int, buffer: MTLBuffer, name: String
    ) throws -> Int {
        guard byteOffset >= 0, byteOffset % 2 == 0 else {
            throw DecodeKernelError.misalignedWeightOffset(byteOffset: byteOffset)
        }
        try requireCapacity(
            buffer, bytes: byteOffset + elementCount * 2, name: name)
        return byteOffset / 2
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
        // 16×16 threadgroups like the P0B matmul; dispatchThreads handles the
        // ragged edge (kernel bounds checks are belt-and-braces).
        let side = 16
        let tgWidth = min(side, pipeline.maxTotalThreadsPerThreadgroup)
        let tgHeight = max(1, min(side, pipeline.maxTotalThreadsPerThreadgroup / tgWidth))
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: tgHeight, depth: 1))
    }
}
