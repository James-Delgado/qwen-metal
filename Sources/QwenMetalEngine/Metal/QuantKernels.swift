import Metal

/// Errors from the fused quant-kernel host wrappers. Every validation failure
/// is explicit and named (METHODOLOGY: clear errors at system boundaries) — a
/// bad dimension, misaligned offset, or short buffer must never reach a GPU
/// dispatch.
public enum QuantKernelError: Error, CustomStringConvertible, Equatable {
    case nonPositiveDimension(name: String, value: Int)
    case inDimNotMultipleOfGroup(inDim: Int)
    case misalignedOffset(buffer: String, byteOffset: Int, alignment: Int)
    case bufferTooSmall(buffer: String, requiredBytes: Int, actualBytes: Int)
    case tokenIdOutOfRange(id: Int, vocabSize: Int)

    public var description: String {
        switch self {
        case .nonPositiveDimension(let name, let value):
            return "Quant kernel dimension \(name) must be positive, got \(value)"
        case .inDimNotMultipleOfGroup(let inDim):
            return "Quant kernel inDim \(inDim) is not a multiple of the q4g64 "
                + "group size \(Q4G64.groupSize) — the schema cannot address it"
        case .misalignedOffset(let buffer, let byteOffset, let alignment):
            return "Buffer '\(buffer)' byte offset \(byteOffset) is not "
                + "\(alignment)-byte aligned; typed loads require it (the packed "
                + "loader validates .q alignment at load — see PackedCheckpoint)"
        case .bufferTooSmall(let buffer, let required, let actual):
            return "Buffer '\(buffer)' holds \(actual) bytes, kernel needs \(required)"
        case .tokenIdOutOfRange(let id, let vocabSize):
            return "Token id \(id) outside packed embedding table vocab \(vocabSize)"
        }
    }
}

/// P3-4 (docs/phases/phase-3.md D4): the fused q4g64 kernels — dequant-tile
/// dump (oracle layer 1), fused dequant-matvec (fp16- and fp32-store), and
/// embedding-gather-dequant. Dequant happens in registers inside the consuming
/// kernel (hard rule 1); weights never exist dequantized in DRAM.
///
/// Arithmetic is the pinned `Q4G64.dequant`: `float(q)·float(scale)+float(bias)`.
/// q·scale is exact in fp32 (≤4-bit integer × 11-bit fp16 significand needs
/// ≤15 significand bits), so the expression is ONE correctly rounded operation
/// and equals `fma(q, scale, bias)` bit-for-bit — CPU and GPU dequant values
/// are identical by construction, which is what makes the layer-1 gate EXACT
/// (DECISIONS.md 2026-08-25 Phase 3 gates entry).
///
/// Each packed matrix arrives as the schema D1 triplet, addressed inside one
/// buffer (P3-5 binds the whole-checkpoint `GPUWeights` buffer three times)
/// by byte offsets: `.q` u32 `[out, in/8]` (8 codes per word, low nibble
/// first), `.scales`/`.biases` fp16 `[out, in/64]`. Offsets are passed to
/// kernels in ELEMENTS (GPUWeights convention — never as `setBuffer` offsets);
/// wrappers reject misalignment (q mod 4, scales/biases mod 2) loudly.
///
/// This first landing is deliberately naive (one thread per output element,
/// P2-2 shape): hard rule 3 puts the correctness tests BEFORE any
/// optimization. Spec D4 licenses optimizing THIS kernel afterward
/// (simdgroup reductions, vectorized loads) provided every iteration
/// re-passes the exact/Tier-K suites; the P3-6 microbench grades it.
public final class QuantKernels {
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    // The pinned q4g64 dequant (Q4G64.dequant): one correctly rounded fp32
    // operation — bit-identical to the CPU reference whether or not the
    // compiler contracts it to fma (see the exactness note above).
    inline float dequant_q4(uint code, half scale, half bias) {
        return float(code) * float(scale) + float(bias);
    }

    // Extracts code `lane` (0..7) from a packed u32 — low nibble first
    // (schema pin, == Q4G64.code).
    inline uint q4_code(uint word, uint lane) {
        return (word >> (4 * lane)) & 0xFu;
    }

    // Oracle layer 1 (test support): register-dequant the whole [out, in]
    // matrix to fp32, one thread per element. Every schema bug species —
    // nibble order, group-boundary indexing, scale/bias lookup — is a hard
    // bitwise mismatch here.
    kernel void dequant_tile_f32(device const uint *q        [[buffer(0)]],
                                 constant ulong &qElemOffset [[buffer(1)]],
                                 device const half *scales   [[buffer(2)]],
                                 constant ulong &scalesElemOffset [[buffer(3)]],
                                 device const half *biases   [[buffer(4)]],
                                 constant ulong &biasesElemOffset [[buffer(5)]],
                                 constant uint &outDim       [[buffer(6)]],
                                 constant uint &inDim        [[buffer(7)]],
                                 device float *out           [[buffer(8)]],
                                 uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= inDim || gid.y >= outDim) return;
        uint wordsPerRow = inDim / 8;
        uint groupsPerRow = inDim / 64;
        uint word = q[qElemOffset + ulong(gid.y) * wordsPerRow + gid.x / 8];
        uint code = q4_code(word, gid.x % 8);
        ulong group = ulong(gid.y) * groupsPerRow + gid.x / 64;
        out[ulong(gid.y) * inDim + gid.x] = dequant_q4(
            code, scales[scalesElemOffset + group], biases[biasesElemOffset + group]);
    }

    // Row gather + register dequant + fp16 store (replaces Phase 2's bf16
    // embedding_lookup on the packed path). Store rounds the exact fp32
    // dequant value once to fp16 => pre-committed EXACT gate.
    kernel void embedding_gather_q4_f16(device const uint *q        [[buffer(0)]],
                                        constant ulong &qElemOffset [[buffer(1)]],
                                        device const half *scales   [[buffer(2)]],
                                        constant ulong &scalesElemOffset [[buffer(3)]],
                                        device const half *biases   [[buffer(4)]],
                                        constant ulong &biasesElemOffset [[buffer(5)]],
                                        constant uint &tokenId      [[buffer(6)]],
                                        constant uint &hiddenSize   [[buffer(7)]],
                                        device half *out            [[buffer(8)]],
                                        uint gid [[thread_position_in_grid]]) {
        if (gid >= hiddenSize) return;
        uint word = q[qElemOffset + ulong(tokenId) * (hiddenSize / 8) + gid / 8];
        uint code = q4_code(word, gid % 8);
        ulong group = ulong(tokenId) * (hiddenSize / 64) + gid / 64;
        out[gid] = half(dequant_q4(
            code, scales[scalesElemOffset + group], biases[biasesElemOffset + group]));
    }

    // y[r] = sum_c dequant(W[r,c]) * x[c] over the packed triplet, fp32
    // accumulation, fp16 store. Serves QKV/o/gate/up/down; lm_head consumes
    // the tied packed embedding triplet directly as [out, in] — no transpose
    // is ever materialized (P2-2 precedent).
    kernel void matvec_q4_f16(device const uint *q        [[buffer(0)]],
                              constant ulong &qElemOffset [[buffer(1)]],
                              device const half *scales   [[buffer(2)]],
                              constant ulong &scalesElemOffset [[buffer(3)]],
                              device const half *biases   [[buffer(4)]],
                              constant ulong &biasesElemOffset [[buffer(5)]],
                              device const half *x        [[buffer(6)]],
                              constant uint &outDim       [[buffer(7)]],
                              constant uint &inDim        [[buffer(8)]],
                              device half *out            [[buffer(9)]],
                              uint gid [[thread_position_in_grid]]) {
        if (gid >= outDim) return;
        uint groupsPerRow = inDim / 64;
        ulong qBase = qElemOffset + ulong(gid) * (inDim / 8);
        ulong groupBase = ulong(gid) * groupsPerRow;
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
        out[gid] = half(acc);
    }

    // fp32-store variant: logits stay fp32 (Phase 2 spec D2 — the gate suite
    // and argmax consume fp32 logits).
    kernel void matvec_q4_f32(device const uint *q        [[buffer(0)]],
                              constant ulong &qElemOffset [[buffer(1)]],
                              device const half *scales   [[buffer(2)]],
                              constant ulong &scalesElemOffset [[buffer(3)]],
                              device const half *biases   [[buffer(4)]],
                              constant ulong &biasesElemOffset [[buffer(5)]],
                              device const half *x        [[buffer(6)]],
                              constant uint &outDim       [[buffer(7)]],
                              constant uint &inDim        [[buffer(8)]],
                              device float *out           [[buffer(9)]],
                              uint gid [[thread_position_in_grid]]) {
        if (gid >= outDim) return;
        uint groupsPerRow = inDim / 64;
        ulong qBase = qElemOffset + ulong(gid) * (inDim / 8);
        ulong groupBase = ulong(gid) * groupsPerRow;
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
        out[gid] = acc;
    }
    """

    private let tilePipeline: MTLComputePipelineState
    private let gatherPipeline: MTLComputePipelineState
    private let matvecF16Pipeline: MTLComputePipelineState
    private let matvecF32Pipeline: MTLComputePipelineState

    /// When set, every encoded dispatch increments it at the dispatchThreads
    /// call site (P2-5 instrumentation; `GPUModel` attaches its per-step
    /// counter in P3-5). nil (the default) counts nothing.
    var dispatchCounter: DispatchCounter?

    public init(context: MetalContext) throws {
        let library = try context.makeLibrary(source: Self.source)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            try context.makeComputePipeline(library: library, function: name)
        }
        tilePipeline = try pipeline("dequant_tile_f32")
        gatherPipeline = try pipeline("embedding_gather_q4_f16")
        matvecF16Pipeline = try pipeline("matvec_q4_f16")
        matvecF32Pipeline = try pipeline("matvec_q4_f32")
    }

    // MARK: - Encode methods (one per kernel)

    /// Dequants the whole packed [outDim, inDim] matrix to row-major fp32 in
    /// `output` — the exact-match layer of the oracle (test support only;
    /// never part of the decode path, so hard rule 1 stands: the dequanted
    /// values land in a test dump buffer, not a weight buffer kernels consume).
    public func encodeDequantTile(
        into encoder: MTLComputeCommandEncoder,
        q: MTLBuffer, qByteOffset: Int,
        scales: MTLBuffer, scalesByteOffset: Int,
        biases: MTLBuffer, biasesByteOffset: Int,
        outDim: Int, inDim: Int, output: MTLBuffer
    ) throws {
        let offsets = try tripletElementOffsets(
            q: q, qByteOffset: qByteOffset,
            scales: scales, scalesByteOffset: scalesByteOffset,
            biases: biases, biasesByteOffset: biasesByteOffset,
            outDim: outDim, inDim: inDim)
        try requireCapacity(output, bytes: outDim * inDim * 4, name: "output")

        encoder.setComputePipelineState(tilePipeline)
        setTriplet(encoder, q: q, scales: scales, biases: biases, offsets: offsets)
        setScalar(encoder, UInt32(outDim), index: 6)
        setScalar(encoder, UInt32(inDim), index: 7)
        encoder.setBuffer(output, offset: 0, index: 8)
        dispatch2D(encoder, width: inDim, height: outDim)
    }

    /// Gathers embedding row `tokenId` from the packed [vocabSize, hiddenSize]
    /// triplet, dequants in registers, and stores fp16 into `output`
    /// ([hiddenSize]).
    public func encodeEmbeddingGather(
        into encoder: MTLComputeCommandEncoder,
        q: MTLBuffer, qByteOffset: Int,
        scales: MTLBuffer, scalesByteOffset: Int,
        biases: MTLBuffer, biasesByteOffset: Int,
        vocabSize: Int, hiddenSize: Int, tokenId: Int, output: MTLBuffer
    ) throws {
        let offsets = try tripletElementOffsets(
            q: q, qByteOffset: qByteOffset,
            scales: scales, scalesByteOffset: scalesByteOffset,
            biases: biases, biasesByteOffset: biasesByteOffset,
            outDim: vocabSize, inDim: hiddenSize,
            outDimName: "vocabSize", inDimName: "hiddenSize")
        guard tokenId >= 0, tokenId < vocabSize else {
            throw QuantKernelError.tokenIdOutOfRange(id: tokenId, vocabSize: vocabSize)
        }
        try requireCapacity(output, bytes: hiddenSize * 2, name: "output")

        encoder.setComputePipelineState(gatherPipeline)
        setTriplet(encoder, q: q, scales: scales, biases: biases, offsets: offsets)
        setScalar(encoder, UInt32(tokenId), index: 6)
        setScalar(encoder, UInt32(hiddenSize), index: 7)
        encoder.setBuffer(output, offset: 0, index: 8)
        dispatch1D(encoder, pipeline: gatherPipeline, count: hiddenSize)
    }

    /// y = W·x with W the packed [outDim, inDim] triplet, x fp16 [inDim];
    /// output fp16 (projections) or fp32 (lm_head logits).
    public func encodeMatvec(
        into encoder: MTLComputeCommandEncoder,
        q: MTLBuffer, qByteOffset: Int,
        scales: MTLBuffer, scalesByteOffset: Int,
        biases: MTLBuffer, biasesByteOffset: Int,
        input: MTLBuffer, outDim: Int, inDim: Int,
        output: MTLBuffer, fp32Output: Bool = false
    ) throws {
        let offsets = try tripletElementOffsets(
            q: q, qByteOffset: qByteOffset,
            scales: scales, scalesByteOffset: scalesByteOffset,
            biases: biases, biasesByteOffset: biasesByteOffset,
            outDim: outDim, inDim: inDim)
        try requireCapacity(input, bytes: inDim * 2, name: "input")
        try requireCapacity(
            output, bytes: outDim * (fp32Output ? 4 : 2), name: "output")

        let pipeline = fp32Output ? matvecF32Pipeline : matvecF16Pipeline
        encoder.setComputePipelineState(pipeline)
        setTriplet(encoder, q: q, scales: scales, biases: biases, offsets: offsets)
        encoder.setBuffer(input, offset: 0, index: 6)
        setScalar(encoder, UInt32(outDim), index: 7)
        setScalar(encoder, UInt32(inDim), index: 8)
        encoder.setBuffer(output, offset: 0, index: 9)
        dispatch1D(encoder, pipeline: pipeline, count: outDim)
    }

    // MARK: - Validation + dispatch helpers

    private struct TripletElementOffsets {
        let q: Int
        let scales: Int
        let biases: Int
    }

    /// Validates dims, alignment, and capacity for one packed triplet and
    /// converts byte offsets to element offsets (u32 for `.q`, fp16 for
    /// scales/biases). Typed u32 loads need 4-byte alignment — structurally
    /// true for a valid packed file (PackedCheckpoint validates it at load);
    /// asserted here too, never assumed.
    private func tripletElementOffsets(
        q: MTLBuffer, qByteOffset: Int,
        scales: MTLBuffer, scalesByteOffset: Int,
        biases: MTLBuffer, biasesByteOffset: Int,
        outDim: Int, inDim: Int,
        outDimName: String = "outDim", inDimName: String = "inDim"
    ) throws -> TripletElementOffsets {
        guard outDim > 0 else {
            throw QuantKernelError.nonPositiveDimension(name: outDimName, value: outDim)
        }
        guard inDim > 0 else {
            throw QuantKernelError.nonPositiveDimension(name: inDimName, value: inDim)
        }
        guard inDim % Q4G64.groupSize == 0 else {
            throw QuantKernelError.inDimNotMultipleOfGroup(inDim: inDim)
        }
        guard qByteOffset >= 0, qByteOffset % 4 == 0 else {
            throw QuantKernelError.misalignedOffset(
                buffer: "q", byteOffset: qByteOffset, alignment: 4)
        }
        guard scalesByteOffset >= 0, scalesByteOffset % 2 == 0 else {
            throw QuantKernelError.misalignedOffset(
                buffer: "scales", byteOffset: scalesByteOffset, alignment: 2)
        }
        guard biasesByteOffset >= 0, biasesByteOffset % 2 == 0 else {
            throw QuantKernelError.misalignedOffset(
                buffer: "biases", byteOffset: biasesByteOffset, alignment: 2)
        }
        let groupBytes = outDim * (inDim / Q4G64.groupSize) * 2
        try requireCapacity(
            q, bytes: qByteOffset + outDim * (inDim / Q4G64.codesPerWord) * 4,
            name: "q")
        try requireCapacity(scales, bytes: scalesByteOffset + groupBytes, name: "scales")
        try requireCapacity(biases, bytes: biasesByteOffset + groupBytes, name: "biases")
        return TripletElementOffsets(
            q: qByteOffset / 4, scales: scalesByteOffset / 2, biases: biasesByteOffset / 2)
    }

    private func requireCapacity(
        _ buffer: MTLBuffer, bytes: Int, name: String
    ) throws {
        guard buffer.length >= bytes else {
            throw QuantKernelError.bufferTooSmall(
                buffer: name, requiredBytes: bytes, actualBytes: buffer.length)
        }
    }

    /// Binds the triplet at indices 0-5 (buffer/offset pairs), the layout all
    /// four kernels share.
    private func setTriplet(
        _ encoder: MTLComputeCommandEncoder,
        q: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer,
        offsets: TripletElementOffsets
    ) {
        encoder.setBuffer(q, offset: 0, index: 0)
        setScalar(encoder, UInt64(offsets.q), index: 1)
        encoder.setBuffer(scales, offset: 0, index: 2)
        setScalar(encoder, UInt64(offsets.scales), index: 3)
        encoder.setBuffer(biases, offset: 0, index: 4)
        setScalar(encoder, UInt64(offsets.biases), index: 5)
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
        _ encoder: MTLComputeCommandEncoder, width: Int, height: Int
    ) {
        dispatchCounter?.increment()
        // 16×16 threadgroups (P0B matmul pattern); dispatchThreads handles the
        // ragged edge (kernel bounds checks are belt-and-braces).
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }
}
