import Foundation
import Metal

/// P2-4 (docs/phases/phase-2.md D5): the full GPU decode pipeline — the
/// P2-1/P2-2/P2-3 pieces (whole-checkpoint residency buffer, naive decode
/// kernels, preallocated KV cache + naive attention) wired into a per-token
/// forward pass. ONE command buffer per decoded token: all dispatches
/// (21/layer × 28 + head/tail) encode into a single `timedDispatch`, so dual
/// timing (hard rule 7) rides along on every step and wall−GPU is the
/// dispatch-overhead metric Phase 4 consumes.
///
/// Precision per spec D2: fp16 activations between kernels, fp32 accumulation
/// inside, fp32 scores/softmax and logits; the RoPE kernel consumes the CPU
/// `RoPE`'s fp32 tables, and argmax stays CPU-side in the shared `DecodeLoop`
/// — GPU and CPU decode share one tie-break.
///
/// Weights come in exactly two formats (`WeightsFormat`), both consumed in
/// registers (hard rule 1, nothing materialized):
/// - **bf16** (Phase 2): raw checkpoint bits upcast via bit-shift in the
///   consuming kernel — bit-identical to the CPU reference's upcast.
/// - **q4g64** (P3-5, phase-3.md D5): packed triplets dequanted in registers
///   by the P3-4 fused kernels — dequant values bit-identical to the
///   CPU-quant reference's by the single-rounding argument (gates entry).
///   Norm vectors pass through as bf16 and ride the Phase 2 RMSNorm kernel;
///   attention/RoPE/SwiGLU/residual kernels are untouched (spec scope).
///
/// Exactly one family, like `QwenModel`: the loader refuses configs that are
/// not Qwen3-shaped (QK-norm, no attention biases), and each format's loader
/// refuses the other format's file with a clear error.
public final class GPUModel {
    /// Where a weight matrix's bytes live inside `weights.buffer`: a bf16
    /// tensor's byte offset (Phase 2 kernels) or the q4g64 triplet's three
    /// byte offsets (P3-4 kernels — the whole-checkpoint buffer bound three
    /// times, as QuantKernels anticipates).
    private enum MatrixRef {
        case bf16(byteOffset: Int)
        case q4(qByteOffset: Int, scalesByteOffset: Int, biasesByteOffset: Int)
    }

    /// Byte offsets / matrix refs of one decoder layer's tensors inside
    /// `weights.buffer`. Norm vectors are bf16 in BOTH formats (schema D1
    /// pass-through), so they stay plain byte offsets.
    private struct LayerRefs {
        let inputNorm: Int
        let qProj: MatrixRef
        let kProj: MatrixRef
        let vProj: MatrixRef
        let oProj: MatrixRef
        let qNorm: Int
        let kNorm: Int
        let postAttentionNorm: Int
        let gateProj: MatrixRef
        let upProj: MatrixRef
        let downProj: MatrixRef
    }

    public let config: ModelConfig
    public let maxContext: Int
    public let weights: GPUWeights
    public let kvCache: KVCache
    /// Which weight encoding this pipeline consumes (P3-5) — benchmark rows
    /// and reports record it alongside residency.
    public let weightsFormat: WeightsFormat

    /// Dual timing of the most recent `step` (hard rule 7). Aggregation into
    /// medians/rates is `DecodeTimingCollector`'s job (P2-5).
    public private(set) var lastStepTiming: DispatchTiming?

    /// Compute dispatches encoded by the most recent `step`, measured at the
    /// dispatchThreads call sites (P2-5, spec D5): 591 at the pinned dims
    /// with logits (21/layer × 28 + embedding + final norm + lm_head),
    /// 589 without the logits tail. The packed path replaces kernels 1:1,
    /// so its measured count is reported by the same counter.
    public private(set) var lastStepDispatchCount: Int?

    /// Tokens whose KV entries currently occupy cache positions
    /// `0..<cachedTokens.count`, in order. `lastPositionLogits` extends this
    /// prefix incrementally and resets on any mismatch.
    public private(set) var cachedTokens: [Int] = []

    private let context: MetalContext
    private let decodeKernels: DecodeKernels
    private let attentionKernels: AttentionKernels
    /// Present exactly on the packed path (created iff any `.q4` ref exists).
    private let quantKernels: QuantKernels?
    private let dispatchCounter = DispatchCounter()

    private let embeddingRef: MatrixRef
    private let finalNormOffset: Int
    private let lmHeadRef: MatrixRef
    private let layerRefs: [LayerRefs]

    // Scratch buffers, allocated once at init (sizes are config-fixed).
    // The residual stream ping-pongs hiddenA → hiddenB → hiddenA per layer
    // (residual_add is out-of-place), ending each layer back in hiddenA.
    private let hiddenA: MTLBuffer      // fp16 [hidden]
    private let hiddenB: MTLBuffer      // fp16 [hidden]
    private let normed: MTLBuffer       // fp16 [hidden] — norm outputs
    private let qRaw: MTLBuffer         // fp16 [numHeads·headDim]
    private let qVec: MTLBuffer         // fp16 [numHeads·headDim] — post QK-norm
    private let kRaw: MTLBuffer         // fp16 [kvHeads·headDim]
    private let kVec: MTLBuffer         // fp16 [kvHeads·headDim] — post QK-norm
    private let vVec: MTLBuffer         // fp16 [kvHeads·headDim]
    private let attnOut: MTLBuffer      // fp16 [numHeads·headDim]
    private let projOut: MTLBuffer      // fp16 [hidden] — o_proj / down_proj out
    private let gateBuf: MTLBuffer      // fp16 [intermediate]
    private let upBuf: MTLBuffer        // fp16 [intermediate]
    private let actBuf: MTLBuffer      // fp16 [intermediate]
    private let scores: MTLBuffer       // fp32 [numHeads·maxContext]
    private let probs: MTLBuffer        // fp32 [numHeads·maxContext]
    private let logitsBuf: MTLBuffer    // fp32 [vocab]
    private let cosTable: MTLBuffer     // fp32 [maxContext·headDim/2]
    private let sinTable: MTLBuffer     // fp32 [maxContext·headDim/2]

    /// The Phase 2 bf16 path. Rejects a q4g64 packed file with a clear error
    /// (phase-3.md edge case 9): the packed loader is `init(packed:)`.
    ///
    /// - Parameter maxContext: sizes the preallocated KV cache (hard rule 4)
    ///   and the RoPE tables. The CLI passes the Phase 2 pinned 4096; tests
    ///   pass something small.
    public convenience init(
        checkpoint: SafetensorsFile, config: ModelConfig, context: MetalContext,
        residency: WeightsResidency = .mmap, maxContext: Int
    ) throws {
        guard checkpoint.metadata[Q4G64.formatMetadataKey] != Q4G64.formatTag else {
            throw ModelError.badInput(detail:
                "checkpoint is a q4g64 packed file — load it through "
                + "PackedCheckpoint + GPUModel(packed:), not the bf16 "
                + "checkpoint initializer")
        }
        try self.init(
            file: checkpoint, packed: nil, config: config, context: context,
            residency: residency, maxContext: maxContext)
    }

    /// The P3-5 packed path: identical pipeline, q4g64 matrices consumed by
    /// the P3-4 fused kernels (register dequant, hard rule 1). The packed
    /// file's validation (format tag, provenance, triplet consistency, `.q`
    /// alignment) already happened in `PackedCheckpoint`.
    public convenience init(
        packed: PackedCheckpoint, config: ModelConfig, context: MetalContext,
        residency: WeightsResidency = .mmap, maxContext: Int
    ) throws {
        try self.init(
            file: packed.file, packed: packed, config: config, context: context,
            residency: residency, maxContext: maxContext)
    }

    private init(
        file: SafetensorsFile, packed: PackedCheckpoint?, config: ModelConfig,
        context: MetalContext, residency: WeightsResidency, maxContext: Int
    ) throws {
        guard config.usesQKNorm, !config.attentionBias else {
            throw ModelError.unsupportedFamily(
                detail: "model_type '\(config.modelType)' with attentionBias="
                    + "\(config.attentionBias), usesQKNorm=\(config.usesQKNorm); "
                    + "this engine implements exactly the pinned Qwen3 family "
                    + "(QK-norm, no QKV biases) — see PLAN.md non-goals and the "
                    + "DECISIONS.md PIN-1 entry")
        }
        guard maxContext > 0 else {
            throw ModelError.badInput(detail: "maxContext \(maxContext) must be positive")
        }
        self.config = config
        self.maxContext = maxContext
        self.context = context
        self.weightsFormat = packed == nil ? .bf16 : .q4g64

        let weights = try GPUWeights(file: file, context: context, residency: residency)
        self.weights = weights

        let hidden = config.hiddenSize
        let headDim = config.headDim
        let numHeads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let intermediate = config.intermediateSize

        // Every tensor's byte offset is resolved (and shape/dtype-validated)
        // up front: a wrong checkpoint fails at load, never mid-decode.
        // Norm vectors are raw bf16 in both formats (schema D1 pass-through).
        func offset(_ name: String, shape: [Int]) throws -> Int {
            let info = try weights.info(for: name)
            guard info.shape == shape else {
                throw ModelError.badWeightShape(
                    tensor: name, expected: shape, actual: info.shape)
            }
            guard info.dtype == .bfloat16 else {
                throw ModelError.badWeightDtype(
                    tensor: name, expected: TensorDType.bfloat16.rawValue,
                    actual: info.dtype.rawValue)
            }
            return try weights.byteOffset(for: name)
        }
        func matrix(_ name: String, shape: [Int]) throws -> MatrixRef {
            guard let packed else {
                return .bf16(byteOffset: try offset(name, shape: shape))
            }
            let dims = try packed.dims(for: name)
            guard [dims.outDim, dims.inDim] == shape else {
                throw ModelError.badWeightShape(
                    tensor: name, expected: shape,
                    actual: [dims.outDim, dims.inDim])
            }
            return .q4(
                qByteOffset: try weights.byteOffset(for: name + Q4G64.qSuffix),
                scalesByteOffset: try weights.byteOffset(for: name + Q4G64.scalesSuffix),
                biasesByteOffset: try weights.byteOffset(for: name + Q4G64.biasesSuffix))
        }

        embeddingRef = try matrix(
            "model.embed_tokens.weight", shape: [config.vocabSize, hidden])
        finalNormOffset = try offset("model.norm.weight", shape: [hidden])
        // Tied embeddings reuse the table as [out, in] directly (P2-2: no
        // transpose is ever materialized; the packed artifact stores the tied
        // triplet once — P3-1). An untied config resolves lm_head.weight in
        // its own format and fails loudly when absent.
        lmHeadRef = config.tieWordEmbeddings
            ? embeddingRef
            : try matrix("lm_head.weight", shape: [config.vocabSize, hidden])

        var layers: [LayerRefs] = []
        layers.reserveCapacity(config.numHiddenLayers)
        for layer in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(layer)."
            layers.append(LayerRefs(
                inputNorm: try offset(prefix + "input_layernorm.weight", shape: [hidden]),
                qProj: try matrix(
                    prefix + "self_attn.q_proj.weight", shape: [numHeads * headDim, hidden]),
                kProj: try matrix(
                    prefix + "self_attn.k_proj.weight", shape: [kvHeads * headDim, hidden]),
                vProj: try matrix(
                    prefix + "self_attn.v_proj.weight", shape: [kvHeads * headDim, hidden]),
                oProj: try matrix(
                    prefix + "self_attn.o_proj.weight", shape: [hidden, numHeads * headDim]),
                qNorm: try offset(prefix + "self_attn.q_norm.weight", shape: [headDim]),
                kNorm: try offset(prefix + "self_attn.k_norm.weight", shape: [headDim]),
                postAttentionNorm: try offset(
                    prefix + "post_attention_layernorm.weight", shape: [hidden]),
                gateProj: try matrix(
                    prefix + "mlp.gate_proj.weight", shape: [intermediate, hidden]),
                upProj: try matrix(
                    prefix + "mlp.up_proj.weight", shape: [intermediate, hidden]),
                downProj: try matrix(
                    prefix + "mlp.down_proj.weight", shape: [hidden, intermediate])))
        }
        layerRefs = layers

        decodeKernels = try DecodeKernels(context: context)
        attentionKernels = try AttentionKernels(context: context)
        quantKernels = packed == nil ? nil : try QuantKernels(context: context)
        decodeKernels.dispatchCounter = dispatchCounter
        attentionKernels.dispatchCounter = dispatchCounter
        quantKernels?.dispatchCounter = dispatchCounter
        kvCache = try KVCache(
            device: context.device, layers: config.numHiddenLayers,
            kvHeads: kvHeads, maxContext: maxContext, headDim: headDim)

        func makeBuffer(bytes: Int) throws -> MTLBuffer {
            guard let buffer = context.device.makeBuffer(
                length: bytes, options: .storageModeShared) else {
                throw MetalHarnessError.bufferAllocationFailed(length: bytes)
            }
            return buffer
        }
        hiddenA = try makeBuffer(bytes: hidden * 2)
        hiddenB = try makeBuffer(bytes: hidden * 2)
        normed = try makeBuffer(bytes: hidden * 2)
        qRaw = try makeBuffer(bytes: numHeads * headDim * 2)
        qVec = try makeBuffer(bytes: numHeads * headDim * 2)
        kRaw = try makeBuffer(bytes: kvHeads * headDim * 2)
        kVec = try makeBuffer(bytes: kvHeads * headDim * 2)
        vVec = try makeBuffer(bytes: kvHeads * headDim * 2)
        attnOut = try makeBuffer(bytes: numHeads * headDim * 2)
        projOut = try makeBuffer(bytes: hidden * 2)
        gateBuf = try makeBuffer(bytes: intermediate * 2)
        upBuf = try makeBuffer(bytes: intermediate * 2)
        actBuf = try makeBuffer(bytes: intermediate * 2)
        scores = try makeBuffer(bytes: numHeads * maxContext * 4)
        probs = try makeBuffer(bytes: numHeads * maxContext * 4)
        logitsBuf = try makeBuffer(bytes: config.vocabSize * 4)

        // The GPU RoPE kernel consumes the CPU reference's fp32 tables —
        // bit-identical angles by construction (P2-2).
        let rope = try RoPE(
            headDim: headDim, theta: config.ropeTheta, positions: maxContext)
        cosTable = try makeBuffer(bytes: rope.cosValues.count * 4)
        sinTable = try makeBuffer(bytes: rope.sinValues.count * 4)
        rope.cosValues.withUnsafeBytes {
            cosTable.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        rope.sinValues.withUnsafeBytes {
            sinTable.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
    }

    /// Forgets the cached prefix. Cache CONTENTS are not zeroed — positions
    /// at or beyond the append point are never read (attention spans
    /// `0...position` only), so stale slots are unreachable by construction.
    public func reset() {
        cachedTokens = []
    }

    /// Runs one token through the full stack at the next cache position,
    /// appending its K/V. Returns fp32 full-vocab logits when `computeLogits`
    /// (final norm + lm_head encode only then — spec D6 computes logits at
    /// the last prompt position, not per prompt token), else nil.
    @discardableResult
    public func step(token: Int, computeLogits: Bool) throws -> [Float]? {
        let position = cachedTokens.count
        guard position < maxContext else {
            throw KVCacheError.contextFull(position: position, maxContext: maxContext)
        }
        guard token >= 0, token < config.vocabSize else {
            throw ModelError.tokenIdOutOfRange(id: token, vocabSize: config.vocabSize)
        }
        dispatchCounter.reset()
        lastStepTiming = try context.timedDispatch { encoder in
            try encodeForward(
                into: encoder, token: token, position: position,
                computeLogits: computeLogits)
        }
        lastStepDispatchCount = dispatchCounter.count
        cachedTokens.append(token)
        guard computeLogits else { return nil }
        let vocab = config.vocabSize
        return logitsBuf.contents().withMemoryRebound(
            to: Float.self, capacity: vocab
        ) { Array(UnsafeBufferPointer(start: $0, count: vocab)) }
    }

    // MARK: - Fixture hook points (Tier-E tests; @testable access)

    /// fp32 view of the fp16 residual stream after the most recent `step` —
    /// the last_layer_output hook point (post layer stack, pre final norm).
    func lastLayerOutput() -> [Float] {
        readF16(hiddenA, count: config.hiddenSize)
    }

    /// fp32 view of the final-norm output — only meaningful when the most
    /// recent `step` had `computeLogits: true` (the buffer otherwise holds
    /// the last layer's post-attention norm).
    func finalNormOutput() -> [Float] {
        readF16(normed, count: config.hiddenSize)
    }

    private func readF16(_ buffer: MTLBuffer, count: Int) -> [Float] {
        buffer.contents().withMemoryRebound(to: Float16.self, capacity: count) {
            let values = UnsafeBufferPointer(start: $0, count: count)
            return values.map(Float.init)
        }
    }

    // MARK: - Format-dispatched encode helpers (P3-5)

    /// One matvec against a weight matrix in whichever format it lives —
    /// bf16 (Phase 2 kernel, register upcast) or q4g64 (P3-4 fused kernel,
    /// register dequant). `quantKernels` exists whenever a `.q4` ref exists:
    /// both are created exactly on the packed path.
    private func encodeMatvec(
        _ ref: MatrixRef, into encoder: MTLComputeCommandEncoder,
        input: MTLBuffer, outDim: Int, inDim: Int, output: MTLBuffer,
        fp32Output: Bool = false
    ) throws {
        switch ref {
        case .bf16(let byteOffset):
            try decodeKernels.encodeMatvec(
                into: encoder, weights: weights.buffer, weightByteOffset: byteOffset,
                input: input, outDim: outDim, inDim: inDim, output: output,
                fp32Output: fp32Output)
        case .q4(let q, let scales, let biases):
            try quantKernels!.encodeMatvec(
                into: encoder, q: weights.buffer, qByteOffset: q,
                scales: weights.buffer, scalesByteOffset: scales,
                biases: weights.buffer, biasesByteOffset: biases,
                input: input, outDim: outDim, inDim: inDim, output: output,
                fp32Output: fp32Output)
        }
    }

    private func encodeEmbedding(
        into encoder: MTLComputeCommandEncoder, token: Int, output: MTLBuffer
    ) throws {
        switch embeddingRef {
        case .bf16(let byteOffset):
            try decodeKernels.encodeEmbeddingLookup(
                into: encoder, table: weights.buffer, tableByteOffset: byteOffset,
                vocabSize: config.vocabSize, hiddenSize: config.hiddenSize,
                tokenId: token, output: output)
        case .q4(let q, let scales, let biases):
            try quantKernels!.encodeEmbeddingGather(
                into: encoder, q: weights.buffer, qByteOffset: q,
                scales: weights.buffer, scalesByteOffset: scales,
                biases: weights.buffer, biasesByteOffset: biases,
                vocabSize: config.vocabSize, hiddenSize: config.hiddenSize,
                tokenId: token, output: output)
        }
    }

    // MARK: - The forward encoding (one command buffer per token, spec D5)

    private func encodeForward(
        into encoder: MTLComputeCommandEncoder, token: Int, position: Int,
        computeLogits: Bool
    ) throws {
        let hidden = config.hiddenSize
        let headDim = config.headDim
        let numHeads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let intermediate = config.intermediateSize
        let eps = Float(config.rmsNormEps)

        try encodeEmbedding(into: encoder, token: token, output: hiddenA)

        for (layer, refs) in layerRefs.enumerated() {
            // h = hiddenA on entry. Attention half: hiddenB = h + attn(norm(h)).
            try decodeKernels.encodeRMSNorm(
                into: encoder, input: hiddenA, weight: weights.buffer,
                weightByteOffset: refs.inputNorm, rows: 1, dim: hidden,
                eps: eps, output: normed)
            try encodeMatvec(
                refs.qProj, into: encoder, input: normed,
                outDim: numHeads * headDim, inDim: hidden, output: qRaw)
            try encodeMatvec(
                refs.kProj, into: encoder, input: normed,
                outDim: kvHeads * headDim, inDim: hidden, output: kRaw)
            try encodeMatvec(
                refs.vProj, into: encoder, input: normed,
                outDim: kvHeads * headDim, inDim: hidden, output: vVec)
            // Family order (PIN-1): per-head Q/K RMSNorm, THEN RoPE.
            try decodeKernels.encodeRMSNorm(
                into: encoder, input: qRaw, weight: weights.buffer,
                weightByteOffset: refs.qNorm, rows: numHeads, dim: headDim,
                eps: eps, output: qVec)
            try decodeKernels.encodeRMSNorm(
                into: encoder, input: kRaw, weight: weights.buffer,
                weightByteOffset: refs.kNorm, rows: kvHeads, dim: headDim,
                eps: eps, output: kVec)
            try decodeKernels.encodeRoPE(
                into: encoder, vector: qVec, cosTable: cosTable, sinTable: sinTable,
                position: position, positions: maxContext, heads: numHeads,
                headDim: headDim)
            try decodeKernels.encodeRoPE(
                into: encoder, vector: kVec, cosTable: cosTable, sinTable: sinTable,
                position: position, positions: maxContext, heads: kvHeads,
                headDim: headDim)
            try attentionKernels.encodeKVAppend(
                into: encoder, cache: kvCache, layer: layer, component: .key,
                position: position, vector: kVec)
            try attentionKernels.encodeKVAppend(
                into: encoder, cache: kvCache, layer: layer, component: .value,
                position: position, vector: vVec)
            try attentionKernels.encodeAttentionScores(
                into: encoder, cache: kvCache, layer: layer, position: position,
                query: qVec, numHeads: numHeads, scores: scores)
            try attentionKernels.encodeSoftmaxRows(
                into: encoder, input: scores, rows: numHeads, count: position + 1,
                rowStride: maxContext, output: probs)
            try attentionKernels.encodeAttentionPV(
                into: encoder, cache: kvCache, layer: layer, position: position,
                probs: probs, numHeads: numHeads, output: attnOut)
            try encodeMatvec(
                refs.oProj, into: encoder, input: attnOut,
                outDim: hidden, inDim: numHeads * headDim, output: projOut)
            try decodeKernels.encodeResidualAdd(
                into: encoder, a: hiddenA, b: projOut, count: hidden, output: hiddenB)

            // MLP half: hiddenA = hiddenB + mlp(norm(hiddenB)).
            try decodeKernels.encodeRMSNorm(
                into: encoder, input: hiddenB, weight: weights.buffer,
                weightByteOffset: refs.postAttentionNorm, rows: 1, dim: hidden,
                eps: eps, output: normed)
            try encodeMatvec(
                refs.gateProj, into: encoder, input: normed,
                outDim: intermediate, inDim: hidden, output: gateBuf)
            try encodeMatvec(
                refs.upProj, into: encoder, input: normed,
                outDim: intermediate, inDim: hidden, output: upBuf)
            try decodeKernels.encodeSwiGLU(
                into: encoder, gate: gateBuf, up: upBuf, count: intermediate,
                output: actBuf)
            try encodeMatvec(
                refs.downProj, into: encoder, input: actBuf,
                outDim: hidden, inDim: intermediate, output: projOut)
            try decodeKernels.encodeResidualAdd(
                into: encoder, a: hiddenB, b: projOut, count: hidden, output: hiddenA)
        }

        guard computeLogits else { return }
        try decodeKernels.encodeRMSNorm(
            into: encoder, input: hiddenA, weight: weights.buffer,
            weightByteOffset: finalNormOffset, rows: 1, dim: hidden,
            eps: eps, output: normed)
        try encodeMatvec(
            lmHeadRef, into: encoder, input: normed,
            outDim: config.vocabSize, inDim: hidden, output: logitsBuf,
            fp32Output: true)
    }
}

// MARK: - Decode-loop conformance (the shared DecodeLoop drives both backends)

extension GPUModel: NextTokenLogitsSource {
    public var vocabSize: Int { config.vocabSize }

    /// Incremental form of the CPU reference's full re-forward: when `ids`
    /// strictly extends the cached prefix, only the new suffix runs (KV for
    /// the prefix is already in the cache); any other shape resets the cache
    /// and replays from scratch. Logits are computed at the last position
    /// only (spec D6).
    public func lastPositionLogits(ids: [Int]) throws -> [Float] {
        guard !ids.isEmpty else {
            throw ModelError.badInput(detail: "lastPositionLogits of an empty sequence")
        }
        if !(ids.count > cachedTokens.count && ids.starts(with: cachedTokens)) {
            reset()
        }
        var logits: [Float]?
        for i in cachedTokens.count..<ids.count {
            logits = try step(token: ids[i], computeLogits: i == ids.count - 1)
        }
        // The loop ran at least once (cachedTokens.count < ids.count holds in
        // both branches above) and its last iteration computed logits.
        return logits!
    }
}
