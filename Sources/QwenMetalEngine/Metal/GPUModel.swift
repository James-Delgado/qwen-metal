import Foundation
import Metal

/// P2-4 (docs/phases/phase-2.md D5): the full GPU decode pipeline — the
/// P2-1/P2-2/P2-3 pieces (whole-checkpoint residency buffer, naive decode
/// kernels, preallocated KV cache + naive attention) wired into a per-token
/// forward pass. ONE command buffer per decoded token: all ~590 dispatches
/// (21/layer × 28 + head/tail) encode into a single `timedDispatch`, so dual
/// timing (hard rule 7) rides along on every step and wall−GPU is the
/// dispatch-overhead metric Phase 4 consumes.
///
/// Precision per spec D2: fp16 activations between kernels, fp32 accumulation
/// inside, fp32 scores/softmax and logits; weights stay raw bf16 checkpoint
/// bits upcast in registers (D1 — bit-identical to the CPU reference's, so
/// every GPU-vs-CPU gate sees zero weight-rounding divergence). The RoPE
/// kernel consumes the CPU `RoPE`'s fp32 tables, and argmax stays CPU-side in
/// the shared `DecodeLoop` — GPU and CPU decode share one tie-break.
///
/// Exactly one family, like `QwenModel`: the loader refuses configs that are
/// not Qwen3-shaped (QK-norm, no attention biases), and refuses weight
/// tensors that are not bf16 (the kernels' register upcast is bf16-specific).
public final class GPUModel {
    /// Byte offsets of one decoder layer's tensors inside `weights.buffer`.
    private struct LayerOffsets {
        let inputNorm: Int
        let qProj: Int
        let kProj: Int
        let vProj: Int
        let oProj: Int
        let qNorm: Int
        let kNorm: Int
        let postAttentionNorm: Int
        let gateProj: Int
        let upProj: Int
        let downProj: Int
    }

    public let config: ModelConfig
    public let maxContext: Int
    public let weights: GPUWeights
    public let kvCache: KVCache

    /// Dual timing of the most recent `step` (hard rule 7). Aggregation into
    /// medians/rates is `DecodeTimingCollector`'s job (P2-5).
    public private(set) var lastStepTiming: DispatchTiming?

    /// Compute dispatches encoded by the most recent `step`, measured at the
    /// dispatchThreads call sites (P2-5, spec D5): 591 at the pinned dims
    /// with logits (21/layer × 28 + embedding + final norm + lm_head),
    /// 589 without the logits tail.
    public private(set) var lastStepDispatchCount: Int?

    /// Tokens whose KV entries currently occupy cache positions
    /// `0..<cachedTokens.count`, in order. `lastPositionLogits` extends this
    /// prefix incrementally and resets on any mismatch.
    public private(set) var cachedTokens: [Int] = []

    private let context: MetalContext
    private let decodeKernels: DecodeKernels
    private let attentionKernels: AttentionKernels
    private let dispatchCounter = DispatchCounter()

    private let embeddingOffset: Int
    private let finalNormOffset: Int
    private let lmHeadOffset: Int
    private let layerOffsets: [LayerOffsets]

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

    /// - Parameter maxContext: sizes the preallocated KV cache (hard rule 4)
    ///   and the RoPE tables. The CLI passes the Phase 2 pinned 4096; tests
    ///   pass something small.
    public init(
        checkpoint: SafetensorsFile, config: ModelConfig, context: MetalContext,
        residency: WeightsResidency = .mmap, maxContext: Int
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

        let weights = try GPUWeights(file: checkpoint, context: context, residency: residency)
        self.weights = weights

        let hidden = config.hiddenSize
        let headDim = config.headDim
        let numHeads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let intermediate = config.intermediateSize

        // Every tensor's byte offset is resolved (and shape/dtype-validated)
        // up front: a wrong checkpoint fails at load, never mid-decode.
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

        embeddingOffset = try offset(
            "model.embed_tokens.weight", shape: [config.vocabSize, hidden])
        finalNormOffset = try offset("model.norm.weight", shape: [hidden])
        // Tied embeddings reuse the table as [out, in] directly (P2-2: no
        // transpose is ever materialized).
        lmHeadOffset = config.tieWordEmbeddings
            ? embeddingOffset
            : try offset("lm_head.weight", shape: [config.vocabSize, hidden])

        var layers: [LayerOffsets] = []
        layers.reserveCapacity(config.numHiddenLayers)
        for layer in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(layer)."
            layers.append(LayerOffsets(
                inputNorm: try offset(prefix + "input_layernorm.weight", shape: [hidden]),
                qProj: try offset(
                    prefix + "self_attn.q_proj.weight", shape: [numHeads * headDim, hidden]),
                kProj: try offset(
                    prefix + "self_attn.k_proj.weight", shape: [kvHeads * headDim, hidden]),
                vProj: try offset(
                    prefix + "self_attn.v_proj.weight", shape: [kvHeads * headDim, hidden]),
                oProj: try offset(
                    prefix + "self_attn.o_proj.weight", shape: [hidden, numHeads * headDim]),
                qNorm: try offset(prefix + "self_attn.q_norm.weight", shape: [headDim]),
                kNorm: try offset(prefix + "self_attn.k_norm.weight", shape: [headDim]),
                postAttentionNorm: try offset(
                    prefix + "post_attention_layernorm.weight", shape: [hidden]),
                gateProj: try offset(
                    prefix + "mlp.gate_proj.weight", shape: [intermediate, hidden]),
                upProj: try offset(
                    prefix + "mlp.up_proj.weight", shape: [intermediate, hidden]),
                downProj: try offset(
                    prefix + "mlp.down_proj.weight", shape: [hidden, intermediate])))
        }
        layerOffsets = layers

        decodeKernels = try DecodeKernels(context: context)
        attentionKernels = try AttentionKernels(context: context)
        decodeKernels.dispatchCounter = dispatchCounter
        attentionKernels.dispatchCounter = dispatchCounter
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

        try decodeKernels.encodeEmbeddingLookup(
            into: encoder, table: weights.buffer, tableByteOffset: embeddingOffset,
            vocabSize: config.vocabSize, hiddenSize: hidden, tokenId: token,
            output: hiddenA)

        for (layer, offsets) in layerOffsets.enumerated() {
            // h = hiddenA on entry. Attention half: hiddenB = h + attn(norm(h)).
            try decodeKernels.encodeRMSNorm(
                into: encoder, input: hiddenA, weight: weights.buffer,
                weightByteOffset: offsets.inputNorm, rows: 1, dim: hidden,
                eps: eps, output: normed)
            try decodeKernels.encodeMatvec(
                into: encoder, weights: weights.buffer, weightByteOffset: offsets.qProj,
                input: normed, outDim: numHeads * headDim, inDim: hidden, output: qRaw)
            try decodeKernels.encodeMatvec(
                into: encoder, weights: weights.buffer, weightByteOffset: offsets.kProj,
                input: normed, outDim: kvHeads * headDim, inDim: hidden, output: kRaw)
            try decodeKernels.encodeMatvec(
                into: encoder, weights: weights.buffer, weightByteOffset: offsets.vProj,
                input: normed, outDim: kvHeads * headDim, inDim: hidden, output: vVec)
            // Family order (PIN-1): per-head Q/K RMSNorm, THEN RoPE.
            try decodeKernels.encodeRMSNorm(
                into: encoder, input: qRaw, weight: weights.buffer,
                weightByteOffset: offsets.qNorm, rows: numHeads, dim: headDim,
                eps: eps, output: qVec)
            try decodeKernels.encodeRMSNorm(
                into: encoder, input: kRaw, weight: weights.buffer,
                weightByteOffset: offsets.kNorm, rows: kvHeads, dim: headDim,
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
            try decodeKernels.encodeMatvec(
                into: encoder, weights: weights.buffer, weightByteOffset: offsets.oProj,
                input: attnOut, outDim: hidden, inDim: numHeads * headDim,
                output: projOut)
            try decodeKernels.encodeResidualAdd(
                into: encoder, a: hiddenA, b: projOut, count: hidden, output: hiddenB)

            // MLP half: hiddenA = hiddenB + mlp(norm(hiddenB)).
            try decodeKernels.encodeRMSNorm(
                into: encoder, input: hiddenB, weight: weights.buffer,
                weightByteOffset: offsets.postAttentionNorm, rows: 1, dim: hidden,
                eps: eps, output: normed)
            try decodeKernels.encodeMatvec(
                into: encoder, weights: weights.buffer, weightByteOffset: offsets.gateProj,
                input: normed, outDim: intermediate, inDim: hidden, output: gateBuf)
            try decodeKernels.encodeMatvec(
                into: encoder, weights: weights.buffer, weightByteOffset: offsets.upProj,
                input: normed, outDim: intermediate, inDim: hidden, output: upBuf)
            try decodeKernels.encodeSwiGLU(
                into: encoder, gate: gateBuf, up: upBuf, count: intermediate,
                output: actBuf)
            try decodeKernels.encodeMatvec(
                into: encoder, weights: weights.buffer, weightByteOffset: offsets.downProj,
                input: actBuf, outDim: hidden, inDim: intermediate, output: projOut)
            try decodeKernels.encodeResidualAdd(
                into: encoder, a: hiddenB, b: projOut, count: hidden, output: hiddenA)
        }

        guard computeLogits else { return }
        try decodeKernels.encodeRMSNorm(
            into: encoder, input: hiddenA, weight: weights.buffer,
            weightByteOffset: finalNormOffset, rows: 1, dim: hidden,
            eps: eps, output: normed)
        try decodeKernels.encodeMatvec(
            into: encoder, weights: weights.buffer, weightByteOffset: lmHeadOffset,
            input: normed, outDim: config.vocabSize, inDim: hidden,
            output: logitsBuf, fp32Output: true)
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
