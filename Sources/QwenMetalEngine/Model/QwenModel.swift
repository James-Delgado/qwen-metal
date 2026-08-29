import Foundation

/// Source of row-major fp32 weight tensors for CPU model assembly — the
/// weight-loading front end of PLAN.md invariant 4's oracle chain.
/// `SafetensorsFile` is the Phase 1 bf16 path; `PackedCheckpoint` (Quant)
/// is the Phase 3 CPU-quant reference (fp32 dequant materialization,
/// phase-3.md D3). The forward math never knows which one fed it.
public protocol WeightSource {
    /// fp32 values for tensor `name`, validated against the expected
    /// `shape`. Throws `ModelError.badWeightShape` on mismatch and a loud
    /// not-found error for unknown names.
    func fp32Tensor(_ name: String, shape: [Int]) throws -> [Float]
}

extension SafetensorsFile: WeightSource {
    public func fp32Tensor(_ name: String, shape: [Int]) throws -> [Float] {
        let info = try info(for: name)
        guard info.shape == shape else {
            throw ModelError.badWeightShape(
                tensor: name, expected: shape, actual: info.shape)
        }
        return try fp32Values(for: name)
    }
}

/// The Phase 1 CPU reference model: the pinned Qwen3 stack assembled from a
/// single-file safetensors checkpoint. This is the oracle-validated side of
/// PLAN.md invariant 4's chain — obviously-correct over fast; the only
/// performance concession is the shared BLAS wrapper.
///
/// Exactly one family (PLAN.md non-goals): the loader refuses configs that
/// are not Qwen3-shaped (QK-norm, no attention biases) instead of growing
/// an architecture registry.
public final class QwenModel {
    public let config: ModelConfig
    public let embedding: Embedding
    public let blocks: [TransformerBlock]
    public let finalNorm: RMSNorm
    /// [vocabSize, hiddenSize] — the embedding table itself when the config
    /// ties word embeddings (the pinned checkpoint does).
    public let lmHeadWeight: [Float]

    /// The Phase 1 bf16 path: assemble straight from the consolidated
    /// checkpoint (identical behavior to the pre-P3-2 initializer).
    public convenience init(
        checkpoint: SafetensorsFile, config: ModelConfig,
        maxSequenceLength: Int? = nil
    ) throws {
        try self.init(weights: checkpoint, config: config,
                      maxSequenceLength: maxSequenceLength)
    }

    /// - Parameter maxSequenceLength: sizes the RoPE angle table. Defaults to
    ///   the config's max_position_embeddings; tests pass something small.
    public init(
        weights: WeightSource, config: ModelConfig,
        maxSequenceLength: Int? = nil
    ) throws {
        guard config.usesQKNorm, !config.attentionBias else {
            throw ModelError.unsupportedFamily(
                detail: "model_type '\(config.modelType)' with attentionBias="
                    + "\(config.attentionBias), usesQKNorm=\(config.usesQKNorm); "
                    + "this engine implements exactly the pinned Qwen3 family "
                    + "(QK-norm, no QKV biases) — see PLAN.md non-goals and the "
                    + "DECISIONS.md PIN-1 entry")
        }
        self.config = config

        let hidden = config.hiddenSize
        let headDim = config.headDim
        let numHeads = config.numAttentionHeads
        let numKVHeads = config.numKeyValueHeads
        let eps = Float(config.rmsNormEps)
        let rope = try RoPE(
            headDim: headDim, theta: config.ropeTheta,
            positions: maxSequenceLength ?? config.maxPositionEmbeddings)

        let embeddingWeight = try weights.fp32Tensor(
            "model.embed_tokens.weight", shape: [config.vocabSize, hidden])
        self.embedding = try Embedding(
            weight: embeddingWeight, vocabSize: config.vocabSize, hiddenSize: hidden)

        var blocks: [TransformerBlock] = []
        blocks.reserveCapacity(config.numHiddenLayers)
        for layer in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(layer)."
            let attention = try Attention(
                numHeads: numHeads, numKVHeads: numKVHeads,
                headDim: headDim, hiddenSize: hidden,
                qProjWeight: try weights.fp32Tensor(
                    prefix + "self_attn.q_proj.weight",
                    shape: [numHeads * headDim, hidden]),
                kProjWeight: try weights.fp32Tensor(
                    prefix + "self_attn.k_proj.weight",
                    shape: [numKVHeads * headDim, hidden]),
                vProjWeight: try weights.fp32Tensor(
                    prefix + "self_attn.v_proj.weight",
                    shape: [numKVHeads * headDim, hidden]),
                oProjWeight: try weights.fp32Tensor(
                    prefix + "self_attn.o_proj.weight",
                    shape: [hidden, numHeads * headDim]),
                qNorm: RMSNorm(
                    weight: try weights.fp32Tensor(
                        prefix + "self_attn.q_norm.weight", shape: [headDim]),
                    eps: eps),
                kNorm: RMSNorm(
                    weight: try weights.fp32Tensor(
                        prefix + "self_attn.k_norm.weight", shape: [headDim]),
                    eps: eps),
                rope: rope)
            let mlp = try SwiGLUMLP(
                gateWeight: try weights.fp32Tensor(
                    prefix + "mlp.gate_proj.weight",
                    shape: [config.intermediateSize, hidden]),
                upWeight: try weights.fp32Tensor(
                    prefix + "mlp.up_proj.weight",
                    shape: [config.intermediateSize, hidden]),
                downWeight: try weights.fp32Tensor(
                    prefix + "mlp.down_proj.weight",
                    shape: [hidden, config.intermediateSize]),
                hiddenSize: hidden, intermediateSize: config.intermediateSize)
            blocks.append(TransformerBlock(
                inputNorm: RMSNorm(
                    weight: try weights.fp32Tensor(
                        prefix + "input_layernorm.weight", shape: [hidden]),
                    eps: eps),
                attention: attention,
                postAttentionNorm: RMSNorm(
                    weight: try weights.fp32Tensor(
                        prefix + "post_attention_layernorm.weight",
                        shape: [hidden]),
                    eps: eps),
                mlp: mlp))
        }
        self.blocks = blocks

        self.finalNorm = RMSNorm(
            weight: try weights.fp32Tensor("model.norm.weight", shape: [hidden]),
            eps: eps)
        // Tied embeddings share the table (invariant 2's quantized-lm_head
        // requirement later applies to this one shared matrix). This is also
        // why the packed path needs no lm_head aliasing: the q4g64 artifact
        // stores the tied matrix once (P3-1) and the tied config never asks
        // for lm_head.weight.
        self.lmHeadWeight = config.tieWordEmbeddings
            ? embeddingWeight
            : try weights.fp32Tensor(
                "lm_head.weight", shape: [config.vocabSize, hidden])
    }

    /// Full forward through embedding + all decoder layers (no final norm):
    /// the last_layer_output fixture hook point. Returns [ids.count, hidden].
    public func hiddenStates(ids: [Int]) throws -> [Float] {
        var h = try embedding(ids)
        for block in blocks {
            h = try block(h, seqLen: ids.count)
        }
        return h
    }

    /// Projects final-norm output to full-vocab logits (tied lm_head).
    /// `normedHidden` is [seqLen, hidden]; returns [seqLen, vocabSize].
    public func lmHead(_ normedHidden: [Float], seqLen: Int) throws -> [Float] {
        return try BLAS.sgemm(
            a: normedHidden, b: lmHeadWeight,
            m: seqLen, k: config.hiddenSize, n: config.vocabSize,
            transposeB: true)
    }
}
