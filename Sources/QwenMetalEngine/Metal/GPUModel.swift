import Foundation
import Metal
import QuartzCore

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
/// `RoPE`'s fp32 tables. Argmax originally stayed CPU-side in the shared
/// `DecodeLoop`; since P4-8 (design change approved 2026-09-12) the
/// free-running decode path selects the token on-GPU under an exact-equality
/// contract with `Argmax.firstIndex` — same tie-break, pinned by test — while
/// `step(computeLogits:)` keeps returning full logits for the oracle suites.
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
    /// P4-2/P4-3/P4-6/P4-7 (phase-4.md D2-D4 + the 2026-09-12 addendum):
    /// which kernel structure the packed pipeline runs. `.naive` is the
    /// Phase 2/3 21-dispatch layer (unfused scores → softmax → PV attention,
    /// standalone elementwise kernels); `.fused` is the Phase 4 7-dispatch
    /// layer (two-pass split-K online-softmax SDPA plus the folding set:
    /// norm-folded QKV concat, qk-norm/RoPE/append cluster, norm-folded
    /// gate+up+SwiGLU, residual-folded matvecs). Selectable ONLY on
    /// the packed (q4g64) pipeline — the bf16 backend runs the naive
    /// structure permanently (it is the Phase 2 correctness artifact).
    /// FUSED is the packed default since P4-4 (spec D4); naive stays
    /// selectable for the P4-5 in-session before/after row. Both paths gate
    /// against the same CPU-quant oracle (D5).
    public enum KernelPath: String, CaseIterable, Sendable {
        case naive
        case fused
    }

    /// P5-3 (phase-5.md D2/D5): how a multi-token prompt suffix is
    /// processed. `.sequential` is the Phase 2–4 per-token loop;
    /// `.tiled` is the chunked batched prefill (GEMM projections over C
    /// positions per chunk, batched norm/RoPE/append, per-position split-K
    /// SDPA, last-position-only lm_head). Tiled exists exactly on the
    /// packed + fused pipeline (the bf16 backend keeps sequential prefill
    /// permanently). Decode — every single-token step after the prompt —
    /// is the unchanged Phase 4 path on BOTH settings. TILED is the packed
    /// (fused) default since P5-4 (spec D5); sequential stays selectable
    /// (CLI `--prefill`, app "Prefill" toggle) for the P5-5 in-session
    /// before/after row, and is the only option on the naive kernel arm.
    public enum PrefillPath: String, CaseIterable, Sendable {
        case sequential
        case tiled
    }

    /// The prefill path an UNSPECIFIED request resolves to on the packed
    /// pipeline (P5-4, spec D5): tiled on the fused kernel path, sequential
    /// on the naive arm (which supports sequential prefill only). Exposed
    /// so the CLI/app can label what a default load will run.
    public static func defaultPrefillPath(for kernelPath: KernelPath) -> PrefillPath {
        kernelPath == .fused ? .tiled : .sequential
    }

    /// Default prefill chunk size C for the tiled path — a REPORTED
    /// parameter, not a pin (spec D2: chosen by measurement, recorded per
    /// row; the selection sweep is in the DECISIONS.md P5-3 entry). An
    /// unspecified C resolves to min(this, maxContext) — a chunk can never
    /// hold more positions than the context — and the resolved value is
    /// what `prefillChunkSize` reports.
    public static let defaultPrefillChunkSize = 512
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
    /// Which attention kernel structure this pipeline runs (P4-2, spec D4).
    /// Always `.naive` on the bf16 backend.
    public let kernelPath: KernelPath
    /// Which prefill structure multi-token suffixes take (P5-3/P5-4,
    /// phase-5.md D5). Tiled by default on the packed + fused pipeline;
    /// always `.sequential` on the bf16 backend and the naive kernel arm.
    public let prefillPath: PrefillPath
    /// The tiled prefill chunk size C (spec D2: reported, not pinned).
    /// Meaningful only when `prefillPath == .tiled`; scratch is sized for
    /// it at load and never grown.
    public let prefillChunkSize: Int

    /// Dual timing of the most recent `step` (hard rule 7). Aggregation into
    /// medians/rates is `DecodeTimingCollector`'s job (P2-5).
    public private(set) var lastStepTiming: DispatchTiming?

    /// Compute dispatches encoded by the most recent `step`, measured at the
    /// dispatchThreads call sites (P2-5, spec D5): naive path 591 at the
    /// pinned dims with logits (21/layer × 28 + embedding + final norm +
    /// lm_head), 589 without the logits tail; fused path 199 with logits
    /// (7/layer × 28 + the same head/tail — P4-7's two-pass SDPA, ≤300
    /// gate; was 171 at P4-6, 227 at P4-3), 197 without. The P4-8
    /// token-selecting step encodes one extra dispatch (the argmax
    /// reduction) on top of the with-logits count: fused 200 measured.
    public private(set) var lastStepDispatchCount: Int?

    /// P5-1 (phase-5.md D1): engine-measured span of the most recent
    /// `lastPositionLogits` / `nextGreedyToken` call — wall bracketing the
    /// call, Σ per-step GPU durations, Σ dispatches, steps run. The first
    /// call of a generation processes the whole prompt, so its span is the
    /// prefill span (callers capture it at that boundary; later decode
    /// calls overwrite it with their own one-step spans). nil before the
    /// first call, and after a call that threw mid-way (cleared at call
    /// entry — a failed call never leaves a stale span behind).
    public private(set) var lastCallSpan: ForwardCallSpan?

    /// Tokens whose KV entries currently occupy cache positions
    /// `0..<cachedTokens.count`, in order. `lastPositionLogits` extends this
    /// prefix incrementally and resets on any mismatch.
    public private(set) var cachedTokens: [Int] = []

    private let context: MetalContext
    private let decodeKernels: DecodeKernels
    private let attentionKernels: AttentionKernels
    /// Present exactly on the packed path (created iff any `.q4` ref exists).
    private let quantKernels: QuantKernels?
    /// Present exactly on the fused kernel path (P4-2, spec D4).
    private let fusedSDPA: FusedSDPAKernel?
    /// Present exactly on the fused kernel path (P4-3/P4-6, spec D3 + the
    /// 2026-09-12 addendum): the folding set that, with the two-pass SDPA
    /// (P4-7), takes the packed pipeline to 7 dispatches per layer.
    private let foldedKernels: FoldedKernels?
    /// P4-8: on-GPU token selection for the free-running decode path (both
    /// kernel paths, both formats — it reads only the fp32 logits buffer).
    private let argmaxKernel: ArgmaxKernel
    /// Present exactly on the tiled prefill path (P5-3): the batched
    /// gather/cluster/copy-row kernels plus the P5-2 dequant-GEMM the
    /// chunk projections run through.
    private let prefillKernels: PrefillKernels?
    private let quantGemm: QuantGemmKernel?
    /// PF-1 lever 1: the batched causal SDPA (one dispatch per layer per
    /// chunk, D4 option 2) — present exactly on the tiled prefill path.
    private let prefillSDPA: PrefillSDPAKernel?
    private let dispatchCounter = DispatchCounter()

    private let embeddingRef: MatrixRef
    private let finalNormOffset: Int
    private let lmHeadRef: MatrixRef
    private let layerRefs: [LayerRefs]

    // Scratch buffers, allocated once at init (sizes are config-fixed).
    // The residual stream ping-pongs hiddenA → hiddenB → hiddenA per layer
    // (the second half-block's store lands in the other buffer), ending each
    // layer back in hiddenA. Buffers exist exactly on the path that uses
    // them (spec: fusion only removes allocations, never adds net memory).
    private let hiddenA: MTLBuffer      // fp16 [hidden]
    private let hiddenB: MTLBuffer      // fp16 [hidden]
    private let normed: MTLBuffer       // fp16 [hidden] — norm outputs
    private let qVec: MTLBuffer         // fp16 [numHeads·headDim] — post QK-norm+RoPE
    private let attnOut: MTLBuffer      // fp16 [numHeads·headDim]
    private let actBuf: MTLBuffer       // fp16 [intermediate] — SwiGLU output
    // Naive-path only: standalone-kernel intermediates the folds eliminate.
    private let qRaw: MTLBuffer?        // fp16 [numHeads·headDim]
    private let kRaw: MTLBuffer?        // fp16 [kvHeads·headDim]
    private let kVec: MTLBuffer?        // fp16 [kvHeads·headDim] — post QK-norm
    private let vVec: MTLBuffer?        // fp16 [kvHeads·headDim]
    private let projOut: MTLBuffer?     // fp16 [hidden] — o_proj / down_proj out
    private let gateBuf: MTLBuffer?     // fp16 [intermediate]
    private let upBuf: MTLBuffer?       // fp16 [intermediate]
    // Naive-path only (the online softmax never materializes scores/probs,
    // spec D2's memory win).
    private let scores: MTLBuffer?      // fp32 [numHeads·maxContext]
    private let probs: MTLBuffer?       // fp32 [numHeads·maxContext]
    // Fused-path only (P4-3): the matvec3 output the cluster kernel reads,
    // [numHeads + 2·kvHeads, headDim] (q heads, then k, then v).
    private let qkvBuf: MTLBuffer?
    private let logitsBuf: MTLBuffer    // fp32 [vocab]
    private let argmaxBuf: MTLBuffer    // u32 [1] — P4-8 selected-token slot
    private let cosTable: MTLBuffer     // fp32 [maxContext·headDim/2]
    private let sinTable: MTLBuffer     // fp32 [maxContext·headDim/2]

    /// P5-3 (spec D2): the tiled-prefill chunk scratch — ALL of it
    /// allocated once at model load, sized for `prefillChunkSize`, never
    /// grown (edge test 11 pins buffer identity across a multi-chunk
    /// prefill; the ≤ 64 MiB budget at the pinned dims is pinned by test
    /// via `prefillScratchBytes`). fp16 activations throughout, matching
    /// the per-token pipeline's boundaries; `tokenIds` is the u32 chunk
    /// the batched gather reads. Internal (not private) so the edge-11
    /// identity test can observe the buffers via @testable.
    struct PrefillScratch {
        let tokenIds: MTLBuffer   // u32 [C]
        let hiddenA: MTLBuffer    // fp16 [C, hidden] — residual stream
        let hiddenB: MTLBuffer    // fp16 [C, hidden] — residual ping-pong
        let normBatch: MTLBuffer  // fp16 [C, hidden] — block-norm outputs
        let qBatch: MTLBuffer     // fp16 [C, numHeads·headDim] — q_proj out
        let kBatch: MTLBuffer     // fp16 [C, kvHeads·headDim] — k_proj out
        let vBatch: MTLBuffer     // fp16 [C, kvHeads·headDim] — v_proj out
        let qRoped: MTLBuffer     // fp16 [C, numHeads·headDim] — cluster out
        let attnBatch: MTLBuffer  // fp16 [C, numHeads·headDim] — SDPA out
        let gateBatch: MTLBuffer  // fp16 [C, intermediate]
        let upBatch: MTLBuffer    // fp16 [C, intermediate]
        let actBatch: MTLBuffer   // fp16 [C, intermediate] — SwiGLU out
        let projBatch: MTLBuffer  // fp16 [C, hidden] — o/down projection out

        var allBuffers: [MTLBuffer] {
            [tokenIds, hiddenA, hiddenB, normBatch, qBatch, kBatch, vBatch,
             qRoped, attnBatch, gateBatch, upBatch, actBatch, projBatch]
        }
    }
    let prefillScratch: PrefillScratch?

    /// Total bytes `PrefillScratch` allocates for the given config and
    /// chunk size — the spec D2 ≤ 64 MiB budget is asserted against this
    /// at the pinned dims (test-pinned, not hard-coded policy).
    public static func prefillScratchBytes(
        config: ModelConfig, chunkSize: Int
    ) -> Int {
        let qDim = config.numAttentionHeads * config.headDim
        let kvDim = config.numKeyValueHeads * config.headDim
        return chunkSize * 4
            + 4 * chunkSize * config.hiddenSize * 2
            + 3 * chunkSize * qDim * 2
            + 2 * chunkSize * kvDim * 2
            + 3 * chunkSize * config.intermediateSize * 2
    }

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
            residency: residency, maxContext: maxContext, kernelPath: .naive,
            prefillPath: .sequential,
            prefillChunkSize: Self.defaultPrefillChunkSize)
    }

    /// The P3-5 packed path: identical pipeline, q4g64 matrices consumed by
    /// the P3-4 fused kernels (register dequant, hard rule 1). The packed
    /// file's validation (format tag, provenance, triplet consistency, `.q`
    /// alignment) already happened in `PackedCheckpoint`.
    ///
    /// - Parameter kernelPath: kernel structure (P4-2/P4-4, spec D4).
    ///   `.fused` (the default since P4-4) runs the Phase 4 6-dispatch
    ///   folded layer (P4-6); `.naive` selects the Phase 2/3 21-dispatch
    ///   structure for the in-session before/after rows.
    /// - Parameter prefillPath: prompt-processing structure (P5-3/P5-4,
    ///   spec D5). `.tiled` requires the fused kernel path and preallocates
    ///   the chunk scratch; `.sequential` is the Phase 2–4 per-token loop.
    ///   nil (the default) resolves via `defaultPrefillPath(for:)`: tiled
    ///   on fused, sequential on naive — so the naive A/B arm keeps
    ///   loading without arguments while tiled + naive asked for
    ///   explicitly still fails at load.
    /// - Parameter prefillChunkSize: tiled chunk size C (spec D2 —
    ///   reported, not pinned). nil resolves to min(`defaultPrefillChunkSize`,
    ///   maxContext); an explicit value must lie in 1...maxContext. Ignored
    ///   on the sequential path.
    public convenience init(
        packed: PackedCheckpoint, config: ModelConfig, context: MetalContext,
        residency: WeightsResidency = .mmap, maxContext: Int,
        kernelPath: KernelPath = .fused,
        prefillPath: PrefillPath? = nil,
        prefillChunkSize: Int? = nil
    ) throws {
        try self.init(
            file: packed.file, packed: packed, config: config, context: context,
            residency: residency, maxContext: maxContext, kernelPath: kernelPath,
            prefillPath: prefillPath ?? Self.defaultPrefillPath(for: kernelPath),
            prefillChunkSize: prefillChunkSize
                ?? min(Self.defaultPrefillChunkSize, max(maxContext, 1)))
    }

    private init(
        file: SafetensorsFile, packed: PackedCheckpoint?, config: ModelConfig,
        context: MetalContext, residency: WeightsResidency, maxContext: Int,
        kernelPath: KernelPath, prefillPath: PrefillPath, prefillChunkSize: Int
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
        self.kernelPath = kernelPath
        self.prefillPath = prefillPath
        self.prefillChunkSize = prefillChunkSize
        if prefillPath == .tiled {
            // Tiled prefill exists exactly on the packed + fused pipeline
            // (spec D5); fail at load with a clear error, never mid-prompt.
            guard packed != nil else {
                throw ModelError.badInput(detail:
                    "tiled prefill is packed-pipeline only (phase-5.md D5: "
                    + "the bf16 backend keeps sequential prefill "
                    + "permanently) — load a q4g64 checkpoint or use "
                    + "prefillPath: .sequential")
            }
            guard kernelPath == .fused else {
                throw ModelError.badInput(detail:
                    "tiled prefill requires the fused kernel path "
                    + "(phase-5.md D5: it is the packed-pipeline default "
                    + "structure); kernelPath .naive supports sequential "
                    + "prefill only")
            }
            guard prefillChunkSize >= 1, prefillChunkSize <= maxContext else {
                throw ModelError.badInput(detail:
                    "prefillChunkSize \(prefillChunkSize) must be in "
                    + "1...maxContext (\(maxContext)) — scratch is sized "
                    + "for it at load (phase-5.md D2)")
            }
        }
        if kernelPath == .fused {
            // Fail at load, not mid-decode: the fused kernel's per-lane
            // register budget caps headDim (the pinned model's 128 fits).
            guard config.headDim <= FusedSDPAKernel.maxHeadDim else {
                throw KVCacheError.headDimExceedsFusedLimit(
                    headDim: config.headDim, limit: FusedSDPAKernel.maxHeadDim)
            }
        }

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
        fusedSDPA = kernelPath == .fused ? try FusedSDPAKernel(context: context) : nil
        foldedKernels = kernelPath == .fused ? try FoldedKernels(context: context) : nil
        argmaxKernel = try ArgmaxKernel(context: context)
        prefillKernels = prefillPath == .tiled
            ? try PrefillKernels(context: context) : nil
        quantGemm = prefillPath == .tiled
            ? try QuantGemmKernel(context: context) : nil
        prefillSDPA = prefillPath == .tiled
            ? try PrefillSDPAKernel(context: context) : nil
        decodeKernels.dispatchCounter = dispatchCounter
        attentionKernels.dispatchCounter = dispatchCounter
        quantKernels?.dispatchCounter = dispatchCounter
        fusedSDPA?.dispatchCounter = dispatchCounter
        foldedKernels?.dispatchCounter = dispatchCounter
        argmaxKernel.dispatchCounter = dispatchCounter
        prefillKernels?.dispatchCounter = dispatchCounter
        quantGemm?.dispatchCounter = dispatchCounter
        prefillSDPA?.dispatchCounter = dispatchCounter
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
        qVec = try makeBuffer(bytes: numHeads * headDim * 2)
        attnOut = try makeBuffer(bytes: numHeads * headDim * 2)
        actBuf = try makeBuffer(bytes: intermediate * 2)
        if kernelPath == .naive {
            qRaw = try makeBuffer(bytes: numHeads * headDim * 2)
            kRaw = try makeBuffer(bytes: kvHeads * headDim * 2)
            kVec = try makeBuffer(bytes: kvHeads * headDim * 2)
            vVec = try makeBuffer(bytes: kvHeads * headDim * 2)
            projOut = try makeBuffer(bytes: hidden * 2)
            gateBuf = try makeBuffer(bytes: intermediate * 2)
            upBuf = try makeBuffer(bytes: intermediate * 2)
            scores = try makeBuffer(bytes: numHeads * maxContext * 4)
            probs = try makeBuffer(bytes: numHeads * maxContext * 4)
            qkvBuf = nil
        } else {
            qRaw = nil
            kRaw = nil
            kVec = nil
            vVec = nil
            projOut = nil
            gateBuf = nil
            upBuf = nil
            scores = nil
            probs = nil
            qkvBuf = try makeBuffer(bytes: (numHeads + 2 * kvHeads) * headDim * 2)
        }
        logitsBuf = try makeBuffer(bytes: config.vocabSize * 4)
        argmaxBuf = try makeBuffer(bytes: 4)

        // P5-3 (spec D2): the whole chunk scratch, allocated here and never
        // grown — a multi-chunk prefill reuses these exact buffers (edge
        // test 11 pins identity).
        if prefillPath == .tiled {
            let c = prefillChunkSize
            let qDim = numHeads * headDim
            let kvDim = kvHeads * headDim
            prefillScratch = PrefillScratch(
                tokenIds: try makeBuffer(bytes: c * 4),
                hiddenA: try makeBuffer(bytes: c * hidden * 2),
                hiddenB: try makeBuffer(bytes: c * hidden * 2),
                normBatch: try makeBuffer(bytes: c * hidden * 2),
                qBatch: try makeBuffer(bytes: c * qDim * 2),
                kBatch: try makeBuffer(bytes: c * kvDim * 2),
                vBatch: try makeBuffer(bytes: c * kvDim * 2),
                qRoped: try makeBuffer(bytes: c * qDim * 2),
                attnBatch: try makeBuffer(bytes: c * qDim * 2),
                gateBatch: try makeBuffer(bytes: c * intermediate * 2),
                upBatch: try makeBuffer(bytes: c * intermediate * 2),
                actBatch: try makeBuffer(bytes: c * intermediate * 2),
                projBatch: try makeBuffer(bytes: c * hidden * 2))
        } else {
            prefillScratch = nil
        }

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
        let position = try validateStep(token: token)
        dispatchCounter.reset()
        lastStepTiming = try context.timedDispatch { encoder in
            try encodeForward(
                token: token, position: position,
                computeLogits: computeLogits) { _ in encoder }
        }
        lastStepDispatchCount = dispatchCounter.count
        cachedTokens.append(token)
        return computeLogits ? readLogits() : nil
    }

    /// P4-8: one decode step that also SELECTS the next token on-GPU. The
    /// forward pass, final norm + lm_head, and the argmax reduction all
    /// encode into the same single command buffer (spec D5 — dual timing and
    /// the dispatch count ride along; the argmax is one extra dispatch), and
    /// the per-token readback is 4 bytes instead of the ~605 KB full-vocab
    /// logits. Exact-equality contract (DECISIONS.md 2026-09-12): the
    /// returned token is precisely `Argmax.firstIndex` of the logits this
    /// step computed — lowest index wins ties, no tolerance constant. The
    /// logits stay in `logitsBuf` untouched; `step(computeLogits:)` remains
    /// the oracle suites' full-logits path.
    public func stepSelectingToken(token: Int) throws -> Int {
        let position = try validateStep(token: token)
        dispatchCounter.reset()
        lastStepTiming = try context.timedDispatch { encoder in
            try encodeForward(
                token: token, position: position,
                computeLogits: true) { _ in encoder }
            try argmaxKernel.encodeArgmax(
                into: encoder, values: logitsBuf, count: config.vocabSize,
                output: argmaxBuf)
        }
        lastStepDispatchCount = dispatchCounter.count
        cachedTokens.append(token)
        return Int(argmaxBuf.contents().load(as: UInt32.self))
    }

    /// P4-9 (phase-4.md 2026-09-12 addendum): the DIAGNOSTIC overhead-
    /// anatomy step. Runs the P4-8 token-selecting step verbatim — same
    /// `encodeForward`, same argmax reduction, one command buffer — through
    /// `MetalContext.anatomyDispatch`, so the wall−GPU overhead of a real
    /// decode token splits into encode / commit / commit→GPU-start /
    /// completion-wakeup spans. Opt-in only: the production decode path
    /// never calls this and its numbers are never benchmark rows (P4-1
    /// invariance precedent); `lastStepTiming`/`lastStepDispatchCount` are
    /// cleared, not populated — the returned tuple carries the diagnostic
    /// record and its measured dispatch count instead. Advances the cache
    /// exactly like `stepSelectingToken`, and the returned token obeys the
    /// same exact-equality contract (identical encode ⇒ identical logits ⇒
    /// identical argmax).
    public func anatomyStepSelectingToken(
        token: Int, unretainedReferences: Bool = false
    ) throws -> (token: Int, anatomy: OverheadAnatomy, dispatchCount: Int) {
        let position = try validateStep(token: token)
        dispatchCounter.reset()
        let anatomy = try context.anatomyDispatch(
            unretainedReferences: unretainedReferences
        ) { encoder in
            try encodeForward(
                token: token, position: position,
                computeLogits: true) { _ in encoder }
            try argmaxKernel.encodeArgmax(
                into: encoder, values: logitsBuf, count: config.vocabSize,
                output: argmaxBuf)
        }
        let dispatchCount = dispatchCounter.count
        lastStepTiming = nil
        lastStepDispatchCount = nil
        cachedTokens.append(token)
        return (Int(argmaxBuf.contents().load(as: UInt32.self)), anatomy,
                dispatchCount)
    }

    /// P4-1 (phase-4.md D1): the DIAGNOSTIC attribution step. Runs the same
    /// forward — same kernels, same encode order, arithmetic bitwise
    /// identical to `step` — but splits the encoding into one command buffer
    /// per class-contiguous dispatch run, so per-class GPU time can be read
    /// from the buffers' own timestamps. Buffers commit back-to-back on the
    /// serial queue (no wait between segments) and the whole run is
    /// wall-bracketed (hard rule 7). Opt-in only: the production decode path
    /// (`step` / `lastPositionLogits`) never calls this, and its numbers are
    /// never benchmark rows. Advances the cache exactly like `step`;
    /// `lastStepTiming`/`lastStepDispatchCount` are cleared, not populated —
    /// there is no production single-buffer timing for a diagnostic step.
    public func attributedStep(
        token: Int, computeLogits: Bool
    ) throws -> (attribution: TokenAttribution, logits: [Float]?) {
        let position = try validateStep(token: token)
        dispatchCounter.reset()
        let (segments, wallSeconds) = try runClassSplit { encoderFor in
            try encodeForward(
                token: token, position: position, computeLogits: computeLogits,
                encoderFor: encoderFor)
        }
        lastStepTiming = nil
        lastStepDispatchCount = nil
        cachedTokens.append(token)
        return (
            TokenAttribution(
                position: position, wallSeconds: wallSeconds,
                segments: segments),
            computeLogits ? readLogits() : nil)
    }

    /// The P4-1 class-split machinery shared by `attributedStep` and the
    /// PF-1 `attributedPrefill`: `encode` receives an `encoderFor(class)`
    /// that hands back the current encoder while the class is unchanged
    /// and rolls to a fresh command buffer on every class transition
    /// (one buffer per contiguous same-class dispatch run). Buffers are
    /// committed back-to-back on the serial queue; one wait per buffer at
    /// the end; per-segment GPU timestamps + DispatchCounter deltas come
    /// back in encode order. On a mid-encode throw the already-committed
    /// segments run to completion and the current encoder is ended — the
    /// caller decides what state (cache append) to roll back.
    private func runClassSplit(
        _ encode: ((KernelClass) throws -> MTLComputeCommandEncoder) throws -> Void
    ) throws -> (segments: [TokenAttribution.Segment], wallSeconds: Double) {
        var committed: [(kernelClass: KernelClass, buffer: MTLCommandBuffer, dispatchCount: Int)] = []
        var currentClass: KernelClass?
        var currentBuffer: MTLCommandBuffer?
        var currentEncoder: MTLComputeCommandEncoder?
        var dispatchesAtSegmentStart = 0

        func closeCurrentSegment() {
            guard let buffer = currentBuffer, let encoder = currentEncoder,
                  let kernelClass = currentClass else { return }
            encoder.endEncoding()
            buffer.commit()
            committed.append((
                kernelClass, buffer,
                dispatchCounter.count - dispatchesAtSegmentStart))
            currentBuffer = nil
            currentEncoder = nil
            currentClass = nil
        }

        let wallStart = CACurrentMediaTime()
        do {
            try encode { kernelClass in
                if kernelClass == currentClass, let encoder = currentEncoder {
                    return encoder
                }
                closeCurrentSegment()
                guard let buffer = self.context.queue.makeCommandBuffer() else {
                    throw MetalHarnessError.commandBufferCreationFailed
                }
                guard let encoder = buffer.makeComputeCommandEncoder() else {
                    throw MetalHarnessError.encoderCreationFailed
                }
                currentClass = kernelClass
                currentBuffer = buffer
                currentEncoder = encoder
                dispatchesAtSegmentStart = dispatchCounter.count
                return encoder
            }
        } catch {
            // Already-committed segments run to completion; the caller does
            // NOT append the token(s), so any partial KV write stays
            // unreachable (attention spans 0...position of APPENDED tokens
            // only) and is overwritten by the next append there.
            currentEncoder?.endEncoding()
            throw error
        }
        closeCurrentSegment()

        var segments: [TokenAttribution.Segment] = []
        segments.reserveCapacity(committed.count)
        for (kernelClass, buffer, dispatchCount) in committed {
            buffer.waitUntilCompleted()
            guard buffer.status == .completed else {
                throw MetalHarnessError.gpuExecutionFailed(
                    status: buffer.status, underlying: buffer.error)
            }
            segments.append(TokenAttribution.Segment(
                kernelClass: kernelClass, gpuStart: buffer.gpuStartTime,
                gpuEnd: buffer.gpuEndTime, dispatchCount: dispatchCount))
        }
        let wallEnd = CACurrentMediaTime()
        return (segments, wallEnd - wallStart)
    }

    private func validateStep(token: Int) throws -> Int {
        let position = cachedTokens.count
        guard position < maxContext else {
            throw KVCacheError.contextFull(position: position, maxContext: maxContext)
        }
        guard token >= 0, token < config.vocabSize else {
            throw ModelError.tokenIdOutOfRange(id: token, vocabSize: config.vocabSize)
        }
        return position
    }

    private func readLogits() -> [Float] {
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
    /// stale data: the last layer's post-attention norm on the naive path,
    /// a previous logits step's final norm on the fused path, where the
    /// P4-6 folds compute block norms in registers and never write it).
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

    /// Encodes one token's forward pass. `encoderFor` supplies the encoder
    /// for each dispatch, keyed by the dispatch's `KernelClass` (P4-1 D1):
    /// the production `step` passes a constant provider (every dispatch into
    /// the ONE command buffer — path unchanged), while the diagnostic
    /// `attributedStep` rolls to a fresh command buffer on each class
    /// transition. The pipeline structure lives here exactly once, so the
    /// two modes cannot drift.
    private func encodeForward(
        token: Int, position: Int, computeLogits: Bool,
        encoderFor: (KernelClass) throws -> MTLComputeCommandEncoder
    ) throws {
        let hidden = config.hiddenSize
        let eps = Float(config.rmsNormEps)

        try encodeEmbedding(
            into: try encoderFor(.headTail), token: token, output: hiddenA)

        for (layer, refs) in layerRefs.enumerated() {
            if foldedKernels != nil {
                try encodeFoldedLayer(
                    refs: refs, layer: layer, position: position,
                    encoderFor: encoderFor)
            } else {
                try encodeNaiveLayer(
                    refs: refs, layer: layer, position: position,
                    encoderFor: encoderFor)
            }
        }

        guard computeLogits else { return }
        try decodeKernels.encodeRMSNorm(
            into: try encoderFor(.headTail), input: hiddenA,
            weight: weights.buffer,
            weightByteOffset: finalNormOffset, rows: 1, dim: hidden,
            eps: eps, output: normed)
        try encodeMatvec(
            lmHeadRef, into: try encoderFor(.headTail), input: normed,
            outDim: config.vocabSize, inDim: hidden, output: logitsBuf,
            fp32Output: true)
    }

    /// The Phase 2/3 naive layer structure — 21 dispatches (19 with the
    /// P4-2 fused SDPA never combined here: `.naive` always runs the full
    /// three-kernel attention chain). Unchanged since P2-4; stays selectable
    /// for the P4-5 before/after row and permanent on the bf16 backend.
    private func encodeNaiveLayer(
        refs: LayerRefs, layer: Int, position: Int,
        encoderFor: (KernelClass) throws -> MTLComputeCommandEncoder
    ) throws {
        let hidden = config.hiddenSize
        let headDim = config.headDim
        let numHeads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let intermediate = config.intermediateSize
        let eps = Float(config.rmsNormEps)

        // h = hiddenA on entry. Attention half: hiddenB = h + attn(norm(h)).
        try decodeKernels.encodeRMSNorm(
            into: try encoderFor(.normElementwise), input: hiddenA,
            weight: weights.buffer,
            weightByteOffset: refs.inputNorm, rows: 1, dim: hidden,
            eps: eps, output: normed)
        try encodeMatvec(
            refs.qProj, into: try encoderFor(.matvec), input: normed,
            outDim: numHeads * headDim, inDim: hidden, output: qRaw!)
        try encodeMatvec(
            refs.kProj, into: try encoderFor(.matvec), input: normed,
            outDim: kvHeads * headDim, inDim: hidden, output: kRaw!)
        try encodeMatvec(
            refs.vProj, into: try encoderFor(.matvec), input: normed,
            outDim: kvHeads * headDim, inDim: hidden, output: vVec!)
        // Family order (PIN-1): per-head Q/K RMSNorm, THEN RoPE.
        try decodeKernels.encodeRMSNorm(
            into: try encoderFor(.normElementwise), input: qRaw!,
            weight: weights.buffer,
            weightByteOffset: refs.qNorm, rows: numHeads, dim: headDim,
            eps: eps, output: qVec)
        try decodeKernels.encodeRMSNorm(
            into: try encoderFor(.normElementwise), input: kRaw!,
            weight: weights.buffer,
            weightByteOffset: refs.kNorm, rows: kvHeads, dim: headDim,
            eps: eps, output: kVec!)
        try decodeKernels.encodeRoPE(
            into: try encoderFor(.normElementwise), vector: qVec,
            cosTable: cosTable, sinTable: sinTable,
            position: position, positions: maxContext, heads: numHeads,
            headDim: headDim)
        try decodeKernels.encodeRoPE(
            into: try encoderFor(.normElementwise), vector: kVec!,
            cosTable: cosTable, sinTable: sinTable,
            position: position, positions: maxContext, heads: kvHeads,
            headDim: headDim)
        try attentionKernels.encodeKVAppend(
            into: try encoderFor(.normElementwise), cache: kvCache,
            layer: layer, component: .key,
            position: position, vector: kVec!)
        try attentionKernels.encodeKVAppend(
            into: try encoderFor(.normElementwise), cache: kvCache,
            layer: layer, component: .value,
            position: position, vector: vVec!)
        try attentionKernels.encodeAttentionScores(
            into: try encoderFor(.attention), cache: kvCache, layer: layer,
            position: position,
            query: qVec, numHeads: numHeads, scores: scores!)
        try attentionKernels.encodeSoftmaxRows(
            into: try encoderFor(.attention), input: scores!, rows: numHeads,
            count: position + 1,
            rowStride: maxContext, output: probs!)
        try attentionKernels.encodeAttentionPV(
            into: try encoderFor(.attention), cache: kvCache, layer: layer,
            position: position,
            probs: probs!, numHeads: numHeads, output: attnOut)
        try encodeMatvec(
            refs.oProj, into: try encoderFor(.matvec), input: attnOut,
            outDim: hidden, inDim: numHeads * headDim, output: projOut!)
        try decodeKernels.encodeResidualAdd(
            into: try encoderFor(.normElementwise), a: hiddenA, b: projOut!,
            count: hidden, output: hiddenB)

        // MLP half: hiddenA = hiddenB + mlp(norm(hiddenB)).
        try decodeKernels.encodeRMSNorm(
            into: try encoderFor(.normElementwise), input: hiddenB,
            weight: weights.buffer,
            weightByteOffset: refs.postAttentionNorm, rows: 1, dim: hidden,
            eps: eps, output: normed)
        try encodeMatvec(
            refs.gateProj, into: try encoderFor(.matvec), input: normed,
            outDim: intermediate, inDim: hidden, output: gateBuf!)
        try encodeMatvec(
            refs.upProj, into: try encoderFor(.matvec), input: normed,
            outDim: intermediate, inDim: hidden, output: upBuf!)
        try decodeKernels.encodeSwiGLU(
            into: try encoderFor(.normElementwise), gate: gateBuf!,
            up: upBuf!, count: intermediate,
            output: actBuf)
        try encodeMatvec(
            refs.downProj, into: try encoderFor(.matvec), input: actBuf,
            outDim: hidden, inDim: intermediate, output: projOut!)
        try decodeKernels.encodeResidualAdd(
            into: try encoderFor(.normElementwise), a: hiddenB, b: projOut!,
            count: hidden, output: hiddenA)
    }

    /// P4-3 (spec D3) + P4-6 (norm→matvec folds) + P4-7 (split-K SDPA):
    /// the folded layer — 7 dispatches. Norm+matvec3 (input norm folded
    /// into the QKV concat) → fused qk-norm/RoPE/append cluster → two-pass
    /// SDPA (split partials + reduce) → o_proj+residual →
    /// norm+gate+up+SwiGLU (post-norm folded) →
    /// down+residual. Every fold keeps the naive chain's fp16-boundary/
    /// fp32-accumulate semantics (FoldedKernels doc); the residual
    /// ping-pong (hiddenA → hiddenB → hiddenA) is unchanged, so the Tier-E
    /// hook points hold. The norm-folded dispatches encode under the
    /// `.matvec` attribution class — the block-norm boundary is no longer
    /// sliceable (spec D5), so its time rides the matvec class from P4-6 on.
    private func encodeFoldedLayer(
        refs: LayerRefs, layer: Int, position: Int,
        encoderFor: (KernelClass) throws -> MTLComputeCommandEncoder
    ) throws {
        let hidden = config.hiddenSize
        let headDim = config.headDim
        let numHeads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let intermediate = config.intermediateSize
        let eps = Float(config.rmsNormEps)
        let foldedKernels = foldedKernels!
        let qkvBuf = qkvBuf!

        // h = hiddenA on entry. Attention half: hiddenB = h + attn(norm(h)).
        try foldedKernels.encodeNormMatvec3(
            into: try encoderFor(.matvec),
            a: try q4Triplet(refs.qProj), outDimA: numHeads * headDim,
            b: try q4Triplet(refs.kProj), outDimB: kvHeads * headDim,
            c: try q4Triplet(refs.vProj), outDimC: kvHeads * headDim,
            inDim: hidden, input: hiddenA,
            normWeight: weights.buffer, normByteOffset: refs.inputNorm,
            eps: eps, output: qkvBuf)
        // Family order (PIN-1) inside the cluster: per-head Q/K RMSNorm,
        // THEN RoPE; k stores into the cache slot, v copies into its slot.
        try foldedKernels.encodeQKNormRoPEAppend(
            into: try encoderFor(.normElementwise), qkv: qkvBuf,
            qNormWeight: weights.buffer, qNormByteOffset: refs.qNorm,
            kNormWeight: weights.buffer, kNormByteOffset: refs.kNorm,
            cosTable: cosTable, sinTable: sinTable,
            position: position, positions: maxContext, numHeads: numHeads,
            eps: eps, cache: kvCache, layer: layer, qOut: qVec)
        try fusedSDPA!.encodeSDPA(
            into: try encoderFor(.attention), cache: kvCache,
            layer: layer, position: position,
            query: qVec, numHeads: numHeads, output: attnOut)
        try foldedKernels.encodeMatvecResidual(
            into: try encoderFor(.matvec), triplet: try q4Triplet(refs.oProj),
            outDim: hidden, inDim: numHeads * headDim,
            input: attnOut, residual: hiddenA, output: hiddenB)

        // MLP half: hiddenA = hiddenB + mlp(norm(hiddenB)).
        try foldedKernels.encodeNormGateUpSwiGLU(
            into: try encoderFor(.matvec),
            gate: try q4Triplet(refs.gateProj), up: try q4Triplet(refs.upProj),
            outDim: intermediate, inDim: hidden, input: hiddenB,
            normWeight: weights.buffer, normByteOffset: refs.postAttentionNorm,
            eps: eps, output: actBuf)
        try foldedKernels.encodeMatvecResidual(
            into: try encoderFor(.matvec), triplet: try q4Triplet(refs.downProj),
            outDim: hidden, inDim: intermediate,
            input: actBuf, residual: hiddenB, output: hiddenA)
    }

    /// The packed triplet of a `.q4` ref as FoldedKernels input. The fused
    /// path exists only on the packed pipeline, so every matrix ref is
    /// `.q4` by construction; `.bf16` here is an internal inconsistency.
    private func q4Triplet(_ ref: MatrixRef) throws -> FoldedKernels.Triplet {
        guard case .q4(let q, let scales, let biases) = ref else {
            throw ModelError.badInput(detail:
                "internal inconsistency: fused kernel path reached a bf16 "
                + "weight ref — the fused path is packed-only (spec D4)")
        }
        return FoldedKernels.Triplet(
            q: weights.buffer, qByteOffset: q,
            scales: weights.buffer, scalesByteOffset: scales,
            biases: weights.buffer, biasesByteOffset: biases)
    }

    // MARK: - P5-3 chunked batched prefill (phase-5.md D2/D4)

    /// Runs the whole uncached suffix through the tiled prefill: chunks of
    /// `prefillChunkSize` positions, ONE command buffer per chunk (dual
    /// timing + dispatch count per chunk, hard rule 7), the last chunk
    /// carrying the final-norm + lm_head tail — logits are computed exactly
    /// once per prefill, at the last position (spec D2) — plus the argmax
    /// reduction when selecting. Validation is up front (token ids, total
    /// context capacity) so a bad prompt throws before any dispatch or
    /// cache write, with the same error payloads the sequential path
    /// eventually throws (edge test 10 pins identity).
    private func runTiledPrefill(
        ids: [Int], selectToken: Bool
    ) throws -> (logits: [Float]?, token: Int?) {
        let scratch = prefillScratch!  // exists iff prefillPath == .tiled
        // P5-1 (spec D1): cleared FIRST — a call that throws anywhere
        // (validation included) leaves nil, never a stale span. The
        // sequential path's mid-loop throws have the same effect.
        lastCallSpan = nil
        let suffix = try validatedTiledSuffix(ids: ids)

        let spanWallStart = CACurrentMediaTime()
        var spanGPUSeconds = 0.0
        var spanDispatches = 0
        var logits: [Float]?
        var token: Int?
        var index = 0
        while index < suffix.count {
            let chunk = Array(
                suffix[index..<min(index + prefillChunkSize, suffix.count)])
            let isLast = index + chunk.count == suffix.count
            // Host-side id upload: the previous chunk's command buffer has
            // completed (timedDispatch is synchronous), so the buffer is
            // idle; ids were range-validated above.
            let idsPtr = scratch.tokenIds.contents()
                .assumingMemoryBound(to: UInt32.self)
            for (i, t) in chunk.enumerated() { idsPtr[i] = UInt32(t) }

            dispatchCounter.reset()
            lastStepTiming = try context.timedDispatch { encoder in
                try encodePrefillChunk(
                    batch: chunk.count, basePosition: cachedTokens.count,
                    computeLogits: isLast,
                    selectToken: isLast && selectToken
                ) { _ in encoder }
            }
            lastStepDispatchCount = dispatchCounter.count
            spanGPUSeconds += lastStepTiming?.gpuDuration ?? 0
            spanDispatches += lastStepDispatchCount ?? 0
            cachedTokens.append(contentsOf: chunk)
            if isLast {
                if selectToken {
                    token = Int(argmaxBuf.contents().load(as: UInt32.self))
                } else {
                    logits = readLogits()
                }
            }
            index += chunk.count
        }
        lastCallSpan = ForwardCallSpan(
            stepCount: suffix.count,
            wallSeconds: CACurrentMediaTime() - spanWallStart,
            gpuSeconds: spanGPUSeconds, dispatchCount: spanDispatches)
        return (logits, token)
    }

    /// The uncached suffix of `ids`, validated up front (token ids, total
    /// context capacity) so a bad prompt throws before any dispatch or
    /// cache write, with the same error payloads the sequential path
    /// eventually throws (edge test 10 pins identity).
    private func validatedTiledSuffix(ids: [Int]) throws -> [Int] {
        let suffix = Array(ids[cachedTokens.count...])
        for token in suffix {
            guard token >= 0, token < config.vocabSize else {
                throw ModelError.tokenIdOutOfRange(
                    id: token, vocabSize: config.vocabSize)
            }
        }
        guard cachedTokens.count + suffix.count <= maxContext else {
            // The sequential loop appends until the cache fills, then
            // throws from validateStep with position == maxContext —
            // identical payload here, thrown before any work.
            throw KVCacheError.contextFull(
                position: maxContext, maxContext: maxContext)
        }
        return suffix
    }

    /// PF-1 (measure-first): the DIAGNOSTIC class-split form of the tiled
    /// prefill — the same `encodePrefillChunk` structure, one command
    /// buffer per contiguous same-class dispatch run per chunk (the P4-1
    /// `attributedStep` machinery), so per-class GPU time inside each chunk
    /// is attributable: GEMM / attention / norm+elementwise / head-tail.
    /// Same validation, same cache writes, same logits as the production
    /// path (production-path invariance is a pre-committed bound —
    /// DECISIONS.md 2026-09-18). Requires the tiled prefill path; resets
    /// the cache first so every attributed prefill runs the whole prompt
    /// from depth 0 (deterministic depth accounting). Production timing
    /// fields (`lastStepTiming`, `lastCallSpan`) are cleared — diagnostic
    /// runs are never rows.
    public func attributedPrefill(
        ids: [Int]
    ) throws -> (attribution: PrefillAttribution, logits: [Float]) {
        guard prefillPath == .tiled, let scratch = prefillScratch else {
            throw ModelError.badInput(detail:
                "attributedPrefill needs the tiled prefill path "
                + "(phase-5.md D5) — this model runs sequential prefill")
        }
        guard ids.count > 1 else {
            throw ModelError.badInput(detail:
                "attributedPrefill needs a multi-token prompt (got "
                + "\(ids.count)); single tokens are decode steps")
        }
        reset()
        lastCallSpan = nil
        lastStepTiming = nil
        lastStepDispatchCount = nil
        let suffix = try validatedTiledSuffix(ids: ids)

        var chunks: [TokenAttribution] = []
        var chunkSizes: [Int] = []
        var index = 0
        while index < suffix.count {
            let chunk = Array(
                suffix[index..<min(index + prefillChunkSize, suffix.count)])
            let isLast = index + chunk.count == suffix.count
            let idsPtr = scratch.tokenIds.contents()
                .assumingMemoryBound(to: UInt32.self)
            for (i, t) in chunk.enumerated() { idsPtr[i] = UInt32(t) }
            let basePosition = cachedTokens.count

            dispatchCounter.reset()
            let (segments, wallSeconds) = try runClassSplit { encoderFor in
                try encodePrefillChunk(
                    batch: chunk.count, basePosition: basePosition,
                    computeLogits: isLast, selectToken: false,
                    encoderFor: encoderFor)
            }
            cachedTokens.append(contentsOf: chunk)
            chunks.append(TokenAttribution(
                position: basePosition, wallSeconds: wallSeconds,
                segments: segments))
            chunkSizes.append(chunk.count)
            index += chunk.count
        }
        return (
            PrefillAttribution(chunks: chunks, chunkSizes: chunkSizes),
            readLogits())
    }

    /// Encodes one prefill chunk of `batch` positions (spec D2): batched
    /// embedding gather → per layer [cooperative batched input norm
    /// (PF-1) → q/k/v GEMMs → batched qk-norm/RoPE/append → ONE batched
    /// causal SDPA dispatch (PF-1, D4 option 2) → o_proj GEMM → batched
    /// residual → cooperative batched post-norm → gate/up GEMMs → batched
    /// SwiGLU → down GEMM → batched residual] → (last chunk only) last-row
    /// copy + final norm + lm_head (+ argmax when selecting). 14 dispatches
    /// per layer, chunk-size independent (was 13 + 2·B at P5-3). The
    /// residual ping-pong
    /// (hiddenA → hiddenB → hiddenA per layer) mirrors the per-token
    /// pipeline row-for-row. `encoderFor(class)` supplies the encoder per
    /// dispatch: the production path returns ONE encoder for every class
    /// (one command buffer per chunk); the PF-1 diagnostic
    /// `attributedPrefill` rolls to a fresh buffer on each class transition.
    /// The chunk structure lives here exactly once, so the two modes cannot
    /// drift (the P4-1 `encodeForward` pattern).
    private func encodePrefillChunk(
        batch: Int, basePosition: Int, computeLogits: Bool,
        selectToken: Bool,
        encoderFor: (KernelClass) throws -> MTLComputeCommandEncoder
    ) throws {
        let scratch = prefillScratch!
        let prefill = prefillKernels!
        let hidden = config.hiddenSize
        let headDim = config.headDim
        let numHeads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let intermediate = config.intermediateSize
        let eps = Float(config.rmsNormEps)
        let qDim = numHeads * headDim
        let kvDim = kvHeads * headDim

        guard case .q4(let embQ, let embScales, let embBiases) = embeddingRef
        else {
            throw ModelError.badInput(detail:
                "internal inconsistency: tiled prefill reached a bf16 "
                + "embedding ref — tiled prefill is packed-only (spec D5)")
        }
        try prefill.encodeEmbeddingGatherBatch(
            into: try encoderFor(.headTail), q: weights.buffer, qByteOffset: embQ,
            scales: weights.buffer, scalesByteOffset: embScales,
            biases: weights.buffer, biasesByteOffset: embBiases,
            tokenIds: scratch.tokenIds, batch: batch,
            vocabSize: config.vocabSize, hiddenSize: hidden,
            output: scratch.hiddenA)

        for (layer, refs) in layerRefs.enumerated() {
            // Attention half: hiddenB = h + attn(norm(h)). PF-1 lever 2:
            // the cooperative row norm (the naive rmsnorm_f16 summed the
            // whole row per THREAD — O(dim²) per row, 33.5% of the Mac
            // prefill span at the PF-1 attribution).
            try prefill.encodeRMSNormRows(
                into: try encoderFor(.normElementwise), input: scratch.hiddenA,
                weight: weights.buffer,
                weightByteOffset: refs.inputNorm, rows: batch, dim: hidden,
                eps: eps, output: scratch.normBatch)
            try encodeGemm(
                refs.qProj, into: try encoderFor(.gemm), input: scratch.normBatch,
                batch: batch, outDim: qDim, inDim: hidden,
                output: scratch.qBatch)
            try encodeGemm(
                refs.kProj, into: try encoderFor(.gemm), input: scratch.normBatch,
                batch: batch, outDim: kvDim, inDim: hidden,
                output: scratch.kBatch)
            try encodeGemm(
                refs.vProj, into: try encoderFor(.gemm), input: scratch.normBatch,
                batch: batch, outDim: kvDim, inDim: hidden,
                output: scratch.vBatch)
            try prefill.encodeQKNormRoPEAppendBatch(
                into: try encoderFor(.normElementwise), qIn: scratch.qBatch, kIn: scratch.kBatch,
                vIn: scratch.vBatch,
                qNormWeight: weights.buffer, qNormByteOffset: refs.qNorm,
                kNormWeight: weights.buffer, kNormByteOffset: refs.kNorm,
                cosTable: cosTable, sinTable: sinTable,
                basePosition: basePosition, batch: batch,
                positions: maxContext, numHeads: numHeads, eps: eps,
                cache: kvCache, layer: layer, qOut: scratch.qRoped)
            // Causal attention (PF-1, spec D4 option 2): the chunk's K/V are
            // all appended; ONE dispatch covers every chunk position, each
            // reading depth 0...basePosition+p only — masking within the
            // chunk is the depth limit itself. Replaces the P5-3 loop of
            // 2·B per-position split-K dispatches (measured 21.9% of the
            // Mac prefill span across 47,712 dispatches at 852 tokens).
            try prefillSDPA!.encodeCausalSDPABatch(
                into: try encoderFor(.attention), cache: kvCache, layer: layer,
                basePosition: basePosition, batch: batch,
                query: scratch.qRoped, numHeads: numHeads,
                output: scratch.attnBatch)
            try encodeGemm(
                refs.oProj, into: try encoderFor(.gemm), input: scratch.attnBatch,
                batch: batch, outDim: hidden, inDim: qDim,
                output: scratch.projBatch)
            try decodeKernels.encodeResidualAdd(
                into: try encoderFor(.normElementwise), a: scratch.hiddenA,
                b: scratch.projBatch,
                count: batch * hidden, output: scratch.hiddenB)

            // MLP half: hiddenA = hiddenB + mlp(norm(hiddenB)).
            try prefill.encodeRMSNormRows(
                into: try encoderFor(.normElementwise), input: scratch.hiddenB,
                weight: weights.buffer,
                weightByteOffset: refs.postAttentionNorm, rows: batch,
                dim: hidden, eps: eps, output: scratch.normBatch)
            try encodeGemm(
                refs.gateProj, into: try encoderFor(.gemm), input: scratch.normBatch,
                batch: batch, outDim: intermediate, inDim: hidden,
                output: scratch.gateBatch)
            try encodeGemm(
                refs.upProj, into: try encoderFor(.gemm), input: scratch.normBatch,
                batch: batch, outDim: intermediate, inDim: hidden,
                output: scratch.upBatch)
            try decodeKernels.encodeSwiGLU(
                into: try encoderFor(.normElementwise), gate: scratch.gateBatch, up: scratch.upBatch,
                count: batch * intermediate, output: scratch.actBatch)
            try encodeGemm(
                refs.downProj, into: try encoderFor(.gemm), input: scratch.actBatch,
                batch: batch, outDim: hidden, inDim: intermediate,
                output: scratch.projBatch)
            try decodeKernels.encodeResidualAdd(
                into: try encoderFor(.normElementwise), a: scratch.hiddenB,
                b: scratch.projBatch,
                count: batch * hidden, output: scratch.hiddenA)
        }

        guard computeLogits else { return }
        // Last-position-only lm_head (spec D2): the last chunk position's
        // residual moves into the decode hidden buffer — so the Tier-E
        // lastLayerOutput hook reads the same buffer on both prefill paths
        // — then the per-token final-norm + lm_head tail runs unchanged.
        try prefill.encodeCopyRow(
            into: try encoderFor(.headTail), source: scratch.hiddenA, row: batch - 1,
            count: hidden, output: hiddenA)
        try decodeKernels.encodeRMSNorm(
            into: try encoderFor(.headTail), input: hiddenA,
            weight: weights.buffer,
            weightByteOffset: finalNormOffset, rows: 1, dim: hidden,
            eps: eps, output: normed)
        try encodeMatvec(
            lmHeadRef, into: try encoderFor(.headTail), input: normed,
            outDim: config.vocabSize, inDim: hidden, output: logitsBuf,
            fp32Output: true)
        if selectToken {
            try argmaxKernel.encodeArgmax(
                into: try encoderFor(.headTail), values: logitsBuf,
                count: config.vocabSize, output: argmaxBuf)
        }
    }

    /// One GEMM against a packed matrix ref through the P5-2 kernel.
    /// Tiled prefill is packed-only (spec D5), so every ref is `.q4` by
    /// construction; `.bf16` here is an internal inconsistency.
    private func encodeGemm(
        _ ref: MatrixRef, into encoder: MTLComputeCommandEncoder,
        input: MTLBuffer, batch: Int, outDim: Int, inDim: Int,
        output: MTLBuffer
    ) throws {
        guard case .q4(let q, let scales, let biases) = ref else {
            throw ModelError.badInput(detail:
                "internal inconsistency: tiled prefill reached a bf16 "
                + "weight ref — tiled prefill is packed-only (spec D5)")
        }
        try quantGemm!.encodeGemm(
            into: encoder, q: weights.buffer, qByteOffset: q,
            scales: weights.buffer, scalesByteOffset: scales,
            biases: weights.buffer, biasesByteOffset: biases,
            input: input, batchM: batch, outDim: outDim, inDim: inDim,
            output: output)
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
        // P5-3 (spec D2/D5): multi-token suffixes take the tiled chunked
        // prefill when selected; single-token suffixes — decode steps —
        // stay on the unchanged per-token path below.
        if prefillPath == .tiled, ids.count - cachedTokens.count > 1 {
            return try runTiledPrefill(ids: ids, selectToken: false).logits!
        }
        // P5-1 (spec D1): the whole call is one dual-timed span — on the
        // first call of a generation it is the prefill span. Cleared up
        // front so a mid-call throw leaves nil, never a stale span.
        lastCallSpan = nil
        let spanWallStart = CACurrentMediaTime()
        var spanGPUSeconds = 0.0
        var spanDispatches = 0
        var spanSteps = 0
        var logits: [Float]?
        for i in cachedTokens.count..<ids.count {
            logits = try step(token: ids[i], computeLogits: i == ids.count - 1)
            spanGPUSeconds += lastStepTiming?.gpuDuration ?? 0
            spanDispatches += lastStepDispatchCount ?? 0
            spanSteps += 1
        }
        lastCallSpan = ForwardCallSpan(
            stepCount: spanSteps,
            wallSeconds: CACurrentMediaTime() - spanWallStart,
            gpuSeconds: spanGPUSeconds, dispatchCount: spanDispatches)
        // The loop ran at least once (cachedTokens.count < ids.count holds in
        // both branches above) and its last iteration computed logits.
        return logits!
    }

    /// P4-8: the on-GPU token-selection form of `lastPositionLogits` — same
    /// incremental-prefix contract, but the last step runs
    /// `stepSelectingToken` so only the chosen token id crosses back to the
    /// CPU. Token-identical to the default implementation (CPU argmax over
    /// `lastPositionLogits`) by the exact-equality contract, pinned by test.
    public func nextGreedyToken(ids: [Int]) throws -> Int {
        guard !ids.isEmpty else {
            throw ModelError.badInput(detail: "nextGreedyToken of an empty sequence")
        }
        if !(ids.count > cachedTokens.count && ids.starts(with: cachedTokens)) {
            reset()
        }
        // P5-3 (spec D2/D5): multi-token suffixes take the tiled chunked
        // prefill when selected; the argmax reduction rides the last
        // chunk's command buffer, so only the chosen token id crosses back.
        if prefillPath == .tiled, ids.count - cachedTokens.count > 1 {
            return try runTiledPrefill(ids: ids, selectToken: true).token!
        }
        // P5-1 (spec D1): span the call — through the selecting step's
        // 4-byte argmax readback, so the span ends when the chosen token
        // (the last prompt position's output on a prefill call) is
        // available on the CPU. Cleared up front so a mid-call throw
        // leaves nil, never a stale span.
        lastCallSpan = nil
        let spanWallStart = CACurrentMediaTime()
        var spanGPUSeconds = 0.0
        var spanDispatches = 0
        var spanSteps = 0
        for i in cachedTokens.count..<(ids.count - 1) {
            try step(token: ids[i], computeLogits: false)
            spanGPUSeconds += lastStepTiming?.gpuDuration ?? 0
            spanDispatches += lastStepDispatchCount ?? 0
            spanSteps += 1
        }
        let token = try stepSelectingToken(token: ids[ids.count - 1])
        spanGPUSeconds += lastStepTiming?.gpuDuration ?? 0
        spanDispatches += lastStepDispatchCount ?? 0
        spanSteps += 1
        lastCallSpan = ForwardCallSpan(
            stepCount: spanSteps,
            wallSeconds: CACurrentMediaTime() - spanWallStart,
            gpuSeconds: spanGPUSeconds, dispatchCount: spanDispatches)
        return token
    }
}
