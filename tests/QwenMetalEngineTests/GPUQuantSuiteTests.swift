import XCTest
@testable import QwenMetalEngine
import Metal

/// Shared GPU-quant pipeline for the P3-5 suites: ONE `GPUModel` over the
/// real packed artifact (mmap residency), shared across the Tier-M/E suites
/// and the free-run report — the SharedGPUModel pattern. maxContext 256
/// covers the activation slices (seq 5), the logit suite's longest
/// teacher-forced sequence, and the 128-step free run.
enum SharedQuantGPUModel {
    static let maxContext = 256

    private static var sharedModel: Result<GPUModel, Error>?

    static func model() throws -> GPUModel {
        if sharedModel == nil {
            sharedModel = Result {
                let packed = try PackedCheckpoint(
                    path: SharedQuantModel.packedURL.path,
                    expectedRevision: SharedCheckpoint.pinnedRevision)
                let config = try ModelConfig(
                    jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
                return try GPUModel(
                    packed: packed, config: config,
                    context: try SharedGPUModel.metalContext(),
                    maxContext: maxContext)
            }
        }
        return try sharedModel!.get()
    }

    /// Call from setUpWithError: skips without the local packed artifact or
    /// without a Metal device (clean skip, not a crash).
    static func skipUnlessReady() throws {
        guard FileManager.default.fileExists(
            atPath: SharedQuantModel.packedURL.path) else {
            throw XCTSkip(
                "packed artifact missing at \(SharedQuantModel.packedURL.path) "
                + "(local-only — produce it with `swift run qwen-metal-cli pack ...`)")
        }
        do {
            _ = try SharedGPUModel.metalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }
}

// MARK: - Tier M: isolated GPU-quant modules vs the LIVE CPU-quant oracle
//
// Phase 3 gates entry: "Phase 2 Tier M constants reused verbatim with the
// oracle swapped to the CPU-quant reference, computed live at the same slice
// points and suite structure." Same slice points as GPUTierMFixtureTests
// (embeddings, layer-0 input norm, layer-0 attention); the oracle activations
// come from the shared CPU-quant model instead of dumped fixtures — both
// sides consume bit-identical dequant values (gates-entry premise).
//
// Heavy setup (the CPU-quant model materializes ~7 GB): run release-mode
// alongside the other oracle suites (DEV-1 precedent).
final class GPUQuantTierMTests: XCTestCase {

    override func setUpWithError() throws {
        try SharedQuantGPUModel.skipUnlessReady()
    }

    private func promptIDs() throws -> [Int] {
        try SharedCheckpoint.promptFixture("short_english").inputIds
    }

    private func makeF16Buffer(_ values: [Float]) throws -> MTLBuffer {
        let context = try SharedGPUModel.metalContext()
        let halves = values.map(Float16.init)
        guard let buffer = context.device.makeBuffer(
            length: max(halves.count, 1) * 2, options: .storageModeShared) else {
            throw MetalHarnessError.bufferAllocationFailed(length: halves.count * 2)
        }
        halves.withUnsafeBytes {
            buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        return buffer
    }

    private func makeEmptyBuffer(bytes: Int) throws -> MTLBuffer {
        let context = try SharedGPUModel.metalContext()
        guard let buffer = context.device.makeBuffer(
            length: bytes, options: .storageModeShared) else {
            throw MetalHarnessError.bufferAllocationFailed(length: bytes)
        }
        return buffer
    }

    private func readF16(_ buffer: MTLBuffer, count: Int) -> [Float] {
        buffer.contents().withMemoryRebound(to: Float16.self, capacity: count) {
            UnsafeBufferPointer(start: $0, count: count).map(Float.init)
        }
    }

    /// The packed triplet's three byte offsets inside the shared model's
    /// whole-checkpoint buffer.
    private func triplet(_ base: String) throws -> (q: Int, scales: Int, biases: Int) {
        let weights = try SharedQuantGPUModel.model().weights
        return (
            q: try weights.byteOffset(for: base + Q4G64.qSuffix),
            scales: try weights.byteOffset(for: base + Q4G64.scalesSuffix),
            biases: try weights.byteOffset(for: base + Q4G64.biasesSuffix))
    }

    /// Tier-M EXACT gate: GPU embedding-gather fp16 == fp16(CPU-quant fp32)
    /// bitwise, on the real packed embedding triplet at real file offsets.
    func testEmbeddingGatherMatchesCPUQuantExactlyInFP16() throws {
        let model = try SharedQuantGPUModel.model()
        let cpu = try SharedQuantModel.model()
        let context = try SharedGPUModel.metalContext()
        let kernels = try QuantKernels(context: context)
        let embedding = try triplet("model.embed_tokens.weight")
        let hidden = model.config.hiddenSize
        let ids = try promptIDs()
        let reference = try cpu.embedding(ids)
        let out = try makeEmptyBuffer(bytes: hidden * 2)

        for (row, token) in ids.enumerated() {
            try context.timedDispatch { encoder in
                try kernels.encodeEmbeddingGather(
                    into: encoder, q: model.weights.buffer, qByteOffset: embedding.q,
                    scales: model.weights.buffer, scalesByteOffset: embedding.scales,
                    biases: model.weights.buffer, biasesByteOffset: embedding.biases,
                    vocabSize: model.config.vocabSize, hiddenSize: hidden,
                    tokenId: token, output: out)
            }
            let gpuHalves = out.contents().withMemoryRebound(
                to: Float16.self, capacity: hidden
            ) { Array(UnsafeBufferPointer(start: $0, count: hidden)) }
            for i in 0..<hidden {
                let expected = Float16(reference[row * hidden + i])
                if gpuHalves[i].bitPattern != expected.bitPattern {
                    return XCTFail(
                        "embedding row \(row) element \(i): GPU \(gpuHalves[i]) "
                        + "!= fp16(CPU-quant) \(expected) (exact bitwise match "
                        + "required — Phase 3 gates)")
                }
            }
        }
    }

    /// Tier-M gate 2⁻⁸: layer-0 input RMSNorm fed the CPU-quant reference
    /// input (bf16 pass-through norm weights read from the PACKED file).
    func testInputLayerNormMatchesCPUQuant() throws {
        let model = try SharedQuantGPUModel.model()
        let cpu = try SharedQuantModel.model()
        let context = try SharedGPUModel.metalContext()
        let kernels = try DecodeKernels(context: context)
        let hidden = model.config.hiddenSize
        let referenceInput = try cpu.embedding(try promptIDs())
        let reference = try cpu.blocks[0].inputNorm(referenceInput)
        let input = try makeF16Buffer(referenceInput)
        let out = try makeEmptyBuffer(bytes: 5 * hidden * 2)

        try context.timedDispatch { encoder in
            try kernels.encodeRMSNorm(
                into: encoder, input: input, weight: model.weights.buffer,
                weightByteOffset: try model.weights.byteOffset(
                    for: "model.layers.0.input_layernorm.weight"),
                rows: 5, dim: hidden, eps: Float(model.config.rmsNormEps),
                output: out)
        }
        assertWithinPhase2Gate(
            readF16(out, count: 5 * hidden), reference,
            rel: exp2(-8), slice: "layer0_pre_attn_norm_output (CPU-quant live)")
    }

    /// Tier-M gate 2⁻⁷: the layer-0 attention module — quant matvecs for
    /// q/k/v/o, shared Phase 2 attention kernels — fed the CPU-quant
    /// reference input slice, run position-by-position over a fresh 1-layer
    /// KV cache (the GPUTierMFixtureTests structure with the oracle live).
    func testAttentionOutputMatchesCPUQuant() throws {
        let model = try SharedQuantGPUModel.model()
        let cpu = try SharedQuantModel.model()
        let context = try SharedGPUModel.metalContext()
        let kernels = try DecodeKernels(context: context)
        let quant = try QuantKernels(context: context)
        let attention = try AttentionKernels(context: context)
        let config = model.config
        let hidden = config.hiddenSize
        let headDim = config.headDim
        let numHeads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let eps = Float(config.rmsNormEps)
        let seqLen = 5

        let qProj = try triplet("model.layers.0.self_attn.q_proj.weight")
        let kProj = try triplet("model.layers.0.self_attn.k_proj.weight")
        let vProj = try triplet("model.layers.0.self_attn.v_proj.weight")
        let oProj = try triplet("model.layers.0.self_attn.o_proj.weight")
        func normOffset(_ suffix: String) throws -> Int {
            try model.weights.byteOffset(for: "model.layers.0.self_attn." + suffix)
        }
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: seqLen, headDim: headDim)
        let rope = try RoPE(
            headDim: headDim, theta: config.ropeTheta, positions: seqLen)
        let cosTable = try makeEmptyBuffer(bytes: rope.cosValues.count * 4)
        let sinTable = try makeEmptyBuffer(bytes: rope.sinValues.count * 4)
        rope.cosValues.withUnsafeBytes {
            cosTable.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        rope.sinValues.withUnsafeBytes {
            sinTable.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }

        let referenceInput = try cpu.blocks[0].inputNorm(
            try cpu.embedding(try promptIDs()))
        let reference = try cpu.blocks[0].attention(referenceInput, seqLen: seqLen)

        let input = try makeEmptyBuffer(bytes: hidden * 2)
        let qRaw = try makeEmptyBuffer(bytes: numHeads * headDim * 2)
        let qVec = try makeEmptyBuffer(bytes: numHeads * headDim * 2)
        let kRaw = try makeEmptyBuffer(bytes: kvHeads * headDim * 2)
        let kVec = try makeEmptyBuffer(bytes: kvHeads * headDim * 2)
        let vVec = try makeEmptyBuffer(bytes: kvHeads * headDim * 2)
        let scores = try makeEmptyBuffer(bytes: numHeads * seqLen * 4)
        let probs = try makeEmptyBuffer(bytes: numHeads * seqLen * 4)
        let attnOut = try makeEmptyBuffer(bytes: numHeads * headDim * 2)
        let out = try makeEmptyBuffer(bytes: hidden * 2)

        func quantMatvec(
            _ encoder: MTLComputeCommandEncoder,
            _ triplet: (q: Int, scales: Int, biases: Int),
            input: MTLBuffer, outDim: Int, inDim: Int, output: MTLBuffer
        ) throws {
            try quant.encodeMatvec(
                into: encoder, q: model.weights.buffer, qByteOffset: triplet.q,
                scales: model.weights.buffer, scalesByteOffset: triplet.scales,
                biases: model.weights.buffer, biasesByteOffset: triplet.biases,
                input: input, outDim: outDim, inDim: inDim, output: output)
        }

        var gpuOutput: [Float] = []
        for position in 0..<seqLen {
            let row = Array(
                referenceInput[(position * hidden)..<((position + 1) * hidden)])
            let halves = row.map(Float16.init)
            halves.withUnsafeBytes {
                input.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
            try context.timedDispatch { encoder in
                try quantMatvec(
                    encoder, qProj, input: input,
                    outDim: numHeads * headDim, inDim: hidden, output: qRaw)
                try quantMatvec(
                    encoder, kProj, input: input,
                    outDim: kvHeads * headDim, inDim: hidden, output: kRaw)
                try quantMatvec(
                    encoder, vProj, input: input,
                    outDim: kvHeads * headDim, inDim: hidden, output: vVec)
                try kernels.encodeRMSNorm(
                    into: encoder, input: qRaw, weight: model.weights.buffer,
                    weightByteOffset: try normOffset("q_norm.weight"),
                    rows: numHeads, dim: headDim, eps: eps, output: qVec)
                try kernels.encodeRMSNorm(
                    into: encoder, input: kRaw, weight: model.weights.buffer,
                    weightByteOffset: try normOffset("k_norm.weight"),
                    rows: kvHeads, dim: headDim, eps: eps, output: kVec)
                try kernels.encodeRoPE(
                    into: encoder, vector: qVec, cosTable: cosTable,
                    sinTable: sinTable, position: position, positions: seqLen,
                    heads: numHeads, headDim: headDim)
                try kernels.encodeRoPE(
                    into: encoder, vector: kVec, cosTable: cosTable,
                    sinTable: sinTable, position: position, positions: seqLen,
                    heads: kvHeads, headDim: headDim)
                try attention.encodeKVAppend(
                    into: encoder, cache: cache, layer: 0, component: .key,
                    position: position, vector: kVec)
                try attention.encodeKVAppend(
                    into: encoder, cache: cache, layer: 0, component: .value,
                    position: position, vector: vVec)
                try attention.encodeAttentionScores(
                    into: encoder, cache: cache, layer: 0, position: position,
                    query: qVec, numHeads: numHeads, scores: scores)
                try attention.encodeSoftmaxRows(
                    into: encoder, input: scores, rows: numHeads,
                    count: position + 1, rowStride: seqLen, output: probs)
                try attention.encodeAttentionPV(
                    into: encoder, cache: cache, layer: 0, position: position,
                    probs: probs, numHeads: numHeads, output: attnOut)
                try quantMatvec(
                    encoder, oProj, input: attnOut,
                    outDim: hidden, inDim: numHeads * headDim, output: out)
            }
            gpuOutput += readF16(out, count: hidden)
        }
        assertWithinPhase2Gate(
            gpuOutput, reference,
            rel: exp2(-7), slice: "layer0_attn_output (CPU-quant live)")
    }
}

// MARK: - Tier E: the full 28-layer GPU-quant stack vs live CPU-quant

final class GPUQuantTierETests: XCTestCase {

    override func setUpWithError() throws {
        try SharedQuantGPUModel.skipUnlessReady()
    }

    /// Tier-E gate 2⁻⁵ on both full-stack slices: the pinned prompt's 5
    /// tokens through the wired packed pipeline (computeLogits at every
    /// position so the final-norm hook point is populated per row), oracle
    /// slices computed live from the CPU-quant reference.
    func testLayerStackAndFinalNormMatchCPUQuant() throws {
        let model = try SharedQuantGPUModel.model()
        let cpu = try SharedQuantModel.model()
        model.reset()
        let ids = try SharedCheckpoint.promptFixture("short_english").inputIds
        var lastLayer: [Float] = []
        var finalNorm: [Float] = []
        for token in ids {
            _ = try model.step(token: token, computeLogits: true)
            lastLayer += model.lastLayerOutput()
            finalNorm += model.finalNormOutput()
        }
        model.reset()
        let referenceStates = try cpu.hiddenStates(ids: ids)
        let referenceNormed = try cpu.finalNorm(referenceStates)
        assertWithinPhase2Gate(
            lastLayer, referenceStates,
            rel: exp2(-5), slice: "last_layer_output (CPU-quant live)")
        assertWithinPhase2Gate(
            finalNorm, referenceNormed,
            rel: exp2(-5), slice: "final_norm_output (CPU-quant live)")
    }
}

/// P3-5 Tier-E teacher-forced logit suite: the GPULogitSuiteTests structure
/// with the oracle swapped to the LIVE CPU-quant reference (spec D5 — no
/// dumped fixtures exist for the quantized fork). Teacher-forcing uses the
/// committed reference fp32 argmax sequences (identical prefixes at every
/// step, the P1-5 premise); every quantity previously read from the manifest
/// (checkpoints, fingerprints, top-64, top-1/top-2/margin) is computed from
/// the CPU-quant logits instead. Phase 2 constants reused verbatim
/// (DECISIONS.md 2026-08-25 Phase 3 gates — never loosen):
///
///   full-vocab checkpoints (steps {0,1,24,49})   |Δ| ≤ 2⁻⁵·M_step
///   per-step top-64 at CPU-quant indices          |Δ| ≤ 2⁻⁵·M64
///   per-step fingerprints (float64)               lse/mean ≤ 2⁻⁵·M64, std ≤ 2⁻⁴·M64
///   tie-aware top-1, all 250 steps                ε_tie = 2⁻⁴·M64, margins from
///                                                 the CPU-quant logits
///
/// Heavy suite (250 CPU-quant full re-forwards): run release-mode, e.g.
///   swift test -c release --filter GPUQuantLogitSuiteTests
final class GPUQuantLogitSuiteTests: XCTestCase {

    // MARK: - Pre-committed gate constants (Phase 2, reused verbatim)

    private static let relE = exp2(Float(-5))     // 2⁻⁵ (checkpoint/top-64/lse/mean)
    private static let relStd = exp2(Float(-4))   // 2⁻⁴ (std is 2-Lipschitz)
    private static let tieRel = exp2(-4.0)        // ε_tie = 2·2⁻⁵·M64 = 2⁻⁴·M64

    private static let generationSteps = 50
    private static let checkpointSteps: Set<Int> = [0, 1, 24, 49]
    private static let topK = 64

    override func setUpWithError() throws {
        try SharedQuantGPUModel.skipUnlessReady()
    }

    private struct StepsFile: Decodable {
        let prompt_id: String
        let argmax_token_ids: [Int]
    }

    private func referenceArgmax(_ prompt: String) throws -> [Int] {
        let url = SharedCheckpoint.fixturesDir
            .appendingPathComponent("prompts/\(prompt)/steps.json")
        let file = try JSONDecoder().decode(StepsFile.self, from: Data(contentsOf: url))
        XCTAssertEqual(file.prompt_id, prompt)
        return file.argmax_token_ids
    }

    /// Indices of the `count` largest logits, ties broken by lower index
    /// (matches Argmax.firstIndex at rank 0).
    private func topIndices(_ logits: [Float], count: Int) -> [Int] {
        let sorted = logits.indices.sorted {
            logits[$0] != logits[$1] ? logits[$0] > logits[$1] : $0 < $1
        }
        return Array(sorted.prefix(count))
    }

    // MARK: - The suite

    private func runSuite(_ prompt: String) throws {
        let gpu = try SharedQuantGPUModel.model()
        let cpu = try SharedQuantModel.model()
        gpu.reset()
        let vocab = gpu.config.vocabSize
        let argmax = try referenceArgmax(prompt)
        XCTAssertEqual(argmax.count, Self.generationSteps)

        var ids = try SharedCheckpoint.promptFixture(prompt).inputIds
        for step in 0..<Self.generationSteps {
            let cpuLogits = try cpu.lastPositionLogits(ids: ids)
            let gpuLogits = try gpu.lastPositionLogits(ids: ids)
            XCTAssertEqual(gpuLogits.count, vocab)
            XCTAssertEqual(cpuLogits.count, vocab)

            // Per-step top-64 of the CPU-quant oracle; M64 scales every gate.
            let top64 = topIndices(cpuLogits, count: Self.topK)
            var m64: Float = 0
            for index in top64 { m64 = max(m64, abs(cpuLogits[index])) }

            // Full-vocab checkpoints at 2⁻⁵ · M_step.
            if Self.checkpointSteps.contains(step) {
                var mStep: Float = 0
                for value in cpuLogits { mStep = max(mStep, abs(value)) }
                let gate = Self.relE * mStep
                var worst: Float = 0
                var worstIndex = -1
                for i in 0..<vocab {
                    let diff = abs(gpuLogits[i] - cpuLogits[i])
                    if diff > worst { worst = diff; worstIndex = i }
                }
                XCTAssertLessThanOrEqual(
                    worst, gate,
                    "\(prompt) step \(step): full-vocab worst |Δ| \(worst) at "
                    + "token \(worstIndex) exceeds 2⁻⁵·M_step = \(gate)")
            }

            // Every-step scalar fingerprints (float64, both sides live).
            let gpuFingerprint = logitFingerprints(gpuLogits)
            let cpuFingerprint = logitFingerprints(cpuLogits)
            let scalarGate = Double(Self.relE * m64)
            XCTAssertLessThanOrEqual(
                abs(gpuFingerprint.logsumexp - cpuFingerprint.logsumexp), scalarGate,
                "\(prompt) step \(step): logsumexp \(gpuFingerprint.logsumexp) vs "
                + "CPU-quant \(cpuFingerprint.logsumexp) (gate 2⁻⁵·M64 = \(scalarGate))")
            XCTAssertLessThanOrEqual(
                abs(gpuFingerprint.mean - cpuFingerprint.mean), scalarGate,
                "\(prompt) step \(step): mean \(gpuFingerprint.mean) vs "
                + "CPU-quant \(cpuFingerprint.mean) (gate 2⁻⁵·M64 = \(scalarGate))")
            XCTAssertLessThanOrEqual(
                abs(gpuFingerprint.std - cpuFingerprint.std),
                Double(Self.relStd * m64),
                "\(prompt) step \(step): std \(gpuFingerprint.std) vs "
                + "CPU-quant \(cpuFingerprint.std) (gate 2⁻⁴·M64)")

            // Every-step top-64: our logits at the CPU-quant indices.
            let topGate = Self.relE * m64
            var worstTop: Float = 0
            var worstRank = -1
            for (rank, index) in top64.enumerated() {
                let diff = abs(gpuLogits[index] - cpuLogits[index])
                if diff > worstTop { worstTop = diff; worstRank = rank }
            }
            XCTAssertLessThanOrEqual(
                worstTop, topGate,
                "\(prompt) step \(step): top-64 worst |Δ| \(worstTop) at rank "
                + "\(worstRank) exceeds 2⁻⁵·M64 = \(topGate)")

            // Tie-aware top-1 agreement at ε_tie = 2⁻⁴·M64, margins from the
            // CPU-quant logits.
            let top1 = top64[0]
            let top2 = top64[1]
            let margin = Double(cpuLogits[top1]) - Double(cpuLogits[top2])
            let ours = Argmax.firstIndex(gpuLogits)
            XCTAssertTrue(
                tieAwareArgmaxHolds(
                    ours: ours, top1: top1, top2: top2, margin: margin,
                    epsilon: Self.tieRel * Double(m64)),
                "\(prompt) step \(step): our argmax \(ours) vs CPU-quant top-1 "
                + "\(top1) / top-2 \(top2) (margin \(margin), "
                + "ε \(Self.tieRel * Double(m64)))")

            // Teacher-force the REFERENCE fp32 argmax token (identical
            // prefixes at every step — the P1-5 premise; the P3-3 band suite
            // teacher-forces the same sequences).
            ids.append(argmax[step])
        }
        gpu.reset()
    }

    func testGPUQuantLogitMatchShortEnglish() throws { try runSuite("short_english") }
    func testGPUQuantLogitMatchMultiSentence() throws { try runSuite("multi_sentence") }
    func testGPUQuantLogitMatchCodeSnippet() throws { try runSuite("code_snippet") }
    func testGPUQuantLogitMatchNonASCII() throws { try runSuite("non_ascii") }
    func testGPUQuantLogitMatchChatTemplate() throws { try runSuite("chat_template") }
}
