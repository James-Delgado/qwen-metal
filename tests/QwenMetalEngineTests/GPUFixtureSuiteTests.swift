import XCTest
@testable import QwenMetalEngine
import Metal

/// Shared GPU pipeline for the P2-4 fixture-gate suites: ONE `GPUModel` over
/// the local consolidated checkpoint (mmap residency), shared across the
/// Tier-M/E fixture tests, the GPU logit suite, and the free-run report —
/// mirroring `SharedCheckpoint`'s one-instance pattern. maxContext 256 covers
/// the activation slices (seq 5), the logit suite's longest teacher-forced
/// sequence, and the 128-step free run.
enum SharedGPUModel {
    static let maxContext = 256

    private static var sharedContext: Result<MetalContext, Error>?
    private static var sharedModel: Result<GPUModel, Error>?

    static func metalContext() throws -> MetalContext {
        if sharedContext == nil {
            sharedContext = Result { try MetalContext() }
        }
        return try sharedContext!.get()
    }

    static func model() throws -> GPUModel {
        if sharedModel == nil {
            sharedModel = Result {
                let checkpoint = try SafetensorsFile(
                    path: SharedCheckpoint.checkpointURL.path)
                guard checkpoint.metadata["source_revision"]
                    == SharedCheckpoint.pinnedRevision else {
                    throw SharedCheckpoint.HelperError(description:
                        "checkpoint has source_revision "
                        + "'\(checkpoint.metadata["source_revision"] ?? "<missing>")', "
                        + "expected the pinned \(SharedCheckpoint.pinnedRevision)")
                }
                let config = try ModelConfig(
                    jsonData: SharedCheckpoint.pinnedConfigJSON.data(using: .utf8)!)
                return try GPUModel(
                    checkpoint: checkpoint, config: config,
                    context: try metalContext(), maxContext: maxContext)
            }
        }
        return try sharedModel!.get()
    }

    /// Call from setUpWithError: skips without the local checkpoint or
    /// without a Metal device (spec edge case 10 — clean skip, not a crash).
    static func skipUnlessReady() throws {
        try SharedCheckpoint.skipUnlessCheckpointPresent()
        do {
            _ = try metalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }
}

// MARK: - Shared gate assertion (pre-committed Phase 2 relative species)

/// |Δ| ≤ max(rel · max|ref|, 2⁻¹¹) per element — the Phase 2 fp16 gate shape
/// (DECISIONS.md 2026-08-23 "Phase 2 gates pre-committed"; never loosens).
func assertWithinPhase2Gate(
    _ actual: [Float], _ reference: [Float], rel: Float, slice: String,
    file: StaticString = #filePath, line: UInt = #line
) {
    XCTAssertEqual(actual.count, reference.count,
                   "\(slice): element count", file: file, line: line)
    var maxAbsReference: Float = 0
    for r in reference { maxAbsReference = max(maxAbsReference, abs(r)) }
    let tolerance = max(rel * maxAbsReference, exp2(-11))
    var worst: Float = 0
    var worstIndex = -1
    for i in 0..<min(actual.count, reference.count) {
        let diff = abs(actual[i] - reference[i])
        if diff > worst { worst = diff; worstIndex = i }
    }
    XCTAssertLessThanOrEqual(
        worst, tolerance,
        "\(slice): worst |Δ| = \(worst) at index \(worstIndex) exceeds the "
        + "pre-committed gate \(tolerance) (rel \(rel) · max|ref| \(maxAbsReference))",
        file: file, line: line)
}

// MARK: - Tier M: isolated GPU modules vs the P1 activation fixtures

/// Each surface is fed its REFERENCE input slice (rounded to fp16, spec D2)
/// and diffed against its reference output at the pre-committed Tier-M gate.
/// The kernels are driven directly (the P2-2/P2-3 public encode API); the
/// full-pipeline wiring is covered by the Tier-E suite below.
final class GPUTierMFixtureTests: XCTestCase {

    override func setUpWithError() throws {
        try SharedGPUModel.skipUnlessReady()
    }

    private func fixture(_ name: String) throws -> [Float] {
        try SharedCheckpoint.floats(
            at: "activations/\(name).bin", expectedCount: 5 * 2048)
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

    /// Tier-M EXACT gate: GPU embedding fp16 == fp16(ref fp32) bitwise.
    func testEmbeddingsMatchFixtureExactlyInFP16() throws {
        let model = try SharedGPUModel.model()
        let context = try SharedGPUModel.metalContext()
        let kernels = try DecodeKernels(context: context)
        let tableOffset = try model.weights.byteOffset(for: "model.embed_tokens.weight")
        let hidden = model.config.hiddenSize
        let reference = try fixture("embeddings_output")
        let out = try makeEmptyBuffer(bytes: hidden * 2)

        for (row, token) in try promptIDs().enumerated() {
            try context.timedDispatch { encoder in
                try kernels.encodeEmbeddingLookup(
                    into: encoder, table: model.weights.buffer,
                    tableByteOffset: tableOffset,
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
                        "embedding row \(row) element \(i): GPU "
                        + "\(gpuHalves[i]) != fp16(ref) \(expected) "
                        + "(exact bitwise match required — Phase 2 gates)")
                }
            }
        }
    }

    /// Tier-M gate 2⁻⁸: layer-0 input RMSNorm fed its reference input.
    func testInputLayerNormMatchesFixture() throws {
        let model = try SharedGPUModel.model()
        let context = try SharedGPUModel.metalContext()
        let kernels = try DecodeKernels(context: context)
        let hidden = model.config.hiddenSize
        let input = try makeF16Buffer(try fixture("embeddings_output"))
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
            readF16(out, count: 5 * hidden),
            try fixture("layer0_pre_attn_norm_output"),
            rel: exp2(-8), slice: "layer0_pre_attn_norm_output")
    }

    /// Tier-M gate 2⁻⁷: the layer-0 attention module fed its reference input
    /// slice, run position-by-position over a fresh 1-layer KV cache (the
    /// decode path's sequential-prefill form, spec D6).
    func testAttentionOutputMatchesFixture() throws {
        let model = try SharedGPUModel.model()
        let context = try SharedGPUModel.metalContext()
        let kernels = try DecodeKernels(context: context)
        let attention = try AttentionKernels(context: context)
        let config = model.config
        let hidden = config.hiddenSize
        let headDim = config.headDim
        let numHeads = config.numAttentionHeads
        let kvHeads = config.numKeyValueHeads
        let eps = Float(config.rmsNormEps)
        let seqLen = 5

        func offset(_ suffix: String) throws -> Int {
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

        let referenceInput = try fixture("layer0_pre_attn_norm_output")
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

        var gpuOutput: [Float] = []
        for position in 0..<seqLen {
            let row = Array(
                referenceInput[(position * hidden)..<((position + 1) * hidden)])
            let halves = row.map(Float16.init)
            halves.withUnsafeBytes {
                input.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
            }
            try context.timedDispatch { encoder in
                try kernels.encodeMatvec(
                    into: encoder, weights: model.weights.buffer,
                    weightByteOffset: try offset("q_proj.weight"),
                    input: input, outDim: numHeads * headDim, inDim: hidden,
                    output: qRaw)
                try kernels.encodeMatvec(
                    into: encoder, weights: model.weights.buffer,
                    weightByteOffset: try offset("k_proj.weight"),
                    input: input, outDim: kvHeads * headDim, inDim: hidden,
                    output: kRaw)
                try kernels.encodeMatvec(
                    into: encoder, weights: model.weights.buffer,
                    weightByteOffset: try offset("v_proj.weight"),
                    input: input, outDim: kvHeads * headDim, inDim: hidden,
                    output: vVec)
                try kernels.encodeRMSNorm(
                    into: encoder, input: qRaw, weight: model.weights.buffer,
                    weightByteOffset: try offset("q_norm.weight"),
                    rows: numHeads, dim: headDim, eps: eps, output: qVec)
                try kernels.encodeRMSNorm(
                    into: encoder, input: kRaw, weight: model.weights.buffer,
                    weightByteOffset: try offset("k_norm.weight"),
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
                try kernels.encodeMatvec(
                    into: encoder, weights: model.weights.buffer,
                    weightByteOffset: try offset("o_proj.weight"),
                    input: attnOut, outDim: hidden, inDim: numHeads * headDim,
                    output: out)
            }
            gpuOutput += readF16(out, count: hidden)
        }
        assertWithinPhase2Gate(
            gpuOutput, try fixture("layer0_attn_output"),
            rel: exp2(-7), slice: "layer0_attn_output")
    }
}

// MARK: - Tier E: the full 28-layer GPU stack vs the P1 fixtures

final class GPUTierEFixtureTests: XCTestCase {

    override func setUpWithError() throws {
        try SharedGPUModel.skipUnlessReady()
    }

    /// Tier-E gate 2⁻⁵ on both full-stack slices: the pinned prompt's 5
    /// tokens run through the wired pipeline (computeLogits at every position
    /// so the final-norm hook point is populated per row).
    func testLayerStackAndFinalNormMatchFixtures() throws {
        let model = try SharedGPUModel.model()
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
        assertWithinPhase2Gate(
            lastLayer,
            try SharedCheckpoint.floats(
                at: "activations/last_layer_output.bin", expectedCount: 5 * 2048),
            rel: exp2(-5), slice: "last_layer_output")
        assertWithinPhase2Gate(
            finalNorm,
            try SharedCheckpoint.floats(
                at: "activations/final_norm_output.bin", expectedCount: 5 * 2048),
            rel: exp2(-5), slice: "final_norm_output")
    }
}
