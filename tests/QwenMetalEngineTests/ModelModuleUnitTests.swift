import XCTest
import QwenMetalEngine

/// Synthetic unit tests for the Phase 1 model modules — small, always-run
/// checks that need no checkpoint. The oracle-diff tests against the dumped
/// HF activations live in ActivationFixtureTests (they need the local
/// consolidated checkpoint).
final class ModelModuleUnitTests: XCTestCase {

    /// Deterministic generator (SplitMix64), same reproducibility rationale
    /// as the P0B kernel tests.
    private struct SplitMix64: RandomNumberGenerator {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }

    private func randomFloats(count: Int, rng: inout SplitMix64) -> [Float] {
        (0..<count).map { _ in Float.random(in: -1...1, using: &rng) }
    }

    // MARK: - Embedding

    func testEmbeddingLookupCopiesRowsExactly() throws {
        let table: [Float] = [0, 1, 10, 11, 20, 21] // vocab 3, hidden 2
        let embedding = try Embedding(weight: table, vocabSize: 3, hiddenSize: 2)
        XCTAssertEqual(try embedding([2, 0, 2]), [20, 21, 0, 1, 20, 21])
    }

    func testEmbeddingRejectsOutOfRangeId() throws {
        let embedding = try Embedding(
            weight: [0, 1, 2, 3], vocabSize: 2, hiddenSize: 2)
        XCTAssertThrowsError(try embedding([0, 2])) { error in
            guard case ModelError.tokenIdOutOfRange(let id, let vocabSize) = error else {
                return XCTFail("expected tokenIdOutOfRange, got \(error)")
            }
            XCTAssertEqual(id, 2)
            XCTAssertEqual(vocabSize, 2)
        }
    }

    // MARK: - RMSNorm

    func testRMSNormMatchesHandComputedCase() throws {
        let x: [Float] = [1, 2, 3, 4]
        let weight: [Float] = [2, 1, 1, 0.5]
        let eps: Float = 1e-6
        let norm = RMSNorm(weight: weight, eps: eps)
        let out = try norm(x)

        // Reference computed in Double: mean(x²) = 7.5.
        let inverseRMS = 1 / (7.5 + Double(eps)).squareRoot()
        for j in 0..<4 {
            let expected = Double(weight[j]) * Double(x[j]) * inverseRMS
            XCTAssertEqual(Double(out[j]), expected, accuracy: 1e-6, "element \(j)")
        }
    }

    func testRMSNormNormalizesEachRowIndependently() throws {
        let norm = RMSNorm(weight: [1, 1], eps: 0)
        // Rows [3, 4] and [30, 40] differ only in scale, so both normalize
        // to the same direction: [3, 4]/rms(3,4) = [0.6, 0.8]·√2.
        let out = try norm([3, 4, 30, 40])
        XCTAssertEqual(out[0], out[2], accuracy: 1e-6)
        XCTAssertEqual(out[1], out[3], accuracy: 1e-6)
    }

    func testRMSNormRejectsMismatchedInputLength() {
        let norm = RMSNorm(weight: [1, 1, 1, 1], eps: 1e-6)
        XCTAssertThrowsError(try norm([1, 2, 3, 4, 5])) { error in
            guard case ModelError.badInput = error else {
                return XCTFail("expected badInput, got \(error)")
            }
        }
    }

    func testRMSNormRejectsEmptyWeightInsteadOfTrapping() {
        // dim == 0 must throw, not crash on division by zero.
        let norm = RMSNorm(weight: [], eps: 1e-6)
        XCTAssertThrowsError(try norm([1, 2])) { error in
            guard case ModelError.badInput = error else {
                return XCTFail("expected badInput, got \(error)")
            }
        }
    }

    // MARK: - RoPE

    func testRoPEPositionZeroIsUnchanged() throws {
        let rope = try RoPE(headDim: 4, theta: 10000, positions: 2)
        var rng = SplitMix64(seed: 0x0517)
        let original = randomFloats(count: 8, rng: &rng) // seq 1, heads 2
        var rotated = original
        try rope.apply(to: &rotated, seqLen: 1, heads: 2)
        // Angle 0 → cos 1, sin 0 exactly: position 0 must be bit-identical.
        XCTAssertEqual(rotated, original)
    }

    func testRoPEMatchesHandComputedRotation() throws {
        // headDim 2 → single frequency 1/theta^0 = 1, so position 1 rotates
        // by exactly 1 radian: [a, b] → [a·cos1 − b·sin1, b·cos1 + a·sin1].
        let rope = try RoPE(headDim: 2, theta: 12345, positions: 2)
        var x: [Float] = [0.25, -0.75, 0.5, 1.5] // seq 2, 1 head
        try rope.apply(to: &x, seqLen: 2, heads: 1)
        XCTAssertEqual(x[0], 0.25)
        XCTAssertEqual(x[1], -0.75)
        XCTAssertEqual(Double(x[2]), 0.5 * cos(1.0) - 1.5 * sin(1.0), accuracy: 1e-6)
        XCTAssertEqual(Double(x[3]), 1.5 * cos(1.0) + 0.5 * sin(1.0), accuracy: 1e-6)
    }

    func testRoPERejectsSequenceLongerThanTable() throws {
        let rope = try RoPE(headDim: 2, theta: 10000, positions: 2)
        var x = [Float](repeating: 0, count: 6) // seq 3 > 2 positions
        XCTAssertThrowsError(try rope.apply(to: &x, seqLen: 3, heads: 1)) { error in
            guard case ModelError.badInput = error else {
                return XCTFail("expected badInput, got \(error)")
            }
        }
    }

    // MARK: - SwiGLU MLP

    func testSwiGLUMatchesHandComputedCase() throws {
        // hidden 2, intermediate 2, seq 1. Weights are [out, in] like HF.
        let mlp = try SwiGLUMLP(
            gateWeight: [1, 0, 0, 1],  // identity
            upWeight: [1, 1, 0, 1],    // [[1,1],[0,1]]
            downWeight: [1, 0, 1, 1],  // [[1,0],[1,1]]
            hiddenSize: 2, intermediateSize: 2)
        let x: [Float] = [0.5, -1.25]
        let out = try mlp(x, seqLen: 1)

        func silu(_ v: Double) -> Double { v / (1 + exp(-v)) }
        let gate = [Double(x[0]), Double(x[1])]                    // identity
        let up = [Double(x[0]) + Double(x[1]), Double(x[1])]
        let activated = [silu(gate[0]) * up[0], silu(gate[1]) * up[1]]
        let expected = [activated[0], activated[0] + activated[1]] // down rows
        XCTAssertEqual(Double(out[0]), expected[0], accuracy: 1e-6)
        XCTAssertEqual(Double(out[1]), expected[1], accuracy: 1e-6)
    }

    // MARK: - Attention

    func testAttentionRowZeroIgnoresLaterTokens() throws {
        // Causality: position 0's output must not depend on position 1's
        // token, whatever the weights are.
        let hidden = 4, numHeads = 2, numKVHeads = 1, headDim = 2
        var rng = SplitMix64(seed: 0xA77E)
        let attention = try Attention(
            numHeads: numHeads, numKVHeads: numKVHeads,
            headDim: headDim, hiddenSize: hidden,
            qProjWeight: randomFloats(count: numHeads * headDim * hidden, rng: &rng),
            kProjWeight: randomFloats(count: numKVHeads * headDim * hidden, rng: &rng),
            vProjWeight: randomFloats(count: numKVHeads * headDim * hidden, rng: &rng),
            oProjWeight: randomFloats(count: hidden * numHeads * headDim, rng: &rng),
            qNorm: RMSNorm(weight: [1, 1], eps: 1e-6),
            kNorm: RMSNorm(weight: [1, 1], eps: 1e-6),
            rope: try RoPE(headDim: headDim, theta: 10000, positions: 4))

        let sharedRow = randomFloats(count: hidden, rng: &rng)
        let outA = try attention(
            sharedRow + randomFloats(count: hidden, rng: &rng), seqLen: 2)
        let outB = try attention(
            sharedRow + randomFloats(count: hidden, rng: &rng), seqLen: 2)
        XCTAssertEqual(
            Array(outA[0..<hidden]), Array(outB[0..<hidden]),
            "row 0 changed when only the future token differed — causal mask broken")
    }

    func testAttentionGQAMapsQueryHeadGroupsToDistinctKVHeads() throws {
        // 4 Q heads over 2 KV heads: heads 0/1 must read KV head 0, heads
        // 2/3 KV head 1. With seqLen 1 the softmax weight is exactly 1, so
        // each head's context IS its KV head's V row — a wrong mapping
        // (e.g. head % groupSize) reorders the output exactly.
        let numHeads = 4, numKVHeads = 2, headDim = 2
        let hidden = numHeads * headDim // 8
        var rng = SplitMix64(seed: 0x69A5)

        // V = x·Wv^T with x = e0 picks column 0 of v_proj: make KV head 0's
        // V row [1, 2] and KV head 1's [3, 4].
        var vProj = [Float](repeating: 0, count: numKVHeads * headDim * hidden)
        for row in 0..<(numKVHeads * headDim) {
            vProj[row * hidden] = Float(row + 1)
        }
        var oProjIdentity = [Float](repeating: 0, count: hidden * hidden)
        for i in 0..<hidden {
            oProjIdentity[i * hidden + i] = 1
        }
        let attention = try Attention(
            numHeads: numHeads, numKVHeads: numKVHeads,
            headDim: headDim, hiddenSize: hidden,
            qProjWeight: randomFloats(count: numHeads * headDim * hidden, rng: &rng),
            kProjWeight: randomFloats(count: numKVHeads * headDim * hidden, rng: &rng),
            vProjWeight: vProj,
            oProjWeight: oProjIdentity,
            qNorm: RMSNorm(weight: [1, 1], eps: 1e-6),
            kNorm: RMSNorm(weight: [1, 1], eps: 1e-6),
            rope: try RoPE(headDim: headDim, theta: 10000, positions: 2))

        var x = [Float](repeating: 0, count: hidden)
        x[0] = 1 // e0
        let out = try attention(x, seqLen: 1)
        // Every step is exact at seqLen 1 (softmax of one element is 1.0).
        XCTAssertEqual(out, [1, 2, 1, 2, 3, 4, 3, 4])
    }

    func testAttentionMatchesHandComputedTwoTokenCase() throws {
        // 1 head, headDim 2, hidden 2, seq 2 — the full pipeline (projection,
        // QK-norm, RoPE, scaled causal softmax, PV, o_proj) against a
        // Double-precision oracle computed right here.
        let wq: [Float] = [0.6, -0.3, 0.2, 0.9] // [out, in]
        let wk: [Float] = [0.5, 0.1, -0.4, 0.7]
        let wv: [Float] = [1.0, 0.5, -0.25, 0.75]
        let wo: [Float] = [0.8, -0.2, 0.3, 0.4]
        let eps: Float = 1e-6
        let attention = try Attention(
            numHeads: 1, numKVHeads: 1, headDim: 2, hiddenSize: 2,
            qProjWeight: wq, kProjWeight: wk, vProjWeight: wv, oProjWeight: wo,
            qNorm: RMSNorm(weight: [1, 1], eps: eps),
            kNorm: RMSNorm(weight: [1, 1], eps: eps),
            rope: try RoPE(headDim: 2, theta: 10000, positions: 4))
        let x: [Float] = [0.3, -0.2, 0.5, 0.4]
        let out = try attention(x, seqLen: 2)

        func matvec(_ w: [Float], _ v: [Double]) -> [Double] {
            [Double(w[0]) * v[0] + Double(w[1]) * v[1],
             Double(w[2]) * v[0] + Double(w[3]) * v[1]]
        }
        func rmsNormed(_ v: [Double]) -> [Double] {
            let inverse = 1 / ((v[0] * v[0] + v[1] * v[1]) / 2 + Double(eps)).squareRoot()
            return [v[0] * inverse, v[1] * inverse]
        }
        func rotated(_ v: [Double], by angle: Double) -> [Double] {
            [v[0] * cos(angle) - v[1] * sin(angle),
             v[1] * cos(angle) + v[0] * sin(angle)]
        }
        let rows: [[Double]] = [[0.3, -0.2], [0.5, 0.4]]
        // headDim 2 → single frequency 1: position s rotates by s radians.
        let q = rows.enumerated().map {
            rotated(rmsNormed(matvec(wq, $0.element)), by: Double($0.offset))
        }
        let k = rows.enumerated().map {
            rotated(rmsNormed(matvec(wk, $0.element)), by: Double($0.offset))
        }
        let v = rows.map { matvec(wv, $0) }

        let context0 = v[0] // row 0 attends only to itself
        let scale = 1 / Double(2).squareRoot()
        let score0 = (q[1][0] * k[0][0] + q[1][1] * k[0][1]) * scale
        let score1 = (q[1][0] * k[1][0] + q[1][1] * k[1][1]) * scale
        let rowMax = max(score0, score1)
        let e0 = exp(score0 - rowMax), e1 = exp(score1 - rowMax)
        let p0 = e0 / (e0 + e1), p1 = e1 / (e0 + e1)
        let context1 = [p0 * v[0][0] + p1 * v[1][0], p0 * v[0][1] + p1 * v[1][1]]
        let expected = matvec(wo, context0) + matvec(wo, context1)
        for i in 0..<4 {
            XCTAssertEqual(Double(out[i]), expected[i], accuracy: 1e-5, "element \(i)")
        }
    }

    func testAttentionRejectsMismatchedInputLength() throws {
        var rng = SplitMix64(seed: 0x515)
        let attention = try Attention(
            numHeads: 1, numKVHeads: 1, headDim: 2, hiddenSize: 2,
            qProjWeight: randomFloats(count: 4, rng: &rng),
            kProjWeight: randomFloats(count: 4, rng: &rng),
            vProjWeight: randomFloats(count: 4, rng: &rng),
            oProjWeight: randomFloats(count: 4, rng: &rng),
            qNorm: RMSNorm(weight: [1, 1], eps: 1e-6),
            kNorm: RMSNorm(weight: [1, 1], eps: 1e-6),
            rope: try RoPE(headDim: 2, theta: 10000, positions: 4))
        XCTAssertThrowsError(try attention([1, 2, 3], seqLen: 1)) { error in
            guard case ModelError.badInput = error else {
                return XCTFail("expected badInput, got \(error)")
            }
        }
    }

    // MARK: - QwenModel loader guards

    /// Minimal valid safetensors file with zero tensors: 8-byte little-endian
    /// header length + "{}".
    private func emptyCheckpoint() throws -> SafetensorsFile {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-metal-model-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let header = "{}".data(using: .utf8)!
        var data = withUnsafeBytes(of: UInt64(header.count).littleEndian) { Data($0) }
        data.append(header)
        let path = dir.appendingPathComponent("model.safetensors").path
        try data.write(to: URL(fileURLWithPath: path))
        return try SafetensorsFile(path: path)
    }

    private func config(modelType: String) throws -> ModelConfig {
        let json = """
        {"model_type": "\(modelType)", "hidden_size": 4, "num_hidden_layers": 1,
         "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
         "intermediate_size": 4, "rms_norm_eps": 1e-6, "rope_theta": 10000,
         "vocab_size": 8, "tie_word_embeddings": true,
         "max_position_embeddings": 8}
        """
        return try ModelConfig(jsonData: json.data(using: .utf8)!)
    }

    func testLoaderRejectsNonQwen3Family() throws {
        // qwen2 resolves to attentionBias=true / usesQKNorm=false — exactly
        // one family is implemented (PLAN.md non-goals), so this must refuse.
        let checkpoint = try emptyCheckpoint()
        XCTAssertThrowsError(
            try QwenModel(checkpoint: checkpoint, config: config(modelType: "qwen2"))
        ) { error in
            guard case ModelError.unsupportedFamily = error else {
                return XCTFail("expected unsupportedFamily, got \(error)")
            }
        }
    }

    func testLoaderFailsLoudlyOnMissingTensor() throws {
        let checkpoint = try emptyCheckpoint()
        XCTAssertThrowsError(
            try QwenModel(checkpoint: checkpoint, config: config(modelType: "qwen3"))
        ) { error in
            guard case SafetensorsError.tensorNotFound(let name) = error else {
                return XCTFail("expected tensorNotFound, got \(error)")
            }
            XCTAssertEqual(name, "model.embed_tokens.weight")
        }
    }
}
