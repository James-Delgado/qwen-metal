import XCTest
@testable import QwenMetalEngine
import Metal

/// P3-5 pipeline-wiring tests that need NO real checkpoint: a tiny
/// Qwen3-shaped bf16 source (1 layer, hidden 64 — every in-dim divides the
/// q4g64 group size) is packed with the real `Q4Packer`, loaded through
/// `PackedCheckpoint`, and driven through `GPUModel(packed:)`. Covers load
/// validation on the packed path (spec edge cases 9/10 at pipeline level),
/// the incremental-prefix contract, dispatch counting, DecodeLoop stop
/// behavior (edge 11 engine-level), and a full-stack sanity diff vs the
/// CPU-quant reference. The BINDING Tier-M/E gates run against the real
/// packed artifact in GPUQuantSuiteTests.
final class GPUQuantModelTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("GPUQuantModelTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    private func path(_ name: String) -> String {
        tempDir.appendingPathComponent(name).path
    }

    // MARK: - Tiny Qwen3-shaped source + packed checkpoint (PackedModelTests dims)

    /// bf16 bit pattern of an EXACTLY representable fp32 value.
    private func bf16Bits(_ v: Float) -> UInt16 {
        let bits = UInt16(truncatingIfNeeded: v.bitPattern >> 16)
        precondition(Float(bitPattern: UInt32(bits) << 16) == v,
                     "test value \(v) is not bf16-exact")
        return bits
    }

    private func blob(headerJSON: String, payload: [UInt8]) -> Data {
        var header = headerJSON
        while (8 + header.utf8.count) % 8 != 0 { header += " " }
        let headerData = Data(header.utf8)
        var out = Data()
        var length = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &length) { out.append(contentsOf: $0) }
        out.append(headerData)
        out.append(contentsOf: payload)
        return out
    }

    private func makeSourceFile(
        tensors: [(name: String, shape: [Int], values: [Float])]
    ) throws -> String {
        var entries: [String] = ["\"__metadata__\":{\"source_revision\":\"r1\"}"]
        var payload: [UInt8] = []
        var offset = 0
        for t in tensors {
            precondition(t.values.count == t.shape.reduce(1, *))
            let bytes = t.values.map(bf16Bits)
                .flatMap { [UInt8($0 & 0xFF), UInt8($0 >> 8)] }
            entries.append("\"\(t.name)\":{\"dtype\":\"BF16\","
                + "\"shape\":[\(t.shape.map(String.init).joined(separator: ","))],"
                + "\"data_offsets\":[\(offset),\(offset + bytes.count)]}")
            payload += bytes
            offset += bytes.count
        }
        let data = blob(headerJSON: "{" + entries.joined(separator: ",") + "}",
                        payload: payload)
        let filePath = path("source.safetensors")
        try data.write(to: URL(fileURLWithPath: filePath))
        return filePath
    }

    /// Deterministic bf16-exact values (1/64 steps, |v| ≤ 0.47), varied per
    /// tensor via `seed` so adjacent groups get distinct scales/biases.
    /// Small on purpose: the GPU path runs fp16 activations, and the ±7.5
    /// values the CPU-side PackedModelTests uses overflow fp16 through the
    /// SwiGLU/down-proj intermediates (the GPUModelTests |v| ≤ 0.44 lesson).
    private func syntheticValues(_ count: Int, seed: Int) -> [Float] {
        (0..<count).map { Float((($0 &* 37 &+ seed) % 61) - 30) * 0.015625 }
    }

    /// One-layer Qwen3-shaped tensor set: hidden 64, headDim 16, 4 heads /
    /// 2 KV heads, intermediate 128, vocab 32 — every in-dim divides 64.
    private func tinyTensors() -> [(name: String, shape: [Int], values: [Float])] {
        var seed = 1
        var tensors: [(name: String, shape: [Int], values: [Float])] = []
        func add(_ name: String, _ shape: [Int]) {
            tensors.append((name, shape, syntheticValues(shape.reduce(1, *), seed: seed)))
            seed += 7
        }
        add("model.embed_tokens.weight", [32, 64])
        add("model.layers.0.self_attn.q_proj.weight", [64, 64])
        add("model.layers.0.self_attn.k_proj.weight", [32, 64])
        add("model.layers.0.self_attn.v_proj.weight", [32, 64])
        add("model.layers.0.self_attn.o_proj.weight", [64, 64])
        add("model.layers.0.self_attn.q_norm.weight", [16])
        add("model.layers.0.self_attn.k_norm.weight", [16])
        add("model.layers.0.input_layernorm.weight", [64])
        add("model.layers.0.post_attention_layernorm.weight", [64])
        add("model.layers.0.mlp.gate_proj.weight", [128, 64])
        add("model.layers.0.mlp.up_proj.weight", [128, 64])
        add("model.layers.0.mlp.down_proj.weight", [64, 128])
        add("model.norm.weight", [64])
        return tensors
    }

    private func tinyConfig(vocabSize: Int = 32, tieWordEmbeddings: Bool = true)
        throws -> ModelConfig
    {
        try ModelConfig(jsonData: Data("""
        {
          "attention_bias": false,
          "eos_token_id": 2,
          "head_dim": 16,
          "hidden_size": 64,
          "intermediate_size": 128,
          "max_position_embeddings": 64,
          "model_type": "qwen3",
          "num_attention_heads": 4,
          "num_hidden_layers": 1,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-06,
          "rope_theta": 10000,
          "tie_word_embeddings": \(tieWordEmbeddings),
          "vocab_size": \(vocabSize)
        }
        """.utf8))
    }

    private func makeContextOrSkip() throws -> MetalContext {
        do {
            return try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    /// Packs the tiny source once per test and opens it.
    private func makePackedCheckpoint() throws -> PackedCheckpoint {
        let source = try makeSourceFile(tensors: tinyTensors())
        let out = path("packed.safetensors")
        _ = try Q4Packer.pack(inputPath: source, outputPath: out)
        return try PackedCheckpoint(path: out)
    }

    private func makeTinyPackedModel(maxContext: Int = 16) throws -> GPUModel {
        let context = try makeContextOrSkip()
        return try GPUModel(
            packed: try makePackedCheckpoint(), config: try tinyConfig(),
            context: context, maxContext: maxContext)
    }

    // MARK: - Load-time validation (packed path)

    func testPackedModelReportsQ4G64Format() throws {
        let model = try makeTinyPackedModel()
        XCTAssertEqual(model.weightsFormat, .q4g64)
    }

    /// Edge 9, GPU side: the bf16 initializer refuses a q4g64 packed file
    /// with an error naming the packed loader — never tensorNotFound noise.
    func testBF16InitializerRejectsPackedFileWithClearError() throws {
        let context = try makeContextOrSkip()
        let packed = try makePackedCheckpoint()
        XCTAssertThrowsError(
            try GPUModel(
                checkpoint: packed.file, config: try tinyConfig(),
                context: context, maxContext: 16)
        ) { error in
            guard case ModelError.badInput(let detail) = error else {
                return XCTFail("expected badInput, got \(error)")
            }
            XCTAssertTrue(detail.contains("GPUModel(packed:)"),
                          "error must name the packed loader: \(detail)")
        }
    }

    /// A config whose shapes disagree with the packed dims fails at load
    /// with badWeightShape (same species as the bf16 path).
    func testPackedModelRejectsWrongShape() throws {
        let context = try makeContextOrSkip()
        let packed = try makePackedCheckpoint()
        XCTAssertThrowsError(
            try GPUModel(
                packed: packed, config: try tinyConfig(vocabSize: 16),
                context: context, maxContext: 16)
        ) { error in
            guard case ModelError.badWeightShape(let tensor, let expected, let actual) = error else {
                return XCTFail("expected badWeightShape, got \(error)")
            }
            XCTAssertEqual(tensor, "model.embed_tokens.weight")
            XCTAssertEqual(expected, [16, 64])
            XCTAssertEqual(actual, [32, 64])
        }
    }

    /// An untied config against the tied packed artifact (which stores the
    /// tied matrix once, P3-1) fails loudly asking for lm_head.weight.q —
    /// never silently substitutes the embedding (PackedModelTests species,
    /// GPU side).
    func testPackedModelUntiedConfigFailsLoudly() throws {
        let context = try makeContextOrSkip()
        let packed = try makePackedCheckpoint()
        XCTAssertThrowsError(
            try GPUModel(
                packed: packed, config: try tinyConfig(tieWordEmbeddings: false),
                context: context, maxContext: 16)
        ) { error in
            guard case SafetensorsError.tensorNotFound(let name) = error else {
                return XCTFail("expected tensorNotFound, got \(error)")
            }
            XCTAssertEqual(name, "lm_head.weight" + Q4G64.qSuffix)
        }
    }

    // MARK: - Incremental-prefix decode contract (packed path)

    func testIncrementalMatchesFreshReplayBitwise() throws {
        let model = try makeTinyPackedModel()
        _ = try model.lastPositionLogits(ids: [1, 2, 3])
        let incremental = try model.lastPositionLogits(ids: [1, 2, 3, 4])
        XCTAssertEqual(model.cachedTokens, [1, 2, 3, 4])

        let fresh = try makeTinyPackedModel()
        let replay = try fresh.lastPositionLogits(ids: [1, 2, 3, 4])
        XCTAssertEqual(
            incremental, replay,
            "incremental packed decode must be bitwise identical to a full replay")
    }

    // MARK: - CPU-quant reference agreement (sanity on the synthetic model)

    /// Wiring sanity vs the CPU-quant reference (the Phase 3 oracle): both
    /// sides consume bit-identical dequant values (gates-entry premise), so
    /// the committed full-stack species bound (2⁻⁵·M, floor 2⁻¹¹) applies
    /// exactly as on the bf16 path. The binding gates run against the real
    /// artifact in GPUQuantSuiteTests.
    func testAgreesWithCPUQuantReferenceOnSyntheticModel() throws {
        let context = try makeContextOrSkip()
        let packed = try makePackedCheckpoint()
        let gpu = try GPUModel(
            packed: packed, config: try tinyConfig(), context: context,
            maxContext: 16)
        let cpu = try QwenModel(
            weights: packed, config: try tinyConfig(), maxSequenceLength: 16)

        let ids = [1, 2, 3, 4, 5]
        let cpuLogits = try cpu.lastPositionLogits(ids: ids)
        let gpuLogits = try gpu.lastPositionLogits(ids: ids)
        XCTAssertEqual(gpuLogits.count, cpuLogits.count)

        let m = cpuLogits.map(abs).max() ?? 0
        let tolerance = max(exp2(-5) * m, exp2(-11))
        for i in 0..<cpuLogits.count {
            XCTAssertLessThanOrEqual(
                abs(gpuLogits[i] - cpuLogits[i]), tolerance,
                "logit \(i): GPU-quant \(gpuLogits[i]) vs CPU-quant \(cpuLogits[i])")
        }
    }

    // MARK: - Instrumentation (P2-5 counter on the packed path)

    /// The packed path replaces kernels 1:1, so the measured dispatch count
    /// matches the bf16 pipeline structure exactly (1 layer → embedding 1 +
    /// layer 21 + logits tail 2 = 24; 22 without the tail) and stays stable.
    func testPackedStepDispatchCountMeasuredNonzeroAndStable() throws {
        let model = try makeTinyPackedModel()
        try model.step(token: 1, computeLogits: false)
        XCTAssertEqual(model.lastStepDispatchCount, 22)
        for token in [2, 3] {
            try model.step(token: token, computeLogits: true)
            XCTAssertEqual(model.lastStepDispatchCount, 24)
        }
        let timing = try XCTUnwrap(model.lastStepTiming)
        XCTAssertGreaterThan(timing.gpuDuration, 0)
        XCTAssertGreaterThanOrEqual(timing.wallDuration, timing.gpuDuration)
    }

    // MARK: - DecodeLoop over the packed backend (spec edge 11, engine level)

    func testDecodeLoopStopsOnEOSWithPackedBackend() throws {
        let model = try makeTinyPackedModel()
        let loop = DecodeLoop(model: model, maxContext: 16)
        let unstopped = try loop.generate(
            promptIds: [1, 2], maxNewTokens: 4, eosTokenIds: [])
        XCTAssertEqual(unstopped.count, 4)

        model.reset()
        let stopped = try loop.generate(
            promptIds: [1, 2], maxNewTokens: 4, eosTokenIds: [unstopped[0]])
        XCTAssertEqual(
            stopped, [unstopped[0]],
            "packed decode must stop right after emitting a stop-set token")
    }

    func testDecodeLoopStopsAtContextLimitWithPackedBackend() throws {
        let model = try makeTinyPackedModel(maxContext: 6)
        let generated = try DecodeLoop(model: model, maxContext: 6).generate(
            promptIds: [1, 2, 3], maxNewTokens: 10, eosTokenIds: [])
        XCTAssertEqual(
            generated.count, 3,
            "prompt 3 + generated 3 fills the 6-token context — clean stop")
    }
}
