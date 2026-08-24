import XCTest
@testable import QwenMetalEngine
import Metal

/// P2-4 unit tests for the wired GPU pipeline that need NO real checkpoint:
/// a tiny synthetic Qwen3-shaped model (1 layer, hidden 8, GQA 2/1, vocab 16)
/// exercises load validation, the incremental-prefix decode contract, the
/// context-limit stop, and DecodeLoop driving the GPU backend (spec edge
/// cases 5, 8, 10 at pipeline level). Values are exact in bf16 (k/16,
/// |k| ≤ 7) so the checkpoint bytes are deterministic.
final class GPUModelTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("GPUModelTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Tiny synthetic checkpoint (Qwen3-shaped, bf16-exact values)

    private static let tinyConfigJSON = """
    {
      "attention_bias": false,
      "eos_token_id": 15,
      "head_dim": 4,
      "hidden_size": 8,
      "intermediate_size": 8,
      "max_position_embeddings": 64,
      "model_type": "qwen3",
      "num_attention_heads": 2,
      "num_hidden_layers": 1,
      "num_key_value_heads": 1,
      "rms_norm_eps": 1e-06,
      "rope_theta": 10000,
      "tie_word_embeddings": true,
      "vocab_size": 16
    }
    """

    private static let tinyTensors: [(name: String, shape: [Int])] = [
        ("model.embed_tokens.weight", [16, 8]),
        ("model.norm.weight", [8]),
        ("model.layers.0.input_layernorm.weight", [8]),
        ("model.layers.0.self_attn.q_proj.weight", [8, 8]),
        ("model.layers.0.self_attn.k_proj.weight", [4, 8]),
        ("model.layers.0.self_attn.v_proj.weight", [4, 8]),
        ("model.layers.0.self_attn.o_proj.weight", [8, 8]),
        ("model.layers.0.self_attn.q_norm.weight", [4]),
        ("model.layers.0.self_attn.k_norm.weight", [4]),
        ("model.layers.0.post_attention_layernorm.weight", [8]),
        ("model.layers.0.mlp.gate_proj.weight", [8, 8]),
        ("model.layers.0.mlp.up_proj.weight", [8, 8]),
        ("model.layers.0.mlp.down_proj.weight", [8, 8]),
    ]

    /// Deterministic per-tensor pattern, exact in both bf16 and fp16.
    private func value(_ tensorIndex: Int, _ elementIndex: Int) -> Float {
        Float((elementIndex * 7 + tensorIndex * 13) % 15 - 7) / 16
    }

    /// Builds the synthetic single-file checkpoint. `shapeOverride` /
    /// `dtypeOverride` swap one tensor's declared shape or dtype to exercise
    /// the load-time validation paths.
    private func writeTinyCheckpoint(
        shapeOverride: [String: [Int]] = [:],
        dtypeOverride: [String: String] = [:]
    ) throws -> String {
        var entries: [String] = []
        var payload = Data()
        for (index, tensor) in Self.tinyTensors.enumerated() {
            let shape = shapeOverride[tensor.name] ?? tensor.shape
            let dtype = dtypeOverride[tensor.name] ?? "BF16"
            let count = shape.reduce(1, *)
            let start = payload.count
            for element in 0..<count {
                let v = value(index, element)
                let halfword: UInt16 = dtype == "F16"
                    ? Float16(v).bitPattern
                    : UInt16(truncatingIfNeeded: v.bitPattern >> 16)
                payload.append(UInt8(halfword & 0xFF))
                payload.append(UInt8(halfword >> 8))
            }
            let shapeJSON = shape.map(String.init).joined(separator: ",")
            entries.append(
                "\"\(tensor.name)\":{\"dtype\":\"\(dtype)\",\"shape\":[\(shapeJSON)],"
                + "\"data_offsets\":[\(start),\(payload.count)]}")
        }
        let header = Data("{\(entries.joined(separator: ","))}".utf8)
        var blob = Data()
        var headerLength = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &headerLength) { blob.append(contentsOf: $0) }
        blob.append(header)
        blob.append(payload)
        let url = tempDir.appendingPathComponent("model.safetensors")
        try blob.write(to: url)
        return url.path
    }

    private func tinyConfig(json: String = tinyConfigJSON) throws -> ModelConfig {
        try ModelConfig(jsonData: json.data(using: .utf8)!)
    }

    private func makeContextOrSkip() throws -> MetalContext {
        do {
            return try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    private func makeTinyModel(maxContext: Int = 16) throws -> GPUModel {
        let context = try makeContextOrSkip()
        let checkpoint = try SafetensorsFile(path: writeTinyCheckpoint())
        return try GPUModel(
            checkpoint: checkpoint, config: try tinyConfig(), context: context,
            maxContext: maxContext)
    }

    // MARK: - Load-time validation

    func testRejectsNonQwen3Family() throws {
        let context = try makeContextOrSkip()
        let checkpoint = try SafetensorsFile(path: writeTinyCheckpoint())
        let biasedJSON = Self.tinyConfigJSON.replacingOccurrences(
            of: "\"attention_bias\": false", with: "\"attention_bias\": true")
        XCTAssertThrowsError(
            try GPUModel(
                checkpoint: checkpoint, config: try tinyConfig(json: biasedJSON),
                context: context, maxContext: 16)
        ) { error in
            guard case ModelError.unsupportedFamily = error else {
                return XCTFail("expected unsupportedFamily, got \(error)")
            }
        }
    }

    func testRejectsNonBF16WeightDtype() throws {
        let context = try makeContextOrSkip()
        let path = try writeTinyCheckpoint(
            dtypeOverride: ["model.layers.0.self_attn.q_norm.weight": "F16"])
        let checkpoint = try SafetensorsFile(path: path)
        XCTAssertThrowsError(
            try GPUModel(
                checkpoint: checkpoint, config: try tinyConfig(),
                context: context, maxContext: 16)
        ) { error in
            guard case ModelError.badWeightDtype(let tensor, let expected, let actual) = error else {
                return XCTFail("expected badWeightDtype, got \(error)")
            }
            XCTAssertEqual(tensor, "model.layers.0.self_attn.q_norm.weight")
            XCTAssertEqual(expected, "BF16")
            XCTAssertEqual(actual, "F16")
        }
    }

    func testRejectsWrongWeightShape() throws {
        let context = try makeContextOrSkip()
        let path = try writeTinyCheckpoint(
            shapeOverride: ["model.embed_tokens.weight": [16, 4]])
        let checkpoint = try SafetensorsFile(path: path)
        XCTAssertThrowsError(
            try GPUModel(
                checkpoint: checkpoint, config: try tinyConfig(),
                context: context, maxContext: 16)
        ) { error in
            guard case ModelError.badWeightShape(let tensor, let expected, let actual) = error else {
                return XCTFail("expected badWeightShape, got \(error)")
            }
            XCTAssertEqual(tensor, "model.embed_tokens.weight")
            XCTAssertEqual(expected, [16, 8])
            XCTAssertEqual(actual, [16, 4])
        }
    }

    // MARK: - Incremental-prefix decode contract

    /// Incremental extension must be BITWISE identical to a from-scratch
    /// replay: same kernels, same dispatch order, deterministic GPU floats.
    func testIncrementalMatchesFreshReplayBitwise() throws {
        let model = try makeTinyModel()
        _ = try model.lastPositionLogits(ids: [1, 2, 3])
        let incremental = try model.lastPositionLogits(ids: [1, 2, 3, 4])
        XCTAssertEqual(model.cachedTokens, [1, 2, 3, 4])

        let fresh = try makeTinyModel()
        let replay = try fresh.lastPositionLogits(ids: [1, 2, 3, 4])
        XCTAssertEqual(
            incremental, replay,
            "incremental decode must be bitwise identical to a full replay")
    }

    func testPrefixMismatchResetsAndReplays() throws {
        let model = try makeTinyModel()
        _ = try model.lastPositionLogits(ids: [1, 2, 3])
        let afterReset = try model.lastPositionLogits(ids: [1, 5, 6])
        XCTAssertEqual(model.cachedTokens, [1, 5, 6])

        let fresh = try makeTinyModel()
        XCTAssertEqual(afterReset, try fresh.lastPositionLogits(ids: [1, 5, 6]))
    }

    func testRepeatedIdenticalIdsRecomputeIdentically() throws {
        let model = try makeTinyModel()
        let first = try model.lastPositionLogits(ids: [3, 1, 4])
        // Same ids again is NOT a strict extension — the cache resets and
        // replays; the result must still be bitwise identical.
        let second = try model.lastPositionLogits(ids: [3, 1, 4])
        XCTAssertEqual(first, second)
    }

    func testEmptyIdsThrows() throws {
        let model = try makeTinyModel()
        XCTAssertThrowsError(try model.lastPositionLogits(ids: [])) { error in
            guard case ModelError.badInput = error else {
                return XCTFail("expected badInput, got \(error)")
            }
        }
    }

    func testTokenIdOutOfRangeThrows() throws {
        let model = try makeTinyModel()
        XCTAssertThrowsError(try model.step(token: 16, computeLogits: true)) { error in
            guard case ModelError.tokenIdOutOfRange(let id, let vocab) = error else {
                return XCTFail("expected tokenIdOutOfRange, got \(error)")
            }
            XCTAssertEqual(id, 16)
            XCTAssertEqual(vocab, 16)
        }
    }

    // MARK: - CPU reference agreement (sanity on the synthetic model)

    /// Wiring sanity vs the oracle-validated CPU stack, using the committed
    /// full-stack species bound (2⁻⁵·M, floor 2⁻¹¹ — DECISIONS.md Phase 2
    /// gates). The binding gates run against the real checkpoint in the
    /// fixture/logit suites; this catches gross wiring bugs without it.
    func testAgreesWithCPUReferenceOnSyntheticModel() throws {
        let gpu = try makeTinyModel()
        let checkpoint = try SafetensorsFile(path: writeTinyCheckpoint())
        let cpu = try QwenModel(
            checkpoint: checkpoint, config: try tinyConfig(), maxSequenceLength: 16)

        let ids = [1, 2, 3, 4, 5]
        let cpuLogits = try cpu.lastPositionLogits(ids: ids)
        let gpuLogits = try gpu.lastPositionLogits(ids: ids)
        XCTAssertEqual(gpuLogits.count, cpuLogits.count)

        let m = cpuLogits.map(abs).max() ?? 0
        let tolerance = max(exp2(-5) * m, exp2(-11))
        for i in 0..<cpuLogits.count {
            XCTAssertLessThanOrEqual(
                abs(gpuLogits[i] - cpuLogits[i]), tolerance,
                "logit \(i): GPU \(gpuLogits[i]) vs CPU \(cpuLogits[i])")
        }
    }

    // MARK: - Context limit (spec edge case 5, pipeline level)

    func testStepAtMaxContextThrowsContextFull() throws {
        let model = try makeTinyModel(maxContext: 4)
        for token in [1, 2, 3, 4] {
            try model.step(token: token, computeLogits: false)
        }
        XCTAssertThrowsError(try model.step(token: 5, computeLogits: true)) { error in
            XCTAssertEqual(
                error as? KVCacheError,
                .contextFull(position: 4, maxContext: 4))
        }
    }

    // MARK: - DecodeLoop over the GPU backend (spec edge case 8)

    /// The EOS-1 stop semantics against the REAL GPU backend: whatever token
    /// greedy decode emits first, putting it in the stop set must stop
    /// generation right after it.
    func testDecodeLoopStopsOnEOSWithGPUBackend() throws {
        let model = try makeTinyModel()
        let loop = DecodeLoop(model: model, maxContext: 16)
        let unstopped = try loop.generate(
            promptIds: [1, 2], maxNewTokens: 4, eosTokenIds: [])
        XCTAssertEqual(unstopped.count, 4)

        model.reset()
        let stopped = try loop.generate(
            promptIds: [1, 2], maxNewTokens: 4, eosTokenIds: [unstopped[0]])
        XCTAssertEqual(
            stopped, [unstopped[0]],
            "decode must stop right after emitting a stop-set token")
    }

    func testDecodeLoopStopsAtContextLimitWithGPUBackend() throws {
        let model = try makeTinyModel(maxContext: 6)
        let generated = try DecodeLoop(model: model, maxContext: 6).generate(
            promptIds: [1, 2, 3], maxNewTokens: 10, eosTokenIds: [])
        XCTAssertEqual(
            generated.count, 3,
            "prompt 3 + generated 3 fills the 6-token context — clean stop, "
            + "no contextFull throw")
    }
}
