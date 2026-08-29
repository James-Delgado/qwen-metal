import XCTest
import Foundation
import QwenMetalEngine

/// P3-2: the CPU-quant reference front end — packed q4g64 file → fp32
/// dequant materialization through the FROZEN CPU model (phase-3.md D3).
/// Everything numeric here is exact by construction (single-rounded dequant,
/// bit-identical pass-through upcast), so all assertions are `==` — no new
/// tolerance constants (the Phase 3 gates entry's premise).
final class PackedModelTests: XCTestCase {

    private var tmpDir: URL!

    override func setUpWithError() throws {
        tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("PackedModelTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tmpDir)
    }

    private func path(_ name: String) -> String {
        tmpDir.appendingPathComponent(name).path
    }

    // MARK: - Tiny Qwen3-shaped checkpoint builders (Q4PackerTests pattern)

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
        tensors: [(name: String, shape: [Int], values: [Float])],
        fileName: String = "source.safetensors"
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
        let filePath = path(fileName)
        try data.write(to: URL(fileURLWithPath: filePath))
        return filePath
    }

    /// Deterministic bf16-exact values (quarter steps, |v| ≤ 7.5), varied
    /// per tensor via `seed` so adjacent groups get distinct scales/biases.
    private func syntheticValues(_ count: Int, seed: Int) -> [Float] {
        (0..<count).map { Float((($0 &* 37 &+ seed) % 61) - 30) * 0.25 }
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

    private func tinyConfig(hiddenSize: Int = 64, tieWordEmbeddings: Bool = true)
        throws -> ModelConfig
    {
        try ModelConfig(jsonData: Data("""
        {
          "attention_bias": false,
          "eos_token_id": 2,
          "head_dim": 16,
          "hidden_size": \(hiddenSize),
          "intermediate_size": 128,
          "max_position_embeddings": 8,
          "model_type": "qwen3",
          "num_attention_heads": 4,
          "num_hidden_layers": 1,
          "num_key_value_heads": 2,
          "rms_norm_eps": 1e-06,
          "rope_theta": 10000,
          "tie_word_embeddings": \(tieWordEmbeddings),
          "vocab_size": 32
        }
        """.utf8))
    }

    /// Packs `tensors`, returning the opened validating reader.
    private func packedCheckpoint(
        tensors: [(name: String, shape: [Int], values: [Float])]
    ) throws -> PackedCheckpoint {
        let source = try makeSourceFile(tensors: tensors)
        let packedPath = path("packed.safetensors")
        _ = try Q4Packer.pack(inputPath: source, outputPath: packedPath)
        return try PackedCheckpoint(path: packedPath)
    }

    /// Independent recomputation of the expected dequant through the pinned
    /// group primitives (P3-1 pattern) — catches bulk-loop indexing bugs.
    private func expectedDequant(_ values: [Float], tensor: String) throws -> [Float] {
        var out = [Float]()
        out.reserveCapacity(values.count)
        var start = 0
        while start < values.count {
            let g = try Q4G64.packGroup(
                values[start..<(start + Q4G64.groupSize)],
                tensor: tensor, elementOffset: start)
            for code in g.codes {
                out.append(Q4G64.dequant(code: UInt32(code), scale: g.scale, bias: g.bias))
            }
            start += Q4G64.groupSize
        }
        return out
    }

    // MARK: - Exactness through the front end

    func testPackedModelWeightsAreBitExactVsSharedArithmetic() throws {
        let tensors = tinyTensors()
        let packed = try packedCheckpoint(tensors: tensors)
        func sourceValues(_ name: String) -> [Float] {
            tensors.first { $0.name == name }!.values
        }

        // Matrix path, straight through the WeightSource front end.
        let qProjName = "model.layers.0.self_attn.q_proj.weight"
        XCTAssertEqual(
            try packed.fp32Tensor(qProjName, shape: [64, 64]),
            try expectedDequant(sourceValues(qProjName), tensor: qProjName))

        // Whole-model assembly: dequantized matrices and bit-identical
        // pass-through norms land in the frozen modules unchanged.
        let model = try QwenModel(
            weights: packed, config: tinyConfig(), maxSequenceLength: 8)
        XCTAssertEqual(
            model.embedding.weight,
            try expectedDequant(sourceValues("model.embed_tokens.weight"),
                                tensor: "model.embed_tokens.weight"))
        // Pass-through norms must equal the bf16 path's exact upcast of the
        // SOURCE bytes (they are never quantized — schema D1).
        XCTAssertEqual(model.finalNorm.weight, sourceValues("model.norm.weight"))
        // Tied lm_head maps to the (dequantized) embedding triplet.
        XCTAssertEqual(model.lmHeadWeight, model.embedding.weight)
    }

    func testTinyPackedModelForwardProducesFiniteLogits() throws {
        let model = try QwenModel(
            weights: packedCheckpoint(tensors: tinyTensors()),
            config: tinyConfig(), maxSequenceLength: 8)
        let logits = try model.lastPositionLogits(ids: [0, 3, 7])
        XCTAssertEqual(logits.count, 32)
        XCTAssertTrue(logits.allSatisfy(\.isFinite),
                      "quantized CPU reference produced non-finite logits")
    }

    // MARK: - Front-end rejects (loud, never silent)

    func testFP32TensorRejectsShapeMismatch() throws {
        let packed = try packedCheckpoint(tensors: tinyTensors())
        XCTAssertThrowsError(
            try packed.fp32Tensor("model.embed_tokens.weight", shape: [32, 128])
        ) { error in
            guard case ModelError.badWeightShape(_, let expected, let actual) = error else {
                return XCTFail("expected badWeightShape, got \(error)")
            }
            XCTAssertEqual(expected, [32, 128])
            XCTAssertEqual(actual, [32, 64])
        }
        XCTAssertThrowsError(
            try packed.fp32Tensor("model.norm.weight", shape: [65])
        ) { error in
            guard case ModelError.badWeightShape = error else {
                return XCTFail("expected badWeightShape, got \(error)")
            }
        }
    }

    func testModelLoadRejectsConfigDimensionMismatch() throws {
        // hidden_size 128 against a hidden-64 packed file: the embedding is
        // requested as [32, 128] and must fail loudly at load, not crash or
        // mis-assemble downstream.
        XCTAssertThrowsError(
            try QwenModel(
                weights: packedCheckpoint(tensors: tinyTensors()),
                config: tinyConfig(hiddenSize: 128), maxSequenceLength: 8)
        ) { error in
            guard case ModelError.badWeightShape = error else {
                return XCTFail("expected badWeightShape, got \(error)")
            }
        }
    }

    func testUntiedConfigAgainstTiedArtifactFailsLoudly() throws {
        // The source materializes lm_head byte-identical to the embedding,
        // so the packer stores the tied matrix once (P3-1). An untied config
        // then asks for a tensor the artifact deliberately omits — that must
        // surface as the parser's not-found error, never a silent substitute.
        var tensors = tinyTensors()
        tensors.append(("lm_head.weight", [32, 64],
                        tensors.first { $0.name == "model.embed_tokens.weight" }!.values))
        XCTAssertThrowsError(
            try QwenModel(
                weights: packedCheckpoint(tensors: tensors),
                config: tinyConfig(tieWordEmbeddings: false), maxSequenceLength: 8)
        ) { error in
            guard case SafetensorsError.tensorNotFound(let name) = error else {
                return XCTFail("expected tensorNotFound, got \(error)")
            }
            XCTAssertTrue(name.hasPrefix("lm_head.weight"))
        }
    }
}

/// Smoke decode on the real packed artifact (skips cleanly when the
/// local-only artifact is absent). Structural assertions only — the quality
/// band vs mlx-lm 4-bit is P3-3's pre-committed gate, not this test's.
final class PackedModelSmokeTests: XCTestCase {

    static let packedURL = SharedCheckpoint.modelsDir
        .appendingPathComponent("qwen3-1.7b-70d244cc-q4g64.safetensors")

    override func setUpWithError() throws {
        guard FileManager.default.fileExists(atPath: Self.packedURL.path) else {
            throw XCTSkip(
                "packed artifact not present at \(Self.packedURL.path) "
                + "(local-only — produce it with `qwen-metal-cli pack`)")
        }
    }

    func testSmokeDecodeThroughCPUQuantReference() async throws {
        let packed = try PackedCheckpoint(
            path: Self.packedURL.path,
            expectedRevision: SharedCheckpoint.pinnedRevision)
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))

        let materializeStart = Date()
        let model = try QwenModel(
            weights: packed, config: config, maxSequenceLength: 32)
        print(String(format:
            "PackedModelSmokeTests: fp32 materialization took %.1f s",
            Date().timeIntervalSince(materializeStart)))

        let prompt = try SharedCheckpoint.promptFixture("short_english")
        var nonFiniteStep: Int?
        let generated = try DecodeLoop(model: model, maxContext: 32).generate(
            promptIds: prompt.inputIds,
            maxNewTokens: 8,
            eosTokenIds: [151645, 151643],
            onStep: { step, logits, _ in
                if nonFiniteStep == nil, !logits.allSatisfy(\.isFinite) {
                    nonFiniteStep = step
                }
            })

        XCTAssertNil(nonFiniteStep, "non-finite logits at step \(nonFiniteStep ?? -1)")
        XCTAssertFalse(generated.isEmpty)
        XCTAssertTrue(generated.allSatisfy { (0..<config.vocabSize).contains($0) },
                      "generated ids out of vocab range: \(generated)")
        XCTAssertGreaterThan(Set(generated).count, 1,
                             "degenerate single-token output: \(generated)")

        // Coherence evidence for the session report (needs the local
        // tokenizer artifacts; ids-only smoke stands on its own without them).
        if let tokenizer = try? await TextTokenizer(
            modelFolder: SharedCheckpoint.modelsDir) {
            let text = tokenizer.decode(generated, skipSpecialTokens: true)
            print("PackedModelSmokeTests: prompt "
                + "\(prompt.inputIds) -> \(generated) \"\(text)\"")
        }
    }
}
