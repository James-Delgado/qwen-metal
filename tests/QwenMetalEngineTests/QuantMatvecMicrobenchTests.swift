import XCTest
@testable import QwenMetalEngine
import Metal

/// P3-6 microbench harness tests (task pins: byte accounting, dual timing
/// sanity, per-shape + aggregate arithmetic). The tiny-model tests need no
/// artifact; the real-artifact byte/dispatch pins ride in
/// QuantMatvecMicrobenchRealArtifactTests below (skip cleanly when the
/// local-only artifact is absent).
final class QuantMatvecMicrobenchTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("QuantMatvecMicrobenchTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    private func path(_ name: String) -> String {
        tempDir.appendingPathComponent(name).path
    }

    // MARK: - Tiny Qwen3-shaped packed checkpoint (GPUQuantModelTests dims)

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

    private func tinyConfig(
        vocabSize: Int = 32, tieWordEmbeddings: Bool = true
    ) throws -> ModelConfig {
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

    private func makePackedCheckpoint() throws -> PackedCheckpoint {
        let source = try makeSourceFile(tensors: tinyTensors())
        let out = path("packed.safetensors")
        _ = try Q4Packer.pack(inputPath: source, outputPath: out)
        return try PackedCheckpoint(path: out)
    }

    private func makeTinyMicrobench() throws -> QuantMatvecMicrobench {
        try QuantMatvecMicrobench(
            packed: try makePackedCheckpoint(), config: try tinyConfig(),
            context: try makeContextOrSkip())
    }

    // MARK: - Byte accounting (task pin: q + scales + biases arithmetic)

    /// Hand-computed triplet sizes: q = out·(in/8)·4, scales = biases =
    /// out·(in/64)·2 ⇒ 0.5625 bytes per weight element.
    func testPackedBytesMatchesHandComputedTripletSizes() {
        // [64, 64]: q 64·8·4 = 2048, scales/biases 64·1·2 = 128 each.
        XCTAssertEqual(QuantMatvecMicrobench.packedBytes(outDim: 64, inDim: 64), 2304)
        // [32, 128]: q 32·16·4 = 2048, scales/biases 32·2·2 = 128 each.
        XCTAssertEqual(QuantMatvecMicrobench.packedBytes(outDim: 32, inDim: 128), 2304)
        // 0.5625 bytes/element at a pinned real shape.
        XCTAssertEqual(
            QuantMatvecMicrobench.packedBytes(outDim: 2048, inDim: 2048),
            Int(Double(2048 * 2048) * 0.5625))
    }

    /// THE byte-accounting pin: at the pinned Qwen3-1.7B dims the one-token
    /// sweep reads exactly 967,753,728 packed bytes (the gates entry's
    /// "q + scales + biases ≈ 0.967 GB"), across 197 dispatches. Pure
    /// arithmetic — no artifact needed.
    func testTotalPackedBytesAndSiteCountPinnedAtRealDims() throws {
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        XCTAssertEqual(
            QuantMatvecMicrobench.totalPackedBytes(config: config), 967_753_728)
        let specs = QuantMatvecMicrobench.siteSpecs(config: config)
        XCTAssertEqual(specs.map(\.count).reduce(0, +), 197)
        XCTAssertEqual(specs.map(\.role), [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head",
        ])
    }

    // MARK: - Aggregate + per-shape arithmetic (task pin)

    /// Rate arithmetic on a hand-built result: GB/s = bytes ÷ GPU seconds
    /// ÷ 1e9; median is the middle measured iteration; best is the max.
    func testAggregateRateArithmeticOnHandBuiltResult() {
        // gpuStart 0 and power-of-two durations: the GB/s arithmetic below
        // is exact in binary floating point.
        func timing(gpuSeconds: Double) -> DispatchTiming {
            DispatchTiming(
                wallStart: 0, wallEnd: gpuSeconds + 0.25,
                gpuStart: 0, gpuEnd: gpuSeconds)
        }
        let result = QuantMatvecMicrobenchResult(
            totalPackedBytes: 1_000_000_000,
            dispatchesPerIteration: 197,
            warmupTimings: [timing(gpuSeconds: 9)],
            measuredTimings: [
                timing(gpuSeconds: 0.5),   // 2 GB/s
                timing(gpuSeconds: 0.25),  // 4 GB/s
                timing(gpuSeconds: 0.125), // 8 GB/s
            ],
            shapes: [],
            spotCheckSite: "s", spotCheckMaxAbsDelta: 0, spotCheckTolerance: 1)
        XCTAssertEqual(result.measuredGBps, [2, 4, 8])
        XCTAssertEqual(result.medianGBps, 4)
        XCTAssertEqual(result.bestGBps, 8)
        XCTAssertEqual(result.minGBps, 2)
        XCTAssertEqual(result.maxGBps, 8)
        // Warmup iterations never contribute to any figure.
        XCTAssertFalse(result.measuredGBps.contains(1.0 / 9))
    }

    func testShapeRateArithmeticOnHandBuiltResult() {
        let shape = QuantMatvecShapeResult(
            role: "q_proj", outDim: 64, inDim: 64,
            dispatchesPerIteration: 28,
            packedBytesPerIteration: 500_000_000,
            warmupTimings: [],
            measuredTimings: [
                DispatchTiming(wallStart: 0, wallEnd: 1, gpuStart: 0, gpuEnd: 0.5),
                DispatchTiming(wallStart: 0, wallEnd: 1, gpuStart: 0, gpuEnd: 0.25),
            ])
        XCTAssertEqual(shape.measuredGBps, [1, 2])
        XCTAssertEqual(shape.medianGBps, 1.5)
        XCTAssertEqual(shape.bestGBps, 2)
    }

    /// The reused Tier K comparison has teeth: a violation fails, an
    /// in-gate delta passes, and a length mismatch never passes.
    func testSpotCheckDeltaTeeth() {
        let m: Float = 4.0 // tolerance = max(2⁻⁹·4, 2⁻¹¹) = 2⁻⁷ = 0.0078125
        let inGate = QuantMatvecMicrobench.spotCheckDelta(
            gpu: [m + 0.0078], reference: [m])
        XCTAssertTrue(inGate.passed)
        XCTAssertEqual(inGate.tolerance, 0.0078125)

        let violating = QuantMatvecMicrobench.spotCheckDelta(
            gpu: [m + 0.008], reference: [m])
        XCTAssertFalse(violating.passed)

        let mismatched = QuantMatvecMicrobench.spotCheckDelta(
            gpu: [m, m], reference: [m])
        XCTAssertFalse(mismatched.passed)
    }

    // MARK: - Tiny-model end-to-end run (task pin: dual timing sanity)

    func testTinyRunTimingSanityCountsAndAccounting() throws {
        let bench = try makeTinyMicrobench()
        let result = try bench.run(warmupIterations: 1, measuredIterations: 2)

        // Dispatch count is MEASURED (DispatchCounter): 1 layer × 7 + lm_head.
        XCTAssertEqual(result.dispatchesPerIteration, 8)
        XCTAssertEqual(result.warmupTimings.count, 1)
        XCTAssertEqual(result.measuredTimings.count, 2)

        // Dual timing sanity (hard rule 7): both clocks present, nonzero
        // GPU time, wall ≥ GPU — on every iteration, warmup included.
        for timing in result.warmupTimings + result.measuredTimings {
            XCTAssertGreaterThan(timing.gpuDuration, 0)
            XCTAssertGreaterThanOrEqual(timing.wallDuration, timing.gpuDuration)
        }
        for rate in result.measuredGBps {
            XCTAssertTrue(rate.isFinite)
            XCTAssertGreaterThan(rate, 0)
        }

        // Per-shape decomposition covers the aggregate exactly: 8 roles,
        // dispatch counts and packed bytes sum to the aggregate's.
        XCTAssertEqual(result.shapes.count, 8)
        XCTAssertEqual(
            result.shapes.map(\.dispatchesPerIteration).reduce(0, +),
            result.dispatchesPerIteration)
        XCTAssertEqual(
            result.shapes.map(\.packedBytesPerIteration).reduce(0, +),
            result.totalPackedBytes)
        for shape in result.shapes {
            XCTAssertEqual(shape.warmupTimings.count, 1)
            XCTAssertEqual(shape.measuredTimings.count, 2)
            for timing in shape.measuredTimings {
                XCTAssertGreaterThan(timing.gpuDuration, 0)
                XCTAssertGreaterThanOrEqual(
                    timing.wallDuration, timing.gpuDuration)
            }
        }

        // The tiny sweep's byte accounting from the tensor list above:
        // q/o [64,64], k/v [32,64], gate/up [128,64], down [64,128],
        // lm_head = tied embedding [32,64].
        let expected = [
            (64, 64), (32, 64), (32, 64), (64, 64),
            (128, 64), (128, 64), (64, 128), (32, 64),
        ].reduce(0) {
            $0 + QuantMatvecMicrobench.packedBytes(outDim: $1.0, inDim: $1.1)
        }
        XCTAssertEqual(result.totalPackedBytes, expected)

        // Spot check passed and recorded a real gate.
        XCTAssertEqual(result.spotCheckSite, "model.layers.0.self_attn.q_proj.weight")
        XCTAssertGreaterThan(result.spotCheckTolerance, 0)
        XCTAssertLessThanOrEqual(
            result.spotCheckMaxAbsDelta, result.spotCheckTolerance)
    }

    func testPerShapeCanBeSkipped() throws {
        let bench = try makeTinyMicrobench()
        let result = try bench.run(
            warmupIterations: 0, measuredIterations: 1, includePerShape: false)
        XCTAssertTrue(result.shapes.isEmpty)
        XCTAssertEqual(result.measuredTimings.count, 1)
    }

    // MARK: - Validation edges

    func testInvalidIterationsThrow() throws {
        let bench = try makeTinyMicrobench()
        XCTAssertThrowsError(
            try bench.run(warmupIterations: 0, measuredIterations: 0))
        XCTAssertThrowsError(
            try bench.run(warmupIterations: -1, measuredIterations: 1))
    }

    /// Untied config against the tied packed artifact fails loudly asking
    /// for lm_head.weight.q (the GPUModel species — never a silent
    /// substitution).
    func testUntiedConfigFailsLoudly() throws {
        let context = try makeContextOrSkip()
        XCTAssertThrowsError(
            try QuantMatvecMicrobench(
                packed: try makePackedCheckpoint(),
                config: try tinyConfig(tieWordEmbeddings: false),
                context: context)
        ) { error in
            guard case SafetensorsError.tensorNotFound(let name) = error else {
                return XCTFail("expected tensorNotFound, got \(error)")
            }
            XCTAssertEqual(name, "lm_head.weight" + Q4G64.qSuffix)
        }
    }

    /// A config whose shapes disagree with the packed dims fails at init
    /// with badWeightShape naming the tensor.
    func testWrongShapeConfigRejected() throws {
        let context = try makeContextOrSkip()
        XCTAssertThrowsError(
            try QuantMatvecMicrobench(
                packed: try makePackedCheckpoint(),
                config: try tinyConfig(vocabSize: 16),
                context: context)
        ) { error in
            guard case ModelError.badWeightShape(let tensor, let expected, let actual) = error else {
                return XCTFail("expected badWeightShape, got \(error)")
            }
            XCTAssertEqual(tensor, "model.embed_tokens.weight")
            XCTAssertEqual(expected, [16, 64])
            XCTAssertEqual(actual, [32, 64])
        }
    }

    // MARK: - Export text (shared CLI/app formatting)

    func testExportTextCarriesRowFields() throws {
        let bench = try makeTinyMicrobench()
        let result = try bench.run(warmupIterations: 0, measuredIterations: 1)
        let text = result.exportText(
            dateStamp: "2026-09-04", deviceLabel: "TestDevice",
            osVersion: "os 1.0", batteryHealthNote: "85%",
            coldOrWarmNote: "warm", residency: .mmap)
        XCTAssertTrue(text.contains("2026-09-04"))
        XCTAssertTrue(text.contains("TestDevice"))
        XCTAssertTrue(text.contains("residency mmap"))
        XCTAssertTrue(text.contains("battery health: 85%"))
        XCTAssertTrue(text.contains("cold/warm: warm"))
        XCTAssertTrue(text.contains("packed bytes/iteration: \(result.totalPackedBytes)"))
        XCTAssertTrue(text.contains("30.7 GB/s"), "gate reminder must name the on-device bar")
        XCTAssertTrue(text.contains("per-shape"))
    }
}

/// Real-artifact pins (skip cleanly when the local-only packed artifact is
/// absent): the 197-dispatch count is MEASURED and the byte accounting hits
/// the gates entry's 967,753,728 on the real file.
final class QuantMatvecMicrobenchRealArtifactTests: XCTestCase {

    override func setUpWithError() throws {
        guard FileManager.default.fileExists(
            atPath: SharedQuantModel.packedURL.path) else {
            throw XCTSkip(
                "packed artifact missing at \(SharedQuantModel.packedURL.path) "
                + "(local-only — produce it with `swift run qwen-metal-cli pack ...`)")
        }
        do {
            _ = try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    func testRealArtifactByteAccountingAndMeasuredDispatchCount() throws {
        let packed = try PackedCheckpoint(
            path: SharedQuantModel.packedURL.path,
            expectedRevision: SharedCheckpoint.pinnedRevision)
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        let bench = try QuantMatvecMicrobench(
            packed: packed, config: config, context: try MetalContext())
        let result = try bench.run(
            warmupIterations: 1, measuredIterations: 2, includePerShape: false)

        XCTAssertEqual(result.totalPackedBytes, 967_753_728)
        XCTAssertEqual(result.dispatchesPerIteration, 197)
        for timing in result.measuredTimings {
            XCTAssertGreaterThan(timing.gpuDuration, 0)
            XCTAssertGreaterThanOrEqual(timing.wallDuration, timing.gpuDuration)
        }
        XCTAssertLessThanOrEqual(
            result.spotCheckMaxAbsDelta, result.spotCheckTolerance)
    }
}
