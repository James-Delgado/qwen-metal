import XCTest
@testable import QwenMetalEngine
import Metal

/// P5-2 M-sweep microbench harness tests (spec edge test 13): the rate /
/// FLOP / fraction arithmetic verified on synthetic timings, the sweep
/// report shape pinned, byte accounting shared with the P3-6 bench, and a
/// tiny-model end-to-end run for dual-timing sanity. Real-artifact pins
/// ride in QuantGemmMicrobenchRealArtifactTests (skip cleanly when the
/// local-only artifact is absent).
final class QuantGemmMicrobenchTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("QuantGemmMicrobenchTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    private func path(_ name: String) -> String {
        tempDir.appendingPathComponent(name).path
    }

    // MARK: - Tiny Qwen3-shaped packed checkpoint (QuantMatvecMicrobenchTests
    // fixture, one layer: hidden 64, headDim 16, 4/2 heads, inter 128, vocab 32)

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

    private func tinyConfig() throws -> ModelConfig {
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
          "tie_word_embeddings": true,
          "vocab_size": 32
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

    private func makeTinyMicrobench() throws -> QuantGemmMicrobench {
        let source = try makeSourceFile(tensors: tinyTensors())
        let out = path("packed.safetensors")
        _ = try Q4Packer.pack(inputPath: source, outputPath: out)
        return try QuantGemmMicrobench(
            packed: try PackedCheckpoint(path: out), config: try tinyConfig(),
            context: try makeContextOrSkip())
    }

    // MARK: - FLOP accounting (edge test 13: pure arithmetic, no artifact)

    /// 2·M·Σ(N·K) hand-computed for the tiny roster: q/o 64·64, k/v 32·64,
    /// gate/up 128·64, down 64·128, lm_head 32·64 ⇒ Σ = 38,912 elements.
    func testTotalFlopsHandComputedOnTinyConfig() throws {
        let config = try tinyConfig()
        XCTAssertEqual(
            QuantGemmMicrobench.totalFlops(config: config, m: 1), 2 * 38_912)
        XCTAssertEqual(
            QuantGemmMicrobench.totalFlops(config: config, m: 4), 2 * 4 * 38_912)
    }

    /// THE compute-accounting pin at the pinned Qwen3-1.7B dims: the sweep's
    /// weight elements are totalPackedBytes ÷ 0.5625 = 1,720,451,072, so
    /// M=8 performs exactly 27,527,217,152 FLOPs per iteration.
    func testTotalFlopsPinnedAtRealDims() throws {
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        XCTAssertEqual(
            QuantGemmMicrobench.totalFlops(config: config, m: 8),
            27_527_217_152)
        // Consistency with the byte protocol the two benches share.
        XCTAssertEqual(
            QuantGemmMicrobench.totalFlops(config: config, m: 1),
            Double(QuantMatvecMicrobench.totalPackedBytes(config: config))
                / 0.5625 * 2)
    }

    // MARK: - Rate arithmetic + fraction math on synthetic timings

    /// GB/s = bytes ÷ GPU s ÷ 1e9 and GFLOPS = FLOPs ÷ GPU s ÷ 1e9 on
    /// hand-built timings (power-of-two durations — exact in binary FP);
    /// the M=8 fraction gate arithmetic (rate ≥ 0.70 × 43.84) is checked
    /// against the committed 30.69 threshold on both sides.
    func testSweepPointRateArithmeticOnSyntheticTimings() {
        func timing(gpuSeconds: Double) -> DispatchTiming {
            DispatchTiming(
                wallStart: 0, wallEnd: gpuSeconds + 0.25,
                gpuStart: 0, gpuEnd: gpuSeconds)
        }
        let point = QuantGemmSweepPoint(
            m: 8,
            totalPackedBytes: 1_000_000_000,
            totalFlops: 500_000_000,
            dispatchesPerIteration: 197,
            warmupTimings: [timing(gpuSeconds: 9)],
            measuredTimings: [
                timing(gpuSeconds: 0.5),   // 2 GB/s, 1 GFLOPS
                timing(gpuSeconds: 0.25),  // 4 GB/s, 2 GFLOPS
                timing(gpuSeconds: 0.125), // 8 GB/s, 4 GFLOPS
            ],
            spotCheckSite: "s", spotCheckMaxAbsDelta: 0, spotCheckTolerance: 1)
        XCTAssertEqual(point.effectiveGBps, [2, 4, 8])
        XCTAssertEqual(point.measuredGFlops, [1, 2, 4])
        XCTAssertEqual(point.medianGBps, 4)
        XCTAssertEqual(point.bestGBps, 8)
        XCTAssertEqual(point.minGBps, 2)
        XCTAssertEqual(point.maxGBps, 8)
        XCTAssertEqual(point.medianGFlops, 2)
        XCTAssertEqual(point.bestGFlops, 4)
        // Warmup iterations never contribute to any figure.
        XCTAssertFalse(point.effectiveGBps.contains(1.0 / 9))

        // Fraction gate arithmetic: the committed threshold is 0.70 × 43.84
        // = 30.688 GB/s; a rate just under fails the comparison, just over
        // passes (the gate itself binds on-device only — this pins the math
        // a P5-5 walk will do).
        let threshold = 0.70 * 43.84
        XCTAssertEqual(threshold, 30.688, accuracy: 1e-12)
        XCTAssertFalse(30.68 >= threshold)
        XCTAssertTrue(30.69 >= threshold)
    }

    // MARK: - Report shape (edge test 13)

    func testExportTextReportShapeCarriesAllSweepFields() {
        func timing(gpuSeconds: Double) -> DispatchTiming {
            DispatchTiming(
                wallStart: 0, wallEnd: gpuSeconds + 0.25,
                gpuStart: 0, gpuEnd: gpuSeconds)
        }
        func point(m: Int) -> QuantGemmSweepPoint {
            QuantGemmSweepPoint(
                m: m, totalPackedBytes: 967_753_728,
                totalFlops: Double(m) * 2 * 1_720_451_072,
                dispatchesPerIteration: 197,
                warmupTimings: [timing(gpuSeconds: 1)],
                measuredTimings: [timing(gpuSeconds: 0.5)],
                spotCheckSite: "site @ M=\(m)",
                spotCheckMaxAbsDelta: 0.001, spotCheckTolerance: 0.01)
        }
        let result = QuantGemmMicrobenchResult(
            points: [point(m: 8), point(m: 64), point(m: 512)])
        let text = result.exportText(
            dateStamp: "2026-09-15", deviceLabel: "TestDevice",
            osVersion: "os 1.0", batteryHealthNote: "85%",
            coldOrWarmNote: "warm", residency: .mmap)

        XCTAssertTrue(text.contains("2026-09-15"))
        XCTAssertTrue(text.contains("TestDevice"))
        XCTAssertTrue(text.contains("residency mmap"))
        XCTAssertTrue(text.contains("battery health: 85%"))
        XCTAssertTrue(text.contains("cold/warm: warm"))
        XCTAssertTrue(text.contains("packed bytes/iteration: 967753728"))
        // One section per M, in run order.
        for m in [8, 64, 512] {
            XCTAssertTrue(text.contains("M = \(m):"), "missing M=\(m) section")
            XCTAssertTrue(text.contains("site @ M=\(m)"), "missing spot check @ M=\(m)")
        }
        XCTAssertTrue(text.contains("GFLOPS"), "compute denominator missing")
        XCTAssertTrue(text.contains("30.7 GB/s"),
                      "gate reminder must name the on-device bar")
        XCTAssertTrue(text.contains("GFLOPS are reported, never gated"))
    }

    // MARK: - Tiny-model end-to-end run (dual timing sanity, per-M points)

    func testTinySweepRunTimingSanityCountsAndAccounting() throws {
        let bench = try makeTinyMicrobench()
        let result = try bench.run(
            mValues: [1, 4], warmupIterations: 1, measuredIterations: 2)

        XCTAssertEqual(result.points.map(\.m), [1, 4])
        let config = try tinyConfig()
        for point in result.points {
            // Dispatch count is MEASURED: 1 layer × 7 + lm_head.
            XCTAssertEqual(point.dispatchesPerIteration, 8)
            XCTAssertEqual(point.warmupTimings.count, 1)
            XCTAssertEqual(point.measuredTimings.count, 2)
            // Byte protocol identical to the matvec bench (shared roster).
            XCTAssertEqual(
                point.totalPackedBytes,
                QuantMatvecMicrobench.totalPackedBytes(config: config))
            XCTAssertEqual(
                point.totalFlops,
                QuantGemmMicrobench.totalFlops(config: config, m: point.m))
            // Dual timing sanity (hard rule 7) on every iteration.
            for timing in point.warmupTimings + point.measuredTimings {
                XCTAssertGreaterThan(timing.gpuDuration, 0)
                XCTAssertGreaterThanOrEqual(
                    timing.wallDuration, timing.gpuDuration)
            }
            for rate in point.effectiveGBps + point.measuredGFlops {
                XCTAssertTrue(rate.isFinite)
                XCTAssertGreaterThan(rate, 0)
            }
            // Spot check passed and recorded a real gate, labeled with M.
            XCTAssertEqual(
                point.spotCheckSite,
                "model.layers.0.self_attn.q_proj.weight @ M=\(point.m)")
            XCTAssertGreaterThan(point.spotCheckTolerance, 0)
            XCTAssertLessThanOrEqual(
                point.spotCheckMaxAbsDelta, point.spotCheckTolerance)
        }
    }

    // MARK: - Validation edges

    func testInvalidIterationsAndMValuesThrow() throws {
        let bench = try makeTinyMicrobench()
        XCTAssertThrowsError(
            try bench.run(mValues: [8], warmupIterations: 0, measuredIterations: 0))
        XCTAssertThrowsError(
            try bench.run(mValues: [8], warmupIterations: -1, measuredIterations: 1))
        XCTAssertThrowsError(try bench.run(mValues: []))
        XCTAssertThrowsError(try bench.run(mValues: [8, 0]))
        XCTAssertThrowsError(try bench.run(mValues: [-1]))
    }
}

/// Real-artifact pins (skip cleanly when the local-only packed artifact is
/// absent): the shared 197-dispatch / 967,753,728-byte protocol holds on
/// the real file through the GEMM path at the gate's M=8 point.
final class QuantGemmMicrobenchRealArtifactTests: XCTestCase {

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

    func testRealArtifactM8PointProtocolPins() throws {
        let packed = try PackedCheckpoint(
            path: SharedQuantModel.packedURL.path,
            expectedRevision: SharedCheckpoint.pinnedRevision)
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        let bench = try QuantGemmMicrobench(
            packed: packed, config: config, context: try MetalContext())
        let result = try bench.run(
            mValues: [8], warmupIterations: 1, measuredIterations: 2)

        let point = try XCTUnwrap(result.points.first)
        XCTAssertEqual(point.totalPackedBytes, 967_753_728)
        XCTAssertEqual(point.dispatchesPerIteration, 197)
        XCTAssertEqual(point.totalFlops, 27_527_217_152)
        for timing in point.measuredTimings {
            XCTAssertGreaterThan(timing.gpuDuration, 0)
            XCTAssertGreaterThanOrEqual(timing.wallDuration, timing.gpuDuration)
        }
        XCTAssertLessThanOrEqual(
            point.spotCheckMaxAbsDelta, point.spotCheckTolerance)
    }
}
