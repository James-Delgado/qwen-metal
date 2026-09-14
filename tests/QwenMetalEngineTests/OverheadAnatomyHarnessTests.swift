import XCTest
@testable import QwenMetalEngine
import Metal

/// OA-1 (seeded by P4-9): engine-side tests for the overhead-anatomy
/// harness the app's benchmark screen exports — aggregation/formatting on
/// hand-built `OverheadAnatomy` records (no GPU needed) plus runner
/// behavior on the tiny synthetic model (arm policy, token identity with
/// the production path, error paths). The app stays thin: everything it
/// shows is formatted here.
final class OverheadAnatomyHarnessTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "OverheadAnatomyHarnessTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Hand-built records (no GPU)

    /// An anatomy record with exactly chosen span durations, built by
    /// accumulating timestamps (spans are differences, so this is exact).
    private func makeAnatomy(
        encode: Double, commit: Double, commitToStart: Double,
        gpu: Double, wakeup: Double
    ) -> OverheadAnatomy {
        let wallStart = 100.0
        let wallEncoded = wallStart + encode
        let wallCommitted = wallEncoded + commit
        let gpuStart = wallCommitted + commitToStart
        let gpuEnd = gpuStart + gpu
        return OverheadAnatomy(
            wallStart: wallStart, wallEncoded: wallEncoded,
            wallCommitted: wallCommitted,
            kernelStart: wallEncoded, kernelEnd: wallCommitted,
            gpuStart: gpuStart, gpuEnd: gpuEnd, wallEnd: gpuEnd + wakeup)
    }

    func testSpanSummariesAggregateHandBuiltRecords() throws {
        // Three records; encode spans 1/2/3 ms → median 2, min 1, max 3.
        let records = [
            makeAnatomy(encode: 0.001, commit: 0.0001, commitToStart: 0.0002,
                        gpu: 0.020, wakeup: 0.0003),
            makeAnatomy(encode: 0.002, commit: 0.0002, commitToStart: 0.0004,
                        gpu: 0.021, wakeup: 0.0006),
            makeAnatomy(encode: 0.003, commit: 0.0003, commitToStart: 0.0006,
                        gpu: 0.022, wakeup: 0.0009),
        ]
        let rows = OverheadAnatomyRunResult.spanSummaries(records)
        XCTAssertEqual(rows.map(\.label), [
            "encode", "commit call", "commit->GPU-start", "  (schedule stage)",
            "GPU execution", "wakeup (GPU->CPU)", "TOTAL wall-GPU"])

        func row(_ label: String) -> OverheadAnatomyRunResult.SpanSummary {
            rows.first { $0.label == label }!
        }
        XCTAssertEqual(row("encode").medianSeconds, 0.002, accuracy: 1e-12)
        XCTAssertEqual(row("encode").minSeconds, 0.001, accuracy: 1e-12)
        XCTAssertEqual(row("encode").maxSeconds, 0.003, accuracy: 1e-12)
        XCTAssertEqual(row("commit call").medianSeconds, 0.0002, accuracy: 1e-12)
        XCTAssertEqual(row("commit->GPU-start").medianSeconds, 0.0004,
                       accuracy: 1e-12)
        XCTAssertEqual(row("GPU execution").medianSeconds, 0.021, accuracy: 1e-12)
        XCTAssertEqual(row("wakeup (GPU->CPU)").medianSeconds, 0.0006,
                       accuracy: 1e-12)
        // TOTAL = wall − GPU = encode + commit + commit→start + wakeup.
        XCTAssertEqual(row("TOTAL wall-GPU").medianSeconds,
                       0.002 + 0.0002 + 0.0004 + 0.0006, accuracy: 1e-12)

        // Empty input never fabricates rows.
        XCTAssertTrue(OverheadAnatomyRunResult.spanSummaries([]).isEmpty)
    }

    func testExportTextCarriesTheDeviceReportFields() throws {
        let records = [
            makeAnatomy(encode: 0.001, commit: 0.0001, commitToStart: 0.0002,
                        gpu: 0.020, wakeup: 0.0003),
            makeAnatomy(encode: 0.003, commit: 0.0003, commitToStart: 0.0006,
                        gpu: 0.022, wakeup: 0.0009),
        ]
        let result = OverheadAnatomyRunResult(
            weightsFormat: .q4g64, kernelPath: .fused, promptTokenCount: 84,
            anatomies: records, unretainedAnatomies: records,
            productionOverheadSeconds: [0.0015, 0.0013, 0.0017],
            productionDispatchCount: 200, anatomyDispatchCount: 200,
            firstDecodePosition: 83, lastDecodePosition: 178,
            generatedTokenIds: [1, 2, 3])
        let text = result.exportText(
            dateStamp: "2026-09-14", deviceLabel: "iPhone16,1",
            osVersion: "iOS 26.6.1", residency: .mmap)

        XCTAssertTrue(text.contains("overhead anatomy — DIAGNOSTIC"))
        XCTAssertTrue(text.contains("date: 2026-09-14"))
        XCTAssertTrue(text.contains("device: iPhone16,1 (iOS 26.6.1)"))
        XCTAssertTrue(text.contains(
            "engine: weights q4g64, residency mmap, fused"))
        XCTAssertTrue(text.contains(
            "decode forwards: 3 production + 2 anatomy + 2 unretained"))
        XCTAssertTrue(text.contains("cache depth 83-178"))
        // Production median of [1.3, 1.5, 1.7] ms = 1.5 ms @ 200.
        XCTAssertTrue(text.contains(
            "production reference: median wall-GPU 1.5000 ms/token @ 200"))
        XCTAssertTrue(text.contains(
            "-- anatomy arm (retained references, 200 dispatches) --"))
        XCTAssertTrue(text.contains(
            "-- anatomy arm (UNRETAINED references experiment) --"))
        // Two-record encode median = (1 + 3)/2 = 2 ms.
        XCTAssertTrue(text.contains("median   2.0000 ms"))
        XCTAssertTrue(text.contains("spans telescope to wall-GPU"))
    }

    func testExportTextMarksUnstableDispatchCounts() throws {
        let result = OverheadAnatomyRunResult(
            weightsFormat: .q4g64, kernelPath: .naive, promptTokenCount: 1,
            anatomies: [], unretainedAnatomies: [],
            productionOverheadSeconds: [0.002],
            productionDispatchCount: nil, anatomyDispatchCount: nil,
            firstDecodePosition: 0, lastDecodePosition: 0,
            generatedTokenIds: [])
        let text = result.exportText(
            dateStamp: "d", deviceLabel: "dev", osVersion: "os",
            residency: .wiredCopy)
        XCTAssertTrue(text.contains("@ UNSTABLE dispatches"))
        XCTAssertTrue(text.contains("UNSTABLE dispatches) --"))
        XCTAssertTrue(text.contains("naive (pre-fusion)"))
        XCTAssertTrue(text.contains("residency wiredCopy"))
    }

    // MARK: - Tiny synthetic checkpoint (GPUModelTests fixture, verbatim)

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

    private func value(_ tensorIndex: Int, _ elementIndex: Int) -> Float {
        Float((elementIndex * 7 + tensorIndex * 13) % 15 - 7) / 16
    }

    private func writeTinyCheckpoint() throws -> String {
        var entries: [String] = []
        var payload = Data()
        for (index, tensor) in Self.tinyTensors.enumerated() {
            let count = tensor.shape.reduce(1, *)
            let start = payload.count
            for element in 0..<count {
                let halfword = UInt16(
                    truncatingIfNeeded: value(index, element).bitPattern >> 16)
                payload.append(UInt8(halfword & 0xFF))
                payload.append(UInt8(halfword >> 8))
            }
            let shapeJSON = tensor.shape.map(String.init).joined(separator: ",")
            entries.append(
                "\"\(tensor.name)\":{\"dtype\":\"BF16\",\"shape\":[\(shapeJSON)],"
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
            checkpoint: checkpoint,
            config: try ModelConfig(jsonData: Self.tinyConfigJSON.data(using: .utf8)!),
            context: context, maxContext: maxContext)
    }

    // MARK: - Runner behavior (tiny model)

    /// Seven forwards round-robin as production/anatomy/unretained =
    /// 3/2/2, the greedy stream is token-identical to a pure production
    /// decode, and both arm dispatch counts land on the tiny selecting
    /// pin (25).
    func testRunnerArmPolicyAndTokenIdentity() throws {
        let promptIds = [1, 2, 3]
        let decodeTokens = 7

        // Reference: pure production selecting decode on a separate model.
        let reference = try makeTinyModel()
        for token in promptIds.dropLast() {
            try reference.step(token: token, computeLogits: false)
        }
        var current = promptIds[promptIds.count - 1]
        var referenceTokens: [Int] = []
        for _ in 0..<decodeTokens {
            current = try reference.stepSelectingToken(token: current)
            referenceTokens.append(current)
        }

        let runner = OverheadAnatomyRunner(
            gpuModel: try makeTinyModel(), maxContext: 16)
        var steps: [Int] = []
        let result = try runner.run(
            promptIds: promptIds, decodeTokens: decodeTokens,
            onStep: { steps.append($0) })

        XCTAssertEqual(result.productionOverheadSeconds.count, 3)
        XCTAssertEqual(result.anatomies.count, 2)
        XCTAssertEqual(result.unretainedAnatomies.count, 2)
        XCTAssertEqual(result.generatedTokenIds, referenceTokens,
                       "all three arms must extend one greedy stream")
        XCTAssertEqual(result.productionDispatchCount, 25)
        XCTAssertEqual(result.anatomyDispatchCount, 25)
        XCTAssertEqual(result.promptTokenCount, 3)
        XCTAssertEqual(result.firstDecodePosition, 2)
        XCTAssertEqual(result.lastDecodePosition, 8)
        XCTAssertEqual(steps, Array(0..<decodeTokens))
        XCTAssertEqual(result.weightsFormat, .bf16)
        XCTAssertEqual(result.kernelPath, .naive)

        for anatomy in result.anatomies + result.unretainedAnatomies {
            XCTAssertGreaterThan(anatomy.gpuSeconds, 0)
            XCTAssertGreaterThanOrEqual(
                anatomy.wallSeconds, anatomy.gpuSeconds,
                "wall must bracket GPU (hard rule 7)")
        }
    }

    func testRunnerErrorPaths() throws {
        let runner = OverheadAnatomyRunner(
            gpuModel: try makeTinyModel(), maxContext: 16)
        XCTAssertThrowsError(
            try runner.run(promptIds: [], decodeTokens: 1)
        ) { XCTAssertTrue($0 is DecodeError, "\($0)") }
        XCTAssertThrowsError(
            try runner.run(promptIds: [1], decodeTokens: 0)
        ) { XCTAssertTrue($0 is DecodeError, "\($0)") }
        XCTAssertThrowsError(
            try runner.run(promptIds: [1, 2, 3], decodeTokens: 14)
        ) { XCTAssertTrue($0 is ModelError, "\($0)") }
    }

    func testRunnerStopsOnEOSAndOnShouldStop() throws {
        // EOS: learn the first selected token, then rerun with it as EOS —
        // exactly one decode forward runs.
        let probeRunner = OverheadAnatomyRunner(
            gpuModel: try makeTinyModel(), maxContext: 16)
        let probe = try probeRunner.run(promptIds: [1, 2, 3], decodeTokens: 6)
        let firstToken = probe.generatedTokenIds[0]

        let eosRunner = OverheadAnatomyRunner(
            gpuModel: try makeTinyModel(), maxContext: 16,
            eosTokenIds: [firstToken])
        let eosResult = try eosRunner.run(
            promptIds: [1, 2, 3], decodeTokens: 6)
        XCTAssertEqual(eosResult.generatedTokenIds, [firstToken])
        XCTAssertEqual(eosResult.productionOverheadSeconds.count, 1)
        XCTAssertTrue(eosResult.anatomies.isEmpty)

        // shouldStop true from the start: no forwards at all.
        let stoppedRunner = OverheadAnatomyRunner(
            gpuModel: try makeTinyModel(), maxContext: 16)
        let stopped = try stoppedRunner.run(
            promptIds: [1, 2, 3], decodeTokens: 6, shouldStop: { true })
        XCTAssertTrue(stopped.generatedTokenIds.isEmpty)
        XCTAssertTrue(stopped.productionOverheadSeconds.isEmpty)
        XCTAssertTrue(stopped.anatomies.isEmpty)
        XCTAssertTrue(stopped.unretainedAnatomies.isEmpty)
    }
}
