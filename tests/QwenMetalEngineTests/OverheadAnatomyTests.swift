import XCTest
@testable import QwenMetalEngine
import Metal

/// P4-9 (phase-4.md 2026-09-12 addendum): overhead-anatomy instrumentation
/// tests. The always-run parts pin the new timing fields' sanity (wall ≥
/// GPU, spans non-negative, spans telescope to wall−GPU) and that the
/// diagnostic step leaves the production path untouched (P4-1 invariance
/// precedent — production fields cleared, production pins unaffected,
/// token choice identical). The opt-in sweep (QWEN_OVERHEAD_ANATOMY=1,
/// QWEN_DISPATCH_DIAG precedent) decodes on the real packed artifact and
/// prints the median span table the DECISIONS.md P4-9 entry records —
/// Mac PROVISIONAL diagnosis inputs, never benchmark rows.
final class OverheadAnatomyTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("OverheadAnatomyTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Span math on hand-built timestamps (no GPU needed)

    /// Synthetic timestamps with known gaps: every derived span is the
    /// expected difference and the four non-GPU spans telescope to
    /// wall − GPU exactly.
    func testSpansOnHandBuiltTimestamps() throws {
        let anatomy = OverheadAnatomy(
            wallStart: 10.000, wallEncoded: 10.003, wallCommitted: 10.0035,
            kernelStart: 10.001, kernelEnd: 10.004,
            gpuStart: 10.005, gpuEnd: 10.045, wallEnd: 10.046)

        XCTAssertEqual(anatomy.encodeSeconds, 0.003, accuracy: 1e-12)
        XCTAssertEqual(anatomy.commitSeconds, 0.0005, accuracy: 1e-12)
        XCTAssertEqual(anatomy.commitToGPUStartSeconds, 0.0015, accuracy: 1e-12)
        XCTAssertEqual(anatomy.gpuSeconds, 0.040, accuracy: 1e-12)
        XCTAssertEqual(anatomy.wakeupSeconds, 0.001, accuracy: 1e-12)
        XCTAssertEqual(anatomy.wallSeconds, 0.046, accuracy: 1e-12)
        XCTAssertEqual(anatomy.overheadSeconds, 0.006, accuracy: 1e-12)
        XCTAssertEqual(anatomy.scheduleStageSeconds, 0.003, accuracy: 1e-12)

        // Telescoping identity: the anatomy accounts for ALL of wall − GPU.
        XCTAssertEqual(
            anatomy.encodeSeconds + anatomy.commitSeconds
                + anatomy.commitToGPUStartSeconds + anatomy.wakeupSeconds,
            anatomy.overheadSeconds, accuracy: 1e-12)

        // The production-shaped view carries the same four corners.
        XCTAssertEqual(anatomy.timing.wallDuration, anatomy.wallSeconds)
        XCTAssertEqual(anatomy.timing.gpuDuration, anatomy.gpuSeconds)
        XCTAssertEqual(anatomy.timing.dispatchOverhead, anatomy.overheadSeconds,
                       accuracy: 1e-12)
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

    // MARK: - Measured anatomy sanity (tiny model, always runs)

    /// A real anatomy step produces sane spans: wall ≥ GPU (hard rule 7),
    /// every span non-negative, and the four non-GPU spans telescope to
    /// wall − GPU.
    func testMeasuredAnatomySpansAreSane() throws {
        let model = try makeTinyModel()
        let result = try model.anatomyStepSelectingToken(token: 1)
        let anatomy = result.anatomy

        XCTAssertGreaterThan(anatomy.gpuSeconds, 0)
        XCTAssertGreaterThanOrEqual(anatomy.wallSeconds, anatomy.gpuSeconds,
                                    "wall must bracket GPU (hard rule 7)")
        XCTAssertGreaterThanOrEqual(anatomy.encodeSeconds, 0)
        XCTAssertGreaterThanOrEqual(anatomy.commitSeconds, 0)
        XCTAssertGreaterThanOrEqual(anatomy.commitToGPUStartSeconds, 0,
                                    "GPU cannot start before commit() returned")
        XCTAssertGreaterThanOrEqual(anatomy.wakeupSeconds, 0,
                                    "wakeup is sampled after GPU end")
        XCTAssertGreaterThanOrEqual(anatomy.scheduleStageSeconds, 0)
        XCTAssertEqual(
            anatomy.encodeSeconds + anatomy.commitSeconds
                + anatomy.commitToGPUStartSeconds + anatomy.wakeupSeconds,
            anatomy.overheadSeconds, accuracy: 1e-9,
            "anatomy spans must account for all of wall − GPU")
    }

    /// The unretained-references experiment variant obeys the same sanity
    /// contract (it changes retention, never structure).
    func testUnretainedVariantSpansAreSane() throws {
        let model = try makeTinyModel()
        let result = try model.anatomyStepSelectingToken(
            token: 1, unretainedReferences: true)
        XCTAssertGreaterThan(result.anatomy.gpuSeconds, 0)
        XCTAssertGreaterThanOrEqual(
            result.anatomy.wallSeconds, result.anatomy.gpuSeconds)
        XCTAssertEqual(result.dispatchCount, 25)
    }

    // MARK: - Production invariance (P4-1 precedent)

    /// The anatomy step selects exactly the production step's token (same
    /// encode ⇒ same logits ⇒ same argmax), advances the cache identically,
    /// measures the production dispatch count in its own return — and leaves
    /// the production fields cleared, with a following production step still
    /// measuring its pinned count.
    func testAnatomyStepMatchesProductionAndLeavesItUntouched() throws {
        let productionModel = try makeTinyModel()
        let anatomyModel = try makeTinyModel()

        var productionToken = 1
        var anatomyToken = 1
        for step in 0..<4 {
            productionToken = try productionModel.stepSelectingToken(
                token: productionToken)
            let result = try anatomyModel.anatomyStepSelectingToken(
                token: anatomyToken)
            anatomyToken = result.token
            XCTAssertEqual(anatomyToken, productionToken,
                           "step \(step): anatomy must select the production token")
            // Tiny bf16 naive selecting step measures 25 (GPUArgmaxDecodeTests
            // pin); the anatomy step encodes the identical dispatch sequence.
            XCTAssertEqual(result.dispatchCount, 25)
            // Diagnostic steps never populate production timing fields.
            XCTAssertNil(anatomyModel.lastStepTiming)
            XCTAssertNil(anatomyModel.lastStepDispatchCount)
        }
        XCTAssertEqual(anatomyModel.cachedTokens, productionModel.cachedTokens,
                       "both step kinds must advance the cache identically")

        // Production after diagnostics: same pinned count, timing populated.
        _ = try anatomyModel.stepSelectingToken(token: anatomyToken)
        XCTAssertEqual(anatomyModel.lastStepDispatchCount, 25)
        XCTAssertNotNil(anatomyModel.lastStepTiming)
    }

    // MARK: - The opt-in anatomy sweep (real packed artifact)

    private static let packedURL = SharedCheckpoint.modelsDir
        .appendingPathComponent("qwen3-1.7b-70d244cc-q4g64.safetensors")

    private struct SpanStats {
        let label: String
        let medianMs: Double
        let minMs: Double
        let maxMs: Double
    }

    private func stats(_ label: String, _ seconds: [Double]) -> SpanStats {
        let sorted = seconds.sorted()
        return SpanStats(
            label: label,
            medianMs: sorted[sorted.count / 2] * 1000,
            minMs: sorted.first! * 1000,
            maxMs: sorted.last! * 1000)
    }

    private func printTable(_ header: String, _ rows: [SpanStats]) {
        print(header)
        for r in rows {
            print(String(
                format: "  %@ median %8.4f ms   min %8.4f   max %8.4f",
                r.label.padding(toLength: 22, withPad: " ", startingAt: 0),
                r.medianMs, r.minMs, r.maxMs))
        }
    }

    /// Set QWEN_OVERHEAD_ANATOMY=1 to run (release build — encode is host
    /// code, debug numbers would be inflated). Free-runs the real packed
    /// artifact on the production fused path, round-robin per token:
    /// production stepSelectingToken / anatomy step / anatomy step with
    /// unretained references — so the three modes sample the same session
    /// interleaved. Prints the span table the DECISIONS.md P4-9 entry
    /// records; asserts only sanity (numbers are findings, not gates).
    func testOverheadAnatomySweep() throws {
        guard ProcessInfo.processInfo.environment["QWEN_OVERHEAD_ANATOMY"] == "1" else {
            throw XCTSkip(
                "overhead-anatomy sweep is opt-in: set QWEN_OVERHEAD_ANATOMY=1 "
                + "(Mac PROVISIONAL diagnosis inputs, never benchmark rows)")
        }
        guard FileManager.default.fileExists(atPath: Self.packedURL.path) else {
            throw XCTSkip(
                "packed artifact not present at \(Self.packedURL.path) "
                + "(local-only — produce it with `qwen-metal-cli pack`)")
        }
        let context = try makeContextOrSkip()
        // A private model instance: the sweep free-runs the cache forward,
        // which must never perturb the shared Tier-suite model.
        let packed = try PackedCheckpoint(
            path: Self.packedURL.path,
            expectedRevision: SharedCheckpoint.pinnedRevision)
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        let model = try GPUModel(
            packed: packed, config: config, context: context, maxContext: 256)
        // Second operating point for the per-span affine split (the P4-5
        // two-point fit, per component): the naive 591-dispatch structure
        // on the same checkpoint, interleaved in the same session. Its own
        // token stream — timing arm only, token choices may differ.
        let naiveModel = try GPUModel(
            packed: packed, config: config, context: context, maxContext: 256,
            kernelPath: .naive)

        let warmup = 16
        let rounds = 60  // per mode; 4 modes round-robin ⇒ 240 measured steps

        var token = 100
        var naiveToken = 100
        for _ in 0..<warmup {
            token = try model.stepSelectingToken(token: token)
            naiveToken = try naiveModel.stepSelectingToken(token: naiveToken)
        }

        var productionOverhead: [Double] = []
        var anatomies: [OverheadAnatomy] = []
        var unretainedAnatomies: [OverheadAnatomy] = []
        var naiveAnatomies: [OverheadAnatomy] = []
        var naiveDispatchCount = 0
        for _ in 0..<rounds {
            token = try model.stepSelectingToken(token: token)
            let timing = try XCTUnwrap(model.lastStepTiming)
            XCTAssertEqual(model.lastStepDispatchCount, 200,
                           "fused real-dims selecting step pin (P4-8)")
            productionOverhead.append(timing.dispatchOverhead)

            let plain = try model.anatomyStepSelectingToken(token: token)
            XCTAssertEqual(plain.dispatchCount, 200)
            token = plain.token
            anatomies.append(plain.anatomy)

            let unretained = try model.anatomyStepSelectingToken(
                token: token, unretainedReferences: true)
            XCTAssertEqual(unretained.dispatchCount, 200)
            token = unretained.token
            unretainedAnatomies.append(unretained.anatomy)

            let naive = try naiveModel.anatomyStepSelectingToken(token: naiveToken)
            naiveToken = naive.token
            naiveDispatchCount = naive.dispatchCount
            naiveAnatomies.append(naive.anatomy)
        }

        func spanRows(_ records: [OverheadAnatomy]) -> [SpanStats] {
            [stats("encode", records.map(\.encodeSeconds)),
             stats("commit call", records.map(\.commitSeconds)),
             stats("commit->GPU-start", records.map(\.commitToGPUStartSeconds)),
             stats("  (schedule stage)", records.map(\.scheduleStageSeconds)),
             stats("GPU execution", records.map(\.gpuSeconds)),
             stats("wakeup (GPU->CPU)", records.map(\.wakeupSeconds)),
             stats("TOTAL wall-GPU", records.map(\.overheadSeconds))]
        }

        print("=== P4-9 overhead anatomy (Mac PROVISIONAL, fused q4g64, "
              + "200 dispatches/token, \(rounds) steps/mode interleaved) ===")
        printTable("-- production stepSelectingToken (reference) --",
                   [stats("wall-GPU overhead", productionOverhead)])
        printTable("-- anatomy step (retained references) --",
                   spanRows(anatomies))
        printTable("-- anatomy step (UNRETAINED references) --",
                   spanRows(unretainedAnatomies))
        printTable("-- anatomy step (NAIVE path, \(naiveDispatchCount) "
                   + "dispatches/token) --",
                   spanRows(naiveAnatomies))

        for anatomy in anatomies + unretainedAnatomies + naiveAnatomies {
            XCTAssertGreaterThan(anatomy.gpuSeconds, 0)
            XCTAssertGreaterThanOrEqual(anatomy.wallSeconds, anatomy.gpuSeconds)
        }
    }
}
