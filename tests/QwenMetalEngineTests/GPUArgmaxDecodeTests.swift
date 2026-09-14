import XCTest
@testable import QwenMetalEngine
import Metal

/// P4-8 pipeline wiring: `GPUModel.stepSelectingToken` / `nextGreedyToken`
/// and `DecodeLoop.generateTokens` must be TOKEN-IDENTICAL to the logits
/// path (CPU `Argmax.firstIndex` over `step(computeLogits:)` logits) — the
/// exact-equality contract at pipeline level, on the same tiny synthetic
/// bf16 model `GPUModelTests` uses. Dispatch pins are MEASURED (P2-5 rule):
/// the selecting step is the logits step + 1 (the argmax reduction).
/// The packed/fused pins live in `GPUQuantModelTests`; the real-dims pin
/// rides `FusedSDPAKernelTests`' real-artifact test.
final class GPUArgmaxDecodeTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("GPUArgmaxDecodeTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
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

    // MARK: - Exact token equality: selecting step vs logits step

    /// A free-run driven step-by-step on two identical models: model A reads
    /// full logits and applies CPU `Argmax.firstIndex`; model B selects
    /// on-GPU. Every chosen token must be identical (GPU kernels are
    /// deterministic, so the logits agree bitwise across instances).
    func testStepSelectingTokenMatchesCPUArgmaxOverLogits() throws {
        let logitsModel = try makeTinyModel()
        let selectingModel = try makeTinyModel()

        var token = 1
        for step in 0..<6 {
            let logits = try XCTUnwrap(
                try logitsModel.step(token: token, computeLogits: true))
            let cpuChoice = Argmax.firstIndex(logits)
            let gpuChoice = try selectingModel.stepSelectingToken(token: token)
            XCTAssertEqual(
                gpuChoice, cpuChoice,
                "step \(step): GPU-selected token must equal CPU argmax")
            token = cpuChoice
        }
        XCTAssertEqual(logitsModel.cachedTokens, selectingModel.cachedTokens,
                       "both paths must advance the cache identically")

        // Dual timing rides the selecting step too (hard rule 7).
        let timing = try XCTUnwrap(selectingModel.lastStepTiming)
        XCTAssertGreaterThan(timing.gpuDuration, 0)
        XCTAssertGreaterThanOrEqual(timing.wallDuration, timing.gpuDuration)
    }

    /// P2-5 rule (measured, never derived): the tiny bf16 naive pipeline
    /// measures 24 dispatches with logits; the selecting step adds exactly
    /// the argmax reduction — 25. `step(computeLogits:)`'s own pins are
    /// untouched (GPUModelTests).
    func testSelectingStepDispatchCountMeasuredPlusOne() throws {
        let model = try makeTinyModel()
        try model.step(token: 1, computeLogits: true)
        XCTAssertEqual(model.lastStepDispatchCount, 24)
        _ = try model.stepSelectingToken(token: 2)
        XCTAssertEqual(model.lastStepDispatchCount, 25)
        _ = try model.stepSelectingToken(token: 3)
        XCTAssertEqual(model.lastStepDispatchCount, 25)
    }

    // MARK: - nextGreedyToken: contract + incremental-prefix semantics

    /// `nextGreedyToken` must equal the protocol's default implementation
    /// (CPU argmax over `lastPositionLogits`) and keep `lastPositionLogits`'
    /// incremental-prefix cache contract: strict extensions reuse the cache,
    /// anything else resets and replays.
    func testNextGreedyTokenMatchesDefaultAndKeepsPrefixContract() throws {
        let reference = try makeTinyModel()
        let model = try makeTinyModel()

        let ids = [1, 2, 3]
        let expected = Argmax.firstIndex(try reference.lastPositionLogits(ids: ids))
        XCTAssertEqual(try model.nextGreedyToken(ids: ids), expected)
        XCTAssertEqual(model.cachedTokens, ids)

        // Strict extension: only the suffix runs (cache grows, no reset).
        let extended = ids + [expected]
        let expected2 = Argmax.firstIndex(
            try reference.lastPositionLogits(ids: extended))
        XCTAssertEqual(try model.nextGreedyToken(ids: extended), expected2)
        XCTAssertEqual(model.cachedTokens, extended)

        // Prefix mismatch: resets and replays from scratch, same answer as a
        // fresh model.
        let divergent = [2, 1]
        let fresh = try makeTinyModel()
        XCTAssertEqual(
            try model.nextGreedyToken(ids: divergent),
            try fresh.nextGreedyToken(ids: divergent))
        XCTAssertEqual(model.cachedTokens, divergent)

        XCTAssertThrowsError(try model.nextGreedyToken(ids: [])) { error in
            guard case ModelError.badInput = error else {
                return XCTFail("expected badInput, got \(error)")
            }
        }
    }

    // MARK: - P5-1 call span on the token path (phase-5.md D1 — edge test 12)

    /// `nextGreedyToken`'s call span sums every prompt step including the
    /// selecting step: a 3-token prompt = 2 × 22 (no logits tail) + 25 (the
    /// P4-8 selecting step's measured pin) — and wall brackets summed GPU.
    func testNextGreedyTokenCallSpanSumsPromptSteps() throws {
        let model = try makeTinyModel()
        _ = try model.nextGreedyToken(ids: [1, 2, 3])
        let span = try XCTUnwrap(model.lastCallSpan)
        XCTAssertEqual(span.stepCount, 3)
        XCTAssertEqual(span.dispatchCount, 2 * 22 + 25)
        XCTAssertGreaterThan(span.gpuSeconds, 0)
        XCTAssertGreaterThanOrEqual(
            span.wallSeconds, span.gpuSeconds,
            "span wall ≥ span GPU (edge test 12)")
    }

    /// End-to-end production wiring: the runner's GPU init captures the
    /// prompt-processing call's span as the prefill span at the first token
    /// boundary — later decode forwards (which overwrite the model's
    /// `lastCallSpan`) must not leak into it. The legacy TTFT-style field
    /// exports alongside (D1 continuity).
    func testRunnerPrefillSpanFromGPUModel() throws {
        let model = try makeTinyModel()
        let runner = BenchGenerationRunner(
            gpuModel: model, maxContext: 16, eosTokenIds: [])
        let metrics = try runner.run(promptIds: [1, 2, 3], maxNewTokens: 3).metrics

        let prefill = try XCTUnwrap(metrics.prefillSpan)
        XCTAssertEqual(prefill.promptTokenCount, 3)
        XCTAssertEqual(
            prefill.span.stepCount, 3,
            "prefill span covers the prompt call only — the first generated "
            + "token's decode forward is excluded by construction")
        XCTAssertEqual(prefill.span.dispatchCount, 2 * 22 + 25)
        XCTAssertGreaterThan(prefill.span.gpuSeconds, 0)
        XCTAssertGreaterThanOrEqual(
            prefill.span.wallSeconds, prefill.span.gpuSeconds)
        XCTAssertFalse(
            prefill.summaryLine.contains("WARM PREFIX"),
            "a cold-cache generation is a real prefill")
        XCTAssertNotNil(metrics.prefillSeconds)
    }

    // MARK: - DecodeLoop.generateTokens ≡ DecodeLoop.generate

    /// The production token-only loop and the logits-observing loop must
    /// emit identical sequences over the GPU backend — including the EOS
    /// stop, exercised by feeding the first free-run token back as EOS.
    func testGenerateTokensMatchesGenerateOnGPUBackend() throws {
        let viaLogits = try makeTinyModel()
        let viaTokens = try makeTinyModel()
        let prompt = [1, 2, 3]

        let reference = try DecodeLoop(model: viaLogits, maxContext: 16)
            .generate(promptIds: prompt, maxNewTokens: 8)
        var onTokenCalls: [(Int, Int)] = []
        let tokens = try DecodeLoop(model: viaTokens, maxContext: 16)
            .generateTokens(
                promptIds: prompt, maxNewTokens: 8,
                onToken: { onTokenCalls.append(($0, $1)) })
        XCTAssertEqual(tokens, reference,
                       "token path must be token-identical to the logits path")
        XCTAssertEqual(onTokenCalls.map(\.0), Array(0..<tokens.count))
        XCTAssertEqual(onTokenCalls.map(\.1), tokens)

        // EOS stop parity on both paths.
        viaLogits.reset()
        viaTokens.reset()
        let eos: Set<Int> = [reference[0]]
        let stoppedReference = try DecodeLoop(model: viaLogits, maxContext: 16)
            .generate(promptIds: prompt, maxNewTokens: 8, eosTokenIds: eos)
        let stoppedTokens = try DecodeLoop(model: viaTokens, maxContext: 16)
            .generateTokens(promptIds: prompt, maxNewTokens: 8, eosTokenIds: eos)
        XCTAssertEqual(stoppedTokens, stoppedReference)
        XCTAssertEqual(stoppedTokens, [reference[0]])
    }

    /// Context-limit stop parity (spec edge species): prompt 3 + 3 generated
    /// fills a 6-token context on both paths.
    func testGenerateTokensStopsAtContextLimit() throws {
        let model = try makeTinyModel(maxContext: 6)
        let generated = try DecodeLoop(model: model, maxContext: 6)
            .generateTokens(promptIds: [1, 2, 3], maxNewTokens: 10)
        XCTAssertEqual(generated.count, 3)
    }

    // MARK: - Protocol routing (no logits readback on the token path)

    /// A source that counts calls on both protocol methods: proves
    /// `generateTokens` routes through the `nextGreedyToken` protocol
    /// requirement (existential dispatch), so GPUModel's override — and the
    /// 4-byte readback — is actually what runs in production.
    private final class TokenOnlySource: NextTokenLogitsSource {
        let vocabSize = 16
        private(set) var logitsCalls = 0
        private(set) var tokenCalls = 0

        func lastPositionLogits(ids: [Int]) throws -> [Float] {
            logitsCalls += 1
            var logits = [Float](repeating: 0, count: vocabSize)
            logits[ids.count % vocabSize] = 1
            return logits
        }

        func nextGreedyToken(ids: [Int]) throws -> Int {
            tokenCalls += 1
            return ids.count % vocabSize
        }
    }

    func testGenerateTokensRoutesThroughProtocolOverride() throws {
        let source = TokenOnlySource()
        var stopAfter = 3
        let generated = try DecodeLoop(model: source, maxContext: 100)
            .generateTokens(
                promptIds: [9], maxNewTokens: 10,
                shouldStop: { stopAfter -= 1; return stopAfter < 0 })
        XCTAssertEqual(generated, [1, 2, 3], "scripted tokens: ids.count % 16")
        XCTAssertEqual(source.tokenCalls, 3)
        XCTAssertEqual(
            source.logitsCalls, 0,
            "the token path must never fetch full logits — that readback is "
                + "exactly what P4-8 removes")
    }
}
