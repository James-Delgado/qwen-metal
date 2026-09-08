import XCTest
@testable import QwenMetalEngine
import Metal

/// P4-1 tests for the D1 diagnostic attribution mode (phase-4.md edge test
/// 11) — no real checkpoint needed. Two synthetic Qwen3-shaped models:
/// - the P2-4 tiny model (1 layer, hidden 8) pins the exact per-class
///   dispatch counts and segment structure;
/// - a medium model (2 layers, hidden 512) gives ms-scale GPU times for the
///   timing sanity bounds pre-committed in DECISIONS.md 2026-09-08.
final class GPUAttributionTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("GPUAttributionTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Synthetic Qwen3-shaped checkpoints (bf16-exact values)

    private struct SyntheticDims {
        let vocab: Int
        let hidden: Int
        let intermediate: Int
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let layers: Int
    }

    /// The P2-4 tiny model — exact structural pins.
    private static let tiny = SyntheticDims(
        vocab: 16, hidden: 8, intermediate: 8, heads: 2, kvHeads: 1,
        headDim: 4, layers: 1)

    /// Big enough that kernel time dominates timestamp granularity (the
    /// pre-committed [0.5×, 2.0×] production cross-check band's regime).
    private static let medium = SyntheticDims(
        vocab: 1024, hidden: 512, intermediate: 1536, heads: 8, kvHeads: 4,
        headDim: 64, layers: 2)

    private func configJSON(_ d: SyntheticDims) -> String {
        """
        {
          "attention_bias": false,
          "eos_token_id": \(d.vocab - 1),
          "head_dim": \(d.headDim),
          "hidden_size": \(d.hidden),
          "intermediate_size": \(d.intermediate),
          "max_position_embeddings": 256,
          "model_type": "qwen3",
          "num_attention_heads": \(d.heads),
          "num_hidden_layers": \(d.layers),
          "num_key_value_heads": \(d.kvHeads),
          "rms_norm_eps": 1e-06,
          "rope_theta": 10000,
          "tie_word_embeddings": true,
          "vocab_size": \(d.vocab)
        }
        """
    }

    private func tensorList(_ d: SyntheticDims) -> [(name: String, shape: [Int])] {
        var tensors: [(name: String, shape: [Int])] = [
            ("model.embed_tokens.weight", [d.vocab, d.hidden]),
            ("model.norm.weight", [d.hidden]),
        ]
        for layer in 0..<d.layers {
            let p = "model.layers.\(layer)."
            tensors += [
                (p + "input_layernorm.weight", [d.hidden]),
                (p + "self_attn.q_proj.weight", [d.heads * d.headDim, d.hidden]),
                (p + "self_attn.k_proj.weight", [d.kvHeads * d.headDim, d.hidden]),
                (p + "self_attn.v_proj.weight", [d.kvHeads * d.headDim, d.hidden]),
                (p + "self_attn.o_proj.weight", [d.hidden, d.heads * d.headDim]),
                (p + "self_attn.q_norm.weight", [d.headDim]),
                (p + "self_attn.k_norm.weight", [d.headDim]),
                (p + "post_attention_layernorm.weight", [d.hidden]),
                (p + "mlp.gate_proj.weight", [d.intermediate, d.hidden]),
                (p + "mlp.up_proj.weight", [d.intermediate, d.hidden]),
                (p + "mlp.down_proj.weight", [d.hidden, d.intermediate]),
            ]
        }
        return tensors
    }

    /// Deterministic per-tensor pattern, exact in bf16 (k/16, |k| ≤ 7 —
    /// the P2-4 convention).
    private func value(_ tensorIndex: Int, _ elementIndex: Int) -> Float {
        Float((elementIndex * 7 + tensorIndex * 13) % 15 - 7) / 16
    }

    private func writeCheckpoint(_ d: SyntheticDims) throws -> String {
        var entries: [String] = []
        var payload = Data()
        for (index, tensor) in tensorList(d).enumerated() {
            let count = tensor.shape.reduce(1, *)
            let start = payload.count
            payload.reserveCapacity(payload.count + count * 2)
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
        // Pad the header to an 8-byte multiple with trailing spaces
        // (safetensors-legal) so tensor data starts even-aligned — the
        // engine's 16-bit weight loads require 2-byte-aligned offsets.
        var headerJSON = "{\(entries.joined(separator: ","))}"
        while (headerJSON.utf8.count + 8) % 8 != 0 { headerJSON += " " }
        let header = Data(headerJSON.utf8)
        var blob = Data()
        var headerLength = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &headerLength) { blob.append(contentsOf: $0) }
        blob.append(header)
        blob.append(payload)
        let url = tempDir.appendingPathComponent("model-\(d.hidden).safetensors")
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

    private func makeModel(
        _ d: SyntheticDims, maxContext: Int = 32
    ) throws -> GPUModel {
        let context = try makeContextOrSkip()
        let checkpoint = try SafetensorsFile(path: writeCheckpoint(d))
        let config = try ModelConfig(
            jsonData: configJSON(d).data(using: .utf8)!)
        return try GPUModel(
            checkpoint: checkpoint, config: config, context: context,
            maxContext: maxContext)
    }

    // MARK: - Structural pins (tiny model): per-class dispatch counts

    /// The class → dispatch mapping at 1 layer, from the P2-4 pipeline
    /// structure: matvec 7 (qkv, o, gate, up, down), attention 3
    /// (scores, softmax, pv), norm+elementwise 11 (input norm, 2 qk-norms,
    /// 2 rope, 2 append, swiglu, 2 residual, post-norm), head/tail 3
    /// (embedding + final norm + lm_head) or 1 without logits.
    func testTinyModelClassDispatchCountPins() throws {
        let model = try makeModel(Self.tiny, maxContext: 16)
        let (a, logits) = try model.attributedStep(token: 1, computeLogits: true)
        XCTAssertNotNil(logits)
        XCTAssertEqual(a.classDispatchCount(.matvec), 7)
        XCTAssertEqual(a.classDispatchCount(.attention), 3)
        XCTAssertEqual(a.classDispatchCount(.normElementwise), 11)
        XCTAssertEqual(a.classDispatchCount(.headTail), 3)
        XCTAssertEqual(a.dispatchCountTotal, 24,
                       "class dispatch counts must sum to the P2-5 pinned total")

        let (b, noLogits) = try model.attributedStep(token: 2, computeLogits: false)
        XCTAssertNil(noLogits)
        XCTAssertEqual(b.classDispatchCount(.headTail), 1)
        XCTAssertEqual(b.dispatchCountTotal, 22)
    }

    /// Segment structure at 1 layer: 12 class-contiguous segments with the
    /// logits tail (head/tail, elem, matvec, elem, attention, matvec, elem,
    /// matvec, elem, matvec, elem, head/tail — counted from the encode
    /// order), 11 without it (the tail segment drops).
    func testTinyModelSegmentCountPins() throws {
        let model = try makeModel(Self.tiny, maxContext: 16)
        let (a, _) = try model.attributedStep(token: 1, computeLogits: true)
        XCTAssertEqual(a.segments.count, 12)
        let (b, _) = try model.attributedStep(token: 2, computeLogits: false)
        XCTAssertEqual(b.segments.count, 11)
    }

    /// Per-class GPU seconds and dispatch counts must be exactly the sums
    /// over that class's segments (bookkeeping bound — exact, no tolerance).
    func testClassTotalsAreExactSegmentSums() throws {
        let model = try makeModel(Self.tiny, maxContext: 16)
        let (a, _) = try model.attributedStep(token: 3, computeLogits: true)
        for kernelClass in KernelClass.allCases {
            let segments = a.segments.filter { $0.kernelClass == kernelClass }
            XCTAssertEqual(
                a.classGPUSeconds(kernelClass),
                segments.reduce(0) { $0 + $1.gpuSeconds })
            XCTAssertEqual(
                a.classDispatchCount(kernelClass),
                segments.reduce(0) { $0 + $1.dispatchCount })
        }
        XCTAssertEqual(
            a.gpuSecondsTotal,
            a.segments.reduce(0) { $0 + $1.gpuSeconds })
        XCTAssertEqual(
            a.dispatchCountTotal,
            a.segments.reduce(0) { $0 + $1.dispatchCount })
    }

    /// Bracketing (hard rule 7 + the pre-committed structural bounds):
    /// every segment nonnegative, diagnostic wall ≥ span ≥ per-class sums,
    /// and the class-time total inside [0.5 × span, 1.01 × span + 1 µs].
    func testAttributionBracketingAndCoverageBounds() throws {
        let model = try makeModel(Self.medium, maxContext: 32)
        for token in [1, 2, 3] {
            try model.step(token: token, computeLogits: false)
        }
        let (a, _) = try model.attributedStep(token: 4, computeLogits: true)
        for segment in a.segments {
            XCTAssertGreaterThanOrEqual(segment.gpuEnd, segment.gpuStart)
        }
        XCTAssertGreaterThan(a.gpuSecondsTotal, 0, "GPU did real work")
        XCTAssertGreaterThanOrEqual(a.wallSeconds, a.spanSeconds)
        for kernelClass in KernelClass.allCases {
            XCTAssertLessThanOrEqual(
                a.classGPUSeconds(kernelClass), a.spanSeconds + 1e-6)
        }
        XCTAssertGreaterThanOrEqual(
            a.gpuSecondsTotal, 0.5 * a.spanSeconds,
            "inter-buffer gaps must not swallow most of the GPU window")
        XCTAssertLessThanOrEqual(
            a.gpuSecondsTotal, 1.01 * a.spanSeconds + 1e-6,
            "class sums cannot exceed the window they occurred in")
    }

    /// The pre-committed production cross-check: median attributed
    /// class-time total within [0.5×, 2.0×] of the median production
    /// single-command-buffer GPU time at adjacent cache depths.
    func testAttributedTotalsWithinProductionCrossCheckBand() throws {
        let model = try makeModel(Self.medium, maxContext: 64)
        try model.step(token: 1, computeLogits: false)
        var attributedTotals: [Double] = []
        var productionGPU: [Double] = []
        for step in 0..<8 {
            let token = 2 + step
            if step.isMultiple(of: 2) {
                let (a, _) = try model.attributedStep(token: token, computeLogits: true)
                attributedTotals.append(a.gpuSecondsTotal)
            } else {
                try model.step(token: token, computeLogits: true)
                productionGPU.append(try XCTUnwrap(model.lastStepTiming).gpuDuration)
            }
        }
        let attributedMedian = attributedTotals.sorted()[attributedTotals.count / 2]
        let productionMedian = productionGPU.sorted()[productionGPU.count / 2]
        let ratio = attributedMedian / productionMedian
        XCTAssertGreaterThanOrEqual(
            ratio, 0.5, "attributed sum lost most of the production GPU time")
        XCTAssertLessThanOrEqual(
            ratio, 2.0, "attributed sum wildly exceeds production GPU time")
    }

    // MARK: - Production-path invariance (exact)

    /// Splitting the encode across command buffers must not change
    /// arithmetic: attributed logits are bitwise identical to a production
    /// step's logits for the same token at the same cache state.
    func testAttributedLogitsBitwiseEqualProductionLogits() throws {
        let production = try makeModel(Self.tiny, maxContext: 16)
        let diagnostic = try makeModel(Self.tiny, maxContext: 16)
        for token in [1, 2, 3] {
            try production.step(token: token, computeLogits: false)
            try diagnostic.step(token: token, computeLogits: false)
        }
        let productionLogits = try XCTUnwrap(
            try production.step(token: 4, computeLogits: true))
        let (_, attributedLogits) = try diagnostic.attributedStep(
            token: 4, computeLogits: true)
        XCTAssertEqual(try XCTUnwrap(attributedLogits), productionLogits)
    }

    /// attributedStep advances the cache exactly like step, and a production
    /// step afterwards still measures the pinned single-command-buffer
    /// counts — diagnostic mode leaves the production path untouched.
    func testAttributedStepAdvancesCacheAndLeavesProductionUntouched() throws {
        let model = try makeModel(Self.tiny, maxContext: 16)
        _ = try model.attributedStep(token: 1, computeLogits: false)
        XCTAssertEqual(model.cachedTokens, [1])
        XCTAssertNil(model.lastStepTiming,
                     "a diagnostic step publishes no production timing")
        XCTAssertNil(model.lastStepDispatchCount)

        try model.step(token: 2, computeLogits: true)
        XCTAssertEqual(model.cachedTokens, [1, 2])
        XCTAssertEqual(model.lastStepDispatchCount, 24)
        let timing = try XCTUnwrap(model.lastStepTiming)
        XCTAssertGreaterThanOrEqual(timing.wallDuration, timing.gpuDuration)
    }

    /// Mixed production/attributed decode is bitwise identical to pure
    /// production decode — the interleaved AttributionRunner pattern is
    /// arithmetically transparent.
    func testInterleavedDecodeMatchesPureProductionBitwise() throws {
        let pure = try makeModel(Self.tiny, maxContext: 16)
        let mixed = try makeModel(Self.tiny, maxContext: 16)
        var pureLogits: [Float]?
        var mixedLogits: [Float]?
        for (index, token) in [1, 2, 3, 4, 5].enumerated() {
            pureLogits = try pure.step(token: token, computeLogits: true)
            if index.isMultiple(of: 2) {
                (_, mixedLogits) = try mixed.attributedStep(
                    token: token, computeLogits: true)
            } else {
                mixedLogits = try mixed.step(token: token, computeLogits: true)
            }
            XCTAssertEqual(mixedLogits, pureLogits, "divergence at step \(index)")
        }
    }

    /// Context-limit and token-range validation behave exactly like step
    /// (edge behavior unchanged in diagnostic mode).
    func testAttributedStepValidatesLikeStep() throws {
        let model = try makeModel(Self.tiny, maxContext: 2)
        _ = try model.attributedStep(token: 1, computeLogits: false)
        _ = try model.attributedStep(token: 2, computeLogits: false)
        XCTAssertThrowsError(
            try model.attributedStep(token: 3, computeLogits: false)
        ) { error in
            XCTAssertEqual(
                error as? KVCacheError,
                .contextFull(position: 2, maxContext: 2))
        }

        let fresh = try makeModel(Self.tiny, maxContext: 16)
        XCTAssertThrowsError(
            try fresh.attributedStep(token: 16, computeLogits: true)
        ) { error in
            guard case ModelError.tokenIdOutOfRange = error else {
                return XCTFail("expected tokenIdOutOfRange, got \(error)")
            }
        }
    }

    // MARK: - AttributionRunner (interleaved harness)

    func testRunnerInterleavesAttributedAndProductionSteps() throws {
        let model = try makeModel(Self.tiny, maxContext: 32)
        let runner = AttributionRunner(gpuModel: model, maxContext: 32)
        let result = try runner.run(promptIds: [1, 2, 3], decodeTokens: 6)
        // Steps 0, 2, 4 attributed; 1, 3, 5 production.
        XCTAssertEqual(result.attributed.count, 3)
        XCTAssertEqual(result.productionGPUSeconds.count, 3)
        XCTAssertEqual(result.promptTokenCount, 3)
        XCTAssertEqual(result.productionDispatchCount, 24)
        // First decode forward runs the LAST prompt token at position 2.
        XCTAssertEqual(result.firstDecodePosition, 2)
        XCTAssertEqual(result.lastDecodePosition, 7)
        XCTAssertEqual(
            result.attributed.map(\.position), [2, 4, 6],
            "attributed tokens sit at even interleave offsets")
    }

    func testRunnerStopsOnEOS() throws {
        let model = try makeModel(Self.tiny, maxContext: 32)
        // Find what greedy decode emits first, then make it the stop token.
        let probeRunner = AttributionRunner(gpuModel: model, maxContext: 32)
        let probe = try probeRunner.run(promptIds: [1, 2], decodeTokens: 4)
        let firstToken = try XCTUnwrap(probe.generatedTokenIds.first)

        model.reset()
        let runner = AttributionRunner(
            gpuModel: model, maxContext: 32, eosTokenIds: [firstToken])
        let stopped = try runner.run(promptIds: [1, 2], decodeTokens: 4)
        XCTAssertEqual(stopped.generatedTokenIds, [firstToken],
                       "decode must stop right after emitting a stop token")
    }

    func testRunnerRejectsBadInputs() throws {
        let model = try makeModel(Self.tiny, maxContext: 8)
        let runner = AttributionRunner(gpuModel: model, maxContext: 8)
        XCTAssertThrowsError(try runner.run(promptIds: [], decodeTokens: 2))
        XCTAssertThrowsError(try runner.run(promptIds: [1, 2], decodeTokens: 0))
        // prompt 6 + 3 decode tokens > 8 context.
        XCTAssertThrowsError(
            try runner.run(promptIds: [1, 2, 3, 4, 5, 6], decodeTokens: 3))
    }

    // MARK: - Report formatting (hand-built values, no GPU)

    func testAttributionExportTextFormatsClassesAndSanityRatio() throws {
        func segment(
            _ kernelClass: KernelClass, start: Double, seconds: Double,
            dispatches: Int
        ) -> TokenAttribution.Segment {
            TokenAttribution.Segment(
                kernelClass: kernelClass, gpuStart: start,
                gpuEnd: start + seconds, dispatchCount: dispatches)
        }
        // One attributed token: matvec 30 ms, attention 10 ms,
        // norm+elementwise 5 ms, head/tail 5 ms → total 50 ms.
        let attribution = TokenAttribution(
            position: 100, wallSeconds: 0.060,
            segments: [
                segment(.headTail, start: 0.000, seconds: 0.002, dispatches: 1),
                segment(.matvec, start: 0.002, seconds: 0.030, dispatches: 196),
                segment(.attention, start: 0.032, seconds: 0.010, dispatches: 84),
                segment(.normElementwise, start: 0.042, seconds: 0.005, dispatches: 308),
                segment(.headTail, start: 0.047, seconds: 0.003, dispatches: 2),
            ])
        let result = AttributionRunResult(
            weightsFormat: .q4g64, promptTokenCount: 84,
            attributed: [attribution],
            productionGPUSeconds: [0.040],
            productionDispatchCount: 591,
            firstDecodePosition: 83, lastDecodePosition: 84,
            generatedTokenIds: [7, 9])
        let text = result.exportText(
            dateStamp: "2026-09-08", deviceLabel: "TestDevice",
            osVersion: "macOS test", residency: .mmap)
        XCTAssertTrue(text.contains("DIAGNOSTIC"), text)
        XCTAssertTrue(text.contains("never a benchmark row"), text)
        XCTAssertTrue(text.contains("matvec"), text)
        XCTAssertTrue(text.contains("attention"), text)
        XCTAssertTrue(text.contains("norm+elementwise"), text)
        XCTAssertTrue(text.contains("head/tail"), text)
        XCTAssertTrue(text.contains("weights q4g64"), text)
        // matvec median 30 ms at 60% share of the 50 ms class-sum.
        XCTAssertTrue(text.contains("30.00 ms"), text)
        XCTAssertTrue(text.contains("60.0%"), text)
        // Sanity ratio 50 / 40 = 1.25 against the pre-committed band.
        XCTAssertTrue(text.contains("1.25"), text)
        XCTAssertTrue(text.contains("591"), text)
    }
}
