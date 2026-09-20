import XCTest
@testable import QwenMetalEngine
import Metal

/// PF-1 tests for the DIAGNOSTIC prefill attribution mode against the
/// bounds pre-committed in DECISIONS.md 2026-09-18 (the P4-1 bounds
/// verbatim, "token" read as "chunk"). Two packed synthetic Qwen3-shaped
/// models built with the real `Q4Packer`:
/// - the P5-3 tiny fixture (1 layer, hidden 64) pins the exact per-class
///   dispatch counts and segment structure per chunk;
/// - a medium model (2 layers, hidden 512) gives ms-scale GPU times for the
///   timing sanity bounds (coverage, production cross-check band).
final class PrefillAttributionTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("PrefillAttributionTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Synthetic packed Qwen3-shaped checkpoints

    private struct Dims {
        let vocab: Int
        let hidden: Int
        let intermediate: Int
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let layers: Int
    }

    /// The P5-3 tiny fixture dims — exact structural pins.
    private static let tiny = Dims(
        vocab: 32, hidden: 64, intermediate: 128, heads: 4, kvHeads: 2,
        headDim: 16, layers: 1)

    /// Every in-dim a multiple of 64 (the q4g64 group); sized so per-chunk
    /// kernel time dominates the ~21 per-chunk command-buffer scheduling
    /// gaps the split mode adds — the timing bounds' regime (with PF-1's
    /// one-dispatch attention and cooperative norms, a hidden-512 chunk of
    /// 16 ran in ≈2.5 ms and the gaps alone matched it; the bound is
    /// pre-committed and stays, the fixture moves into its regime). Two
    /// FULL chunks — a ragged 32-position tail chunk sat below the 0.5×
    /// coverage line on Mac for the same reason; ragged chunking itself is
    /// pinned on the tiny fixture ([4, 1]).
    private static let medium = Dims(
        vocab: 1024, hidden: 1024, intermediate: 3072, heads: 8, kvHeads: 4,
        headDim: 128, layers: 2)
    private static let mediumChunk = 128
    private static let mediumPrompt = 256
    private static let mediumContext = 256

    private func configJSON(_ d: Dims) -> String {
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

    private func tensorList(_ d: Dims) -> [(name: String, shape: [Int])] {
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

    /// Deterministic bf16-exact values (1/64 steps, |v| ≤ 0.47) — the
    /// P5-3 fixture range that keeps fp16 intermediates in range.
    private func value(_ tensorIndex: Int, _ elementIndex: Int) -> Float {
        Float(((elementIndex &* 37 &+ tensorIndex &* 7 &+ 1) % 61) - 30) * 0.015625
    }

    private func writePackedCheckpoint(_ d: Dims) throws -> PackedCheckpoint {
        var entries: [String] = ["\"__metadata__\":{\"source_revision\":\"r1\"}"]
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
        var headerJSON = "{\(entries.joined(separator: ","))}"
        while (headerJSON.utf8.count + 8) % 8 != 0 { headerJSON += " " }
        let header = Data(headerJSON.utf8)
        var blob = Data()
        var headerLength = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &headerLength) { blob.append(contentsOf: $0) }
        blob.append(header)
        blob.append(payload)
        let source = tempDir.appendingPathComponent("source-\(d.hidden).safetensors")
        try blob.write(to: source)
        let packedPath = tempDir.appendingPathComponent("packed-\(d.hidden).safetensors").path
        _ = try Q4Packer.pack(inputPath: source.path, outputPath: packedPath)
        return try PackedCheckpoint(path: packedPath)
    }

    private func makeContextOrSkip() throws -> MetalContext {
        do {
            return try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    private func makeTiledModel(
        _ d: Dims, maxContext: Int = 32, chunkSize: Int
    ) throws -> GPUModel {
        try GPUModel(
            packed: try writePackedCheckpoint(d),
            config: try ModelConfig(jsonData: Data(configJSON(d).utf8)),
            context: try makeContextOrSkip(), maxContext: maxContext,
            prefillPath: .tiled, prefillChunkSize: chunkSize)
    }

    private func prompt(_ count: Int, vocab: Int) -> [Int] {
        (0..<count).map { ($0 * 7 + 3) % vocab }
    }

    // MARK: - Structural pins (tiny model, C=4, prompt 5 → chunks 4 + 1)

    /// Per-chunk class → dispatch mapping from the chunk structure at
    /// 1 layer: gemm 7 (q,k,v,o,gate,up,down), attention 1 (PF-1: one
    /// batched causal dispatch, was 2·B), norm+elementwise 6 (norm,
    /// cluster, residual, norm, swiglu, residual), head/tail 1 (gather) + 3
    /// on the last chunk (row copy, final norm, lm_head). Totals 15 + 18 =
    /// the production span pin (measured, P2-5 rule).
    func testTinyModelPerChunkClassDispatchPins() throws {
        let model = try makeTiledModel(Self.tiny, chunkSize: 4)
        let ids = prompt(5, vocab: Self.tiny.vocab)
        let attribution = try model.attributedPrefill(ids: ids).attribution

        XCTAssertEqual(attribution.chunkSizes, [4, 1])
        XCTAssertEqual(attribution.positionCount, 5)
        let first = attribution.chunks[0]
        XCTAssertEqual(first.position, 0)
        XCTAssertEqual(first.classDispatchCount(.gemm), 7)
        XCTAssertEqual(first.classDispatchCount(.attention), 1)
        XCTAssertEqual(first.classDispatchCount(.normElementwise), 6)
        XCTAssertEqual(first.classDispatchCount(.headTail), 1)
        XCTAssertEqual(first.classDispatchCount(.matvec), 0,
                       "prefill projections are GEMMs, never the decode matvec class")
        XCTAssertEqual(first.dispatchCountTotal, 15)
        let last = attribution.chunks[1]
        XCTAssertEqual(last.position, 4)
        XCTAssertEqual(last.classDispatchCount(.gemm), 7)
        XCTAssertEqual(last.classDispatchCount(.attention), 1)
        XCTAssertEqual(last.classDispatchCount(.normElementwise), 6)
        XCTAssertEqual(last.classDispatchCount(.headTail), 4)
        XCTAssertEqual(last.dispatchCountTotal, 18)
        XCTAssertEqual(attribution.dispatchCountTotal, 33,
                       "= the production span pin (15 + 18)")
        XCTAssertEqual(model.cachedTokens, ids,
                       "attributed prefill appends the prompt like production")
    }

    /// Segment structure: one buffer per contiguous same-class run, in
    /// encode order — the gather segment, then per layer NE(norm),
    /// gemm(q,k,v), NE(cluster), attention, gemm(o), NE(res, norm),
    /// gemm(gate,up), NE(swiglu), gemm(down), NE(res) → 10 segments per
    /// layer, plus the head/tail segment on the last chunk (measured pins,
    /// P2-5 rule: 11 and 12 at 1 layer).
    func testTinyModelSegmentCountPins() throws {
        let model = try makeTiledModel(Self.tiny, chunkSize: 4)
        let attribution = try model.attributedPrefill(
            ids: prompt(5, vocab: Self.tiny.vocab)).attribution
        XCTAssertEqual(attribution.chunks[0].segments.count, 1 + 10)
        XCTAssertEqual(attribution.chunks[1].segments.count, 1 + 10 + 1)
        // Adjacent segments never share a class (they would have merged).
        for chunk in attribution.chunks {
            for pair in zip(chunk.segments, chunk.segments.dropFirst()) {
                XCTAssertNotEqual(pair.0.kernelClass, pair.1.kernelClass)
            }
        }
    }

    // MARK: - Pre-committed sanity bounds (DECISIONS.md 2026-09-18)

    /// Bookkeeping is exact: class sums equal the segment sums, class
    /// dispatch counts sum to the chunk total, and the run-level rollups
    /// equal the Σ over chunks.
    func testClassTotalsAreExactSegmentSums() throws {
        let model = try makeTiledModel(Self.tiny, chunkSize: 4)
        let attribution = try model.attributedPrefill(
            ids: prompt(5, vocab: Self.tiny.vocab)).attribution
        for chunk in attribution.chunks {
            var classSum = 0.0
            var dispatchSum = 0
            for kernelClass in KernelClass.allCases {
                let segments = chunk.segments.filter { $0.kernelClass == kernelClass }
                XCTAssertEqual(
                    chunk.classGPUSeconds(kernelClass),
                    segments.reduce(0) { $0 + $1.gpuSeconds })
                XCTAssertEqual(
                    chunk.classDispatchCount(kernelClass),
                    segments.reduce(0) { $0 + $1.dispatchCount })
                classSum += chunk.classGPUSeconds(kernelClass)
                dispatchSum += chunk.classDispatchCount(kernelClass)
            }
            XCTAssertEqual(classSum, chunk.gpuSecondsTotal, accuracy: 1e-12)
            XCTAssertEqual(dispatchSum, chunk.dispatchCountTotal)
        }
        XCTAssertEqual(
            attribution.gpuSecondsTotal,
            attribution.chunks.reduce(0) { $0 + $1.gpuSecondsTotal },
            accuracy: 1e-12)
        XCTAssertEqual(
            attribution.dispatchCountTotal,
            attribution.chunks.reduce(0) { $0 + $1.dispatchCountTotal })
    }

    /// Bracketing (hard rule 7) + coverage on the medium model: per chunk,
    /// wall ≥ span ≥ each class sum; every segment gpuEnd ≥ gpuStart;
    /// class-sum within [0.5 × span, 1.01 × span + 1 µs].
    func testBracketingAndCoverageBounds() throws {
        let model = try makeTiledModel(
            Self.medium, maxContext: Self.mediumContext, chunkSize: Self.mediumChunk)
        let attribution = try model.attributedPrefill(
            ids: prompt(Self.mediumPrompt, vocab: Self.medium.vocab)).attribution
        XCTAssertEqual(attribution.chunkSizes, [128, 128])
        for chunk in attribution.chunks {
            for segment in chunk.segments {
                XCTAssertGreaterThanOrEqual(segment.gpuEnd, segment.gpuStart)
            }
            let span = chunk.spanSeconds
            XCTAssertGreaterThanOrEqual(chunk.wallSeconds, span)
            for kernelClass in KernelClass.allCases {
                XCTAssertGreaterThanOrEqual(span, chunk.classGPUSeconds(kernelClass))
            }
            XCTAssertGreaterThanOrEqual(
                chunk.gpuSecondsTotal, 0.5 * span,
                "inter-buffer gaps must not swallow the majority of the chunk window")
            XCTAssertLessThanOrEqual(chunk.gpuSecondsTotal, 1.01 * span + 1e-6)
        }
    }

    /// Production cross-check: median attributed class-sum within
    /// [0.5×, 2.0×] of the median production prefill GPU time for the same
    /// prompt (interleaved by the runner).
    func testAttributedTotalsWithinProductionCrossCheckBand() throws {
        let model = try makeTiledModel(
            Self.medium, maxContext: Self.mediumContext, chunkSize: Self.mediumChunk)
        let result = try PrefillAttributionRunner(
            gpuModel: model, maxContext: Self.mediumContext
        ).run(promptIds: prompt(Self.mediumPrompt, vocab: Self.medium.vocab), runs: 6)
        XCTAssertEqual(result.attributed.count, 3)
        XCTAssertEqual(result.productionGPUSeconds.count, 3)
        let total = try XCTUnwrap(result.medianGPUSecondsTotal)
        let production = try XCTUnwrap(result.medianProductionGPUSeconds)
        XCTAssertGreaterThan(production, 0)
        let ratio = total / production
        XCTAssertGreaterThanOrEqual(ratio, 0.5, "ratio \(ratio)")
        XCTAssertLessThanOrEqual(ratio, 2.0, "ratio \(ratio)")
        XCTAssertEqual(result.productionDispatchCount,
                       result.attributed[0].dispatchCountTotal,
                       "production and attributed dispatch counts agree")
    }

    /// Production-path invariance (exact): attributed prefill logits are
    /// bitwise equal to the production tiled prefill's for the same prompt
    /// — splitting command buffers changes no arithmetic — and the cache
    /// contents match too.
    func testAttributedLogitsBitwiseEqualProductionLogits() throws {
        let model = try makeTiledModel(
            Self.medium, maxContext: Self.mediumContext, chunkSize: Self.mediumChunk)
        let ids = prompt(Self.mediumPrompt, vocab: Self.medium.vocab)
        let production = try model.lastPositionLogits(ids: ids)
        let attributed = try model.attributedPrefill(ids: ids).logits
        XCTAssertEqual(attributed, production,
                       "class-split prefill must be bitwise identical to production")
        // And production after an attributed run is unchanged (the
        // attributed run left a clean, fully-appended cache).
        XCTAssertEqual(model.cachedTokens, ids)
        model.reset()
        XCTAssertEqual(try model.lastPositionLogits(ids: ids), production)
    }

    /// Diagnostic runs leave no production timing behind (they are never
    /// rows) and reject what production rejects.
    func testAttributedPrefillClearsTimingAndValidatesLikeProduction() throws {
        let model = try makeTiledModel(Self.tiny, chunkSize: 4)
        _ = try model.lastPositionLogits(ids: prompt(5, vocab: Self.tiny.vocab))
        XCTAssertNotNil(model.lastCallSpan)
        _ = try model.attributedPrefill(ids: prompt(5, vocab: Self.tiny.vocab))
        XCTAssertNil(model.lastCallSpan)
        XCTAssertNil(model.lastStepTiming)
        XCTAssertNil(model.lastStepDispatchCount)

        XCTAssertThrowsError(try model.attributedPrefill(ids: [1, 99])) { error in
            guard case ModelError.tokenIdOutOfRange = error else {
                return XCTFail("expected tokenIdOutOfRange, got \(error)")
            }
        }
        XCTAssertThrowsError(try model.attributedPrefill(ids: [1])) { error in
            guard case ModelError.badInput = error else {
                return XCTFail("expected badInput, got \(error)")
            }
        }
        XCTAssertThrowsError(
            try model.attributedPrefill(ids: prompt(33, vocab: Self.tiny.vocab))
        ) { error in
            guard case KVCacheError.contextFull = error else {
                return XCTFail("expected contextFull, got \(error)")
            }
        }

        let sequential = try GPUModel(
            packed: try writePackedCheckpoint(Self.tiny),
            config: try ModelConfig(jsonData: Data(configJSON(Self.tiny).utf8)),
            context: try makeContextOrSkip(), maxContext: 32,
            prefillPath: .sequential)
        XCTAssertThrowsError(
            try sequential.attributedPrefill(ids: prompt(5, vocab: Self.tiny.vocab))
        ) { error in
            guard case ModelError.badInput = error else {
                return XCTFail("expected badInput, got \(error)")
            }
        }
    }

    // MARK: - Runner + report

    func testRunnerInterleavesAndRejectsBadInputs() throws {
        let model = try makeTiledModel(Self.tiny, chunkSize: 4)
        let runner = PrefillAttributionRunner(gpuModel: model, maxContext: 32)
        var runs: [Int] = []
        let result = try runner.run(
            promptIds: prompt(5, vocab: Self.tiny.vocab), runs: 5,
            onRun: { runs.append($0) })
        XCTAssertEqual(runs, [0, 1, 2, 3, 4])
        XCTAssertEqual(result.attributed.count, 3)
        XCTAssertEqual(result.productionGPUSeconds.count, 2)
        XCTAssertEqual(result.productionDispatchCount, 33)
        XCTAssertEqual(result.prefillChunkSize, 4)
        XCTAssertEqual(result.promptTokenCount, 5)
        XCTAssertTrue(model.cachedTokens.isEmpty, "runner leaves the cache reset")

        XCTAssertThrowsError(try runner.run(promptIds: [1], runs: 2))
        XCTAssertThrowsError(
            try runner.run(promptIds: prompt(5, vocab: Self.tiny.vocab), runs: 1))
        XCTAssertThrowsError(
            try runner.run(promptIds: prompt(33, vocab: Self.tiny.vocab), runs: 2))
    }

    func testExportTextFormatsClassesAndSanityRatio() {
        func segment(
            _ kernelClass: KernelClass, start: Double, seconds: Double,
            dispatches: Int
        ) -> TokenAttribution.Segment {
            TokenAttribution.Segment(
                kernelClass: kernelClass, gpuStart: start,
                gpuEnd: start + seconds, dispatchCount: dispatches)
        }
        // One attributed prefill of two chunks: gemm 3.0 s, attention
        // 4.0 s, norm+elementwise 1.0 s, head/tail 0.5 s → 8.5 s.
        let chunk1 = TokenAttribution(
            position: 0, wallSeconds: 5.0,
            segments: [
                segment(.headTail, start: 0.0, seconds: 0.1, dispatches: 1),
                segment(.gemm, start: 0.1, seconds: 2.0, dispatches: 196),
                segment(.attention, start: 2.1, seconds: 2.0, dispatches: 28672),
                segment(.normElementwise, start: 4.1, seconds: 0.6, dispatches: 168),
            ])
        let chunk2 = TokenAttribution(
            position: 512, wallSeconds: 4.0,
            segments: [
                segment(.headTail, start: 5.0, seconds: 0.1, dispatches: 1),
                segment(.gemm, start: 5.1, seconds: 1.0, dispatches: 196),
                segment(.attention, start: 6.1, seconds: 2.0, dispatches: 19040),
                segment(.normElementwise, start: 8.1, seconds: 0.4, dispatches: 168),
                segment(.headTail, start: 8.5, seconds: 0.3, dispatches: 3),
            ])
        let result = PrefillAttributionRunResult(
            weightsFormat: .q4g64, kernelPath: .fused, prefillChunkSize: 512,
            prefillAttention: .queryTiled, promptTokenCount: 852,
            attributed: [PrefillAttribution(chunks: [chunk1, chunk2], chunkSizes: [512, 340])],
            productionGPUSeconds: [8.3],
            productionDispatchCount: 48446)
        let text = result.exportText(
            dateStamp: "2026-09-18", deviceLabel: "TestDevice",
            osVersion: "macOS test", residency: .mmap)
        XCTAssertTrue(text.contains("PREFILL"), text)
        XCTAssertTrue(text.contains("DIAGNOSTIC"), text)
        XCTAssertTrue(text.contains("never a benchmark row"), text)
        XCTAssertTrue(text.contains("prefill tiled (C=512), attention query-tiled"), text)
        XCTAssertTrue(text.contains("852 tokens in 2 chunk(s)"), text)
        // gemm 3000 ms at 3.0/8.5 = 35.3% of class-sum; attention 47.1%.
        XCTAssertTrue(text.contains("3000.00 ms"), text)
        XCTAssertTrue(text.contains("35.3%"), text)
        XCTAssertTrue(text.contains("47.1%"), text)
        XCTAssertTrue(text.contains("47712 dispatches"), text)
        // Sanity ratio 8.5 / 8.3 = 1.02; production 852 / 8.3 = 102.65 tok/s.
        XCTAssertTrue(text.contains("1.02"), text)
        XCTAssertTrue(text.contains("48446"), text)
        XCTAssertTrue(text.contains("102.65 tok/s"), text)
    }
}
