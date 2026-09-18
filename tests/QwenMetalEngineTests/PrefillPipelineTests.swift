import XCTest
@testable import QwenMetalEngine
import Metal

/// P5-3 pipeline tests for the chunked batched prefill (phase-5.md edge
/// tests 3–8, 10–11), on the GPUQuantModelTests tiny fixture: a 1-layer
/// Qwen3-shaped bf16 source packed with the real `Q4Packer`, loaded through
/// `PackedCheckpoint` and driven through `GPUModel(packed:)` with
/// `prefillPath: .tiled`. The CPU-quant reference (`QwenModel` over the
/// same packed file) is the oracle; every constant is a pre-committed
/// Phase 2/3/5 gate (extended span-mapping rule — DECISIONS.md 2026-09-14
/// Phase 5 gates entry; nothing new, nothing loosened):
///
///   full-stack logits            max(2⁻⁵·M, 2⁻¹¹)
///   KV contents (norm-inclusive) max(2⁻⁸·M, 2⁻¹¹)
///   copies (v-append, gather)    EXACT bitwise
final class PrefillPipelineTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("PrefillPipelineTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    private func path(_ name: String) -> String {
        tempDir.appendingPathComponent(name).path
    }

    // MARK: - Tiny Qwen3-shaped source + packed checkpoint (GPUQuantModelTests dims)

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

    /// Deterministic bf16-exact values (1/64 steps, |v| ≤ 0.47) — the
    /// GPUQuantModelTests range that keeps fp16 intermediates in range.
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

    private func makePackedCheckpoint() throws -> PackedCheckpoint {
        let source = try makeSourceFile(tensors: tinyTensors())
        let out = path("packed.safetensors")
        _ = try Q4Packer.pack(inputPath: source, outputPath: out)
        return try PackedCheckpoint(path: out)
    }

    private func makeTiledModel(
        maxContext: Int = 16, chunkSize: Int = 4
    ) throws -> GPUModel {
        let context = try makeContextOrSkip()
        return try GPUModel(
            packed: try makePackedCheckpoint(), config: try tinyConfig(),
            context: context, maxContext: maxContext,
            prefillPath: .tiled, prefillChunkSize: chunkSize)
    }

    /// The sequential comparator asks for `.sequential` explicitly — since
    /// P5-4 the packed + fused default is `.tiled` (spec D5).
    private func makeSequentialModel(maxContext: Int = 16) throws -> GPUModel {
        let context = try makeContextOrSkip()
        return try GPUModel(
            packed: try makePackedCheckpoint(), config: try tinyConfig(),
            context: context, maxContext: maxContext, prefillPath: .sequential)
    }

    /// The production DEFAULT (no prefill arguments) — tiled since P5-4.
    private func makeDefaultModel(maxContext: Int = 16) throws -> GPUModel {
        let context = try makeContextOrSkip()
        return try GPUModel(
            packed: try makePackedCheckpoint(), config: try tinyConfig(),
            context: context, maxContext: maxContext)
    }

    private func makeCPUModel(maxSequenceLength: Int = 16) throws -> QwenModel {
        try QwenModel(
            weights: try makePackedCheckpoint(), config: try tinyConfig(),
            maxSequenceLength: maxSequenceLength)
    }

    /// Full-stack species gate (Phase 2, reused verbatim — spec D6).
    private func assertFullStack(
        _ got: [Float], _ ref: [Float], _ surface: String
    ) {
        XCTAssertEqual(got.count, ref.count, surface)
        let m = ref.map(abs).max() ?? 0
        let tolerance = max(exp2(-5) * m, exp2(-11))
        for i in 0..<ref.count where abs(got[i] - ref[i]) > tolerance {
            XCTFail("\(surface): index \(i), got \(got[i]) vs ref \(ref[i]) "
                + "(tolerance \(tolerance))")
            return
        }
    }

    private func readCacheSlot(
        _ model: GPUModel, component: KVCache.Component, head: Int, position: Int
    ) throws -> [Float16] {
        let offset = try model.kvCache.elementOffset(
            layer: 0, component: component, head: head, position: position)
        let headDim = model.kvCache.headDim
        return model.kvCache.buffer.contents()
            .advanced(by: offset * 2)
            .withMemoryRebound(to: Float16.self, capacity: headDim) {
                Array(UnsafeBufferPointer(start: $0, count: headDim))
            }
    }

    // MARK: - Edge 3: causal masking

    /// Prefix invariance IS causality at pipeline level: extending the
    /// prompt must not change any already-computed position, so an
    /// incremental tiled prefill is BITWISE identical to a fresh replay.
    /// Every per-row computation (GEMM row, norm row, cluster head, SDPA at
    /// depth i) is independent of later rows by construction — if position
    /// i ever read a slot j > i, the incremental run (where slot j did not
    /// yet exist) would diverge. Batches stay ≤ 8 so all chunkings hit the
    /// same GEMM kernel (m8) — bitwise comparability across partitions.
    func testIncrementalTiledPrefillMatchesFreshReplayBitwise() throws {
        let incremental = try makeTiledModel()
        _ = try incremental.lastPositionLogits(ids: [1, 2, 3])
        let extended = try incremental.lastPositionLogits(ids: [1, 2, 3, 4, 5])
        XCTAssertEqual(incremental.cachedTokens, [1, 2, 3, 4, 5])

        let fresh = try makeTiledModel()
        let replay = try fresh.lastPositionLogits(ids: [1, 2, 3, 4, 5])
        XCTAssertEqual(
            extended, replay,
            "incremental tiled prefill must be bitwise identical to a fresh "
            + "replay — a difference means a position read a later slot")
    }

    /// Cache slots beyond the prompt are poisoned with NaN before the
    /// prefill: bitwise-identical logits to an unpoisoned twin prove no
    /// SDPA read past its causal depth (NaN propagates through softmax).
    func testTiledPrefillIgnoresPoisonedCacheBeyondPrompt() throws {
        let clean = try makeTiledModel()
        let cleanLogits = try clean.lastPositionLogits(ids: [1, 2, 3, 4, 5])

        let poisoned = try makeTiledModel()
        let nanPattern: UInt16 = 0x7E00
        let count = poisoned.kvCache.buffer.length / 2
        poisoned.kvCache.buffer.contents()
            .withMemoryRebound(to: UInt16.self, capacity: count) {
                for i in 0..<count { $0[i] = nanPattern }
            }
        let poisonedLogits = try poisoned.lastPositionLogits(ids: [1, 2, 3, 4, 5])
        XCTAssertEqual(
            poisonedLogits, cleanLogits,
            "poisoned out-of-prompt cache slots must be unreachable")
        XCTAssertTrue(poisonedLogits.allSatisfy { $0.isFinite })
    }

    // MARK: - Edge 4: chunk-boundary continuity (multi-chunk vs oracle)

    /// A 10-token prompt at C=4 spans three chunks (4+4+2), covering
    /// positions {0, 1, C−1, C, last}: last-position logits gate against
    /// the CPU-quant oracle at the full-stack constant (chunk-2/3
    /// attention reads chunk-1 slots; absolute RoPE indexing across
    /// boundaries).
    func testMultiChunkPrefillLogitsMatchCPUQuantOracle() throws {
        let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let gpu = try makeTiledModel(chunkSize: 4)
        let cpu = try makeCPUModel()
        assertFullStack(
            try gpu.lastPositionLogits(ids: ids),
            try cpu.lastPositionLogits(ids: ids),
            "multi-chunk tiled prefill logits (C=4, prompt 10)")
        XCTAssertEqual(gpu.cachedTokens, ids)
    }

    /// The tiled (batchM > 8) GEMM kernel in-pipeline: one chunk of 10
    /// positions (C=16) gates at the same full-stack constant.
    func testLargeSingleChunkPrefillUsesTiledGemmAndMatchesOracle() throws {
        let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let gpu = try makeTiledModel(chunkSize: 16)
        let cpu = try makeCPUModel()
        assertFullStack(
            try gpu.lastPositionLogits(ids: ids),
            try cpu.lastPositionLogits(ids: ids),
            "single-chunk tiled prefill logits (C=16, prompt 10, tiled GEMM)")
    }

    // MARK: - Edge 5: KV cache contents after batched prefill

    /// Every prompt position's cache K/V vs the CPU-quant fp32 reference
    /// rounded to fp16, at the norm-species constant max(2⁻⁸·M, 2⁻¹¹) —
    /// the span from the residual stream includes the block norm (k-side
    /// additionally qk-norm + rope). Multi-chunk so both sides of a
    /// boundary are covered.
    func testKVCacheAfterBatchedPrefillMatchesCPUOracleAtEveryPosition() throws {
        let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let gpu = try makeTiledModel(chunkSize: 4)
        _ = try gpu.lastPositionLogits(ids: ids)

        let cpu = try makeCPUModel()
        let config = try tinyConfig()
        let hidden = config.hiddenSize
        let headDim = config.headDim
        let kvHeads = config.numKeyValueHeads
        let kvDim = kvHeads * headDim
        let attention = cpu.blocks[0].attention

        // CPU fp32 reference K/V: embed → input norm → k/v projections
        // (through the validated sgemm wrapper, hard rule 8) → per-head
        // k-norm + rope (FoldedKernelTests cpuNormRope arithmetic).
        let x = try cpu.embedding(ids)
        let normed = try cpu.blocks[0].inputNorm(x)
        let k = try BLAS.sgemm(
            a: normed, b: attention.kProjWeight,
            m: ids.count, k: hidden, n: kvDim, transposeB: true)
        let v = try BLAS.sgemm(
            a: normed, b: attention.vProjWeight,
            m: ids.count, k: hidden, n: kvDim, transposeB: true)
        let rope = attention.rope
        let half = headDim / 2

        for position in 0..<ids.count {
            for head in 0..<kvHeads {
                let base = position * kvDim + head * headDim
                var kRef = try attention.kNorm(
                    Array(k[base..<(base + headDim)]))
                for i in 0..<half {
                    let c = rope.cosValues[position * half + i]
                    let sn = rope.sinValues[position * half + i]
                    let x1 = kRef[i]
                    let x2 = kRef[half + i]
                    kRef[i] = x1 * c - x2 * sn
                    kRef[half + i] = x2 * c + x1 * sn
                }
                let kExpected = kRef.map { Float(Float16($0)) }
                let kGot = try readCacheSlot(
                    gpu, component: .key, head: head, position: position)
                    .map(Float.init)
                let kTol = max(
                    exp2(-8) * (kExpected.map(abs).max() ?? 0), exp2(-11))
                for i in 0..<headDim {
                    XCTAssertLessThanOrEqual(
                        abs(kGot[i] - kExpected[i]), kTol,
                        "k slot position \(position) head \(head) dim \(i)")
                }

                let vExpected = v[base..<(base + headDim)]
                    .map { Float(Float16($0)) }
                let vGot = try readCacheSlot(
                    gpu, component: .value, head: head, position: position)
                    .map(Float.init)
                let vTol = max(
                    exp2(-8) * (vExpected.map(abs).max() ?? 0), exp2(-11))
                for i in 0..<headDim {
                    XCTAssertLessThanOrEqual(
                        abs(vGot[i] - vExpected[i]), vTol,
                        "v slot position \(position) head \(head) dim \(i)")
                }
            }
        }
    }

    /// v-side append is a bitwise COPY of the v-projection GEMM output
    /// (single-chunk prompt so the scratch still holds every row).
    func testVSideAppendIsBitwiseCopyOfProjectionOutput() throws {
        let ids = [1, 2, 3, 4, 5]
        let gpu = try makeTiledModel(chunkSize: 8)
        _ = try gpu.lastPositionLogits(ids: ids)

        let scratch = try XCTUnwrap(gpu.prefillScratch)
        let config = try tinyConfig()
        let kvDim = config.numKeyValueHeads * config.headDim
        let vRows = scratch.vBatch.contents()
            .withMemoryRebound(to: Float16.self, capacity: ids.count * kvDim) {
                Array(UnsafeBufferPointer(start: $0, count: ids.count * kvDim))
            }
        for position in 0..<ids.count {
            for head in 0..<config.numKeyValueHeads {
                let base = position * kvDim + head * config.headDim
                let slot = try readCacheSlot(
                    gpu, component: .value, head: head, position: position)
                XCTAssertEqual(
                    slot.map(\.bitPattern),
                    vRows[base..<(base + config.headDim)].map(\.bitPattern),
                    "v append must be a bitwise copy (position \(position), "
                    + "head \(head))")
            }
        }
    }

    // MARK: - Edge 6: ragged/short prompts

    /// prompt == C exactly (one full chunk) and prompt < C (one partial
    /// chunk) both gate against the oracle.
    func testExactAndPartialChunkPromptsMatchOracle() throws {
        let cpu = try makeCPUModel()
        for (ids, chunk) in [([1, 2, 3, 4], 4), ([1, 2, 3], 8)] {
            let gpu = try makeTiledModel(chunkSize: chunk)
            assertFullStack(
                try gpu.lastPositionLogits(ids: ids),
                try cpu.lastPositionLogits(ids: ids),
                "prompt \(ids.count) at C=\(chunk)")
        }
    }

    /// A single-token prompt routes through the UNCHANGED sequential step
    /// (decode is untouched, spec scope): bitwise-identical logits to a
    /// sequential-prefill model and the fused per-step dispatch pin (10).
    func testSingleTokenPromptRoutesThroughSequentialStep() throws {
        let tiled = try makeTiledModel()
        let sequential = try makeSequentialModel()
        let tiledLogits = try tiled.lastPositionLogits(ids: [3])
        let sequentialLogits = try sequential.lastPositionLogits(ids: [3])
        XCTAssertEqual(
            tiledLogits, sequentialLogits,
            "single-token prompts must take the per-token decode path")
        XCTAssertEqual(tiled.lastStepDispatchCount, 10,
                       "the fused per-step pin — not a chunk count")
        XCTAssertEqual(tiled.lastCallSpan?.stepCount, 1)
    }

    // MARK: - Edge 7: last-position-only lm_head + dispatch pins

    /// Measured per-chunk dispatch pins (P2-5 rule), 1 layer, C=4,
    /// prompt 5 → chunks of 4 and 1. The layer's fixed cost is 13
    /// dispatches (norm, q/k/v GEMMs, cluster, o GEMM, residual, norm,
    /// gate, up, swiglu, down, residual) plus 2 SDPA dispatches per
    /// position:
    ///   chunk 1 (no logits): gather 1 + layer (13 + 2·4) = 22
    ///   chunk 2 (logits):    gather 1 + layer (13 + 2·1) + copy-row 1 +
    ///                        final norm 1 + lm_head 1 = 19
    /// Span total 41. The lm_head appears exactly once (last chunk) —
    /// logits are computed once per prefill (spec D2).
    func testDispatchCountsPinnedPerChunkAndLogitsComputedOnce() throws {
        let model = try makeTiledModel(chunkSize: 4)
        _ = try model.lastPositionLogits(ids: [1, 2, 3, 4, 5])
        XCTAssertEqual(model.lastStepDispatchCount, 18,
                       "last chunk: 1 gather + 15 layer + 3 logits tail")
        let span = try XCTUnwrap(model.lastCallSpan)
        XCTAssertEqual(span.dispatchCount, 33, "15 (chunk 1) + 18 (chunk 2)")
        XCTAssertEqual(span.stepCount, 5, "span accounts prompt tokens")
    }

    /// The selecting form adds exactly the argmax dispatch to the last
    /// chunk, and the chosen token obeys the exact-equality contract
    /// (`Argmax.firstIndex` of the same prompt's logits — the twin model
    /// computes bitwise-identical logits by chunk-partition determinism).
    func testSelectingPrefillAddsOneDispatchAndMatchesCPUArgmax() throws {
        let logitsModel = try makeTiledModel(chunkSize: 4)
        let logits = try logitsModel.lastPositionLogits(ids: [1, 2, 3, 4, 5])

        let selecting = try makeTiledModel(chunkSize: 4)
        let token = try selecting.nextGreedyToken(ids: [1, 2, 3, 4, 5])
        XCTAssertEqual(token, Argmax.firstIndex(logits))
        XCTAssertEqual(selecting.lastStepDispatchCount, 19,
                       "last chunk gains exactly the argmax dispatch")
        XCTAssertEqual(selecting.lastCallSpan?.dispatchCount, 34)
    }

    // MARK: - Edge 8: decode handoff

    /// Teacher-forced continuation after a tiled prefill: each subsequent
    /// single-token step (the unchanged fused decode reading the
    /// prefill-written cache) gates against the CPU-quant oracle.
    func testTeacherForcedContinuationAfterTiledPrefillGatesVsOracle() throws {
        let gpu = try makeTiledModel(chunkSize: 4)
        let cpu = try makeCPUModel()
        var ids = [1, 2, 3, 4, 5]
        assertFullStack(
            try gpu.lastPositionLogits(ids: ids),
            try cpu.lastPositionLogits(ids: ids),
            "prefill logits")
        for step in 0..<3 {
            ids.append([6, 7, 8][step])
            assertFullStack(
                try gpu.lastPositionLogits(ids: ids),
                try cpu.lastPositionLogits(ids: ids),
                "teacher-forced decode step \(step) after tiled prefill")
        }
    }

    /// The production token loop over a tiled model is token-identical to
    /// the logits loop (the P4-8 contract riding the tiled prefill).
    func testGenerateTokensMatchesGenerateOnTiledModel() throws {
        let viaLogits = try makeTiledModel(chunkSize: 4)
        let viaTokens = try makeTiledModel(chunkSize: 4)
        let reference = try DecodeLoop(model: viaLogits, maxContext: 16)
            .generate(promptIds: [1, 2, 3, 4, 5], maxNewTokens: 6)
        let tokens = try DecodeLoop(model: viaTokens, maxContext: 16)
            .generateTokens(promptIds: [1, 2, 3, 4, 5], maxNewTokens: 6)
        XCTAssertEqual(tokens, reference)
    }

    // MARK: - Edge 10: edge behavior unchanged

    func testEmptyPromptErrorIdenticalOnTiledPath() throws {
        let tiled = try makeTiledModel()
        let sequential = try makeSequentialModel()
        var tiledDetail: String?
        var sequentialDetail: String?
        XCTAssertThrowsError(try tiled.lastPositionLogits(ids: [])) {
            if case ModelError.badInput(let d) = $0 { tiledDetail = d }
        }
        XCTAssertThrowsError(try sequential.lastPositionLogits(ids: [])) {
            if case ModelError.badInput(let d) = $0 { sequentialDetail = d }
        }
        XCTAssertNotNil(tiledDetail)
        XCTAssertEqual(tiledDetail, sequentialDetail)
    }

    /// A prompt beyond maxContext throws the SAME `.contextFull` payload
    /// the sequential path eventually throws (position == maxContext) —
    /// on the tiled path before any dispatch or cache write.
    func testOverlongPromptErrorPayloadIdenticalToSequential() throws {
        let ids = [1, 2, 3, 4, 5, 6, 7, 8]
        let tiled = try makeTiledModel(maxContext: 6, chunkSize: 4)
        let sequential = try makeSequentialModel(maxContext: 6)
        var tiledError: KVCacheError?
        var sequentialError: KVCacheError?
        XCTAssertThrowsError(try tiled.lastPositionLogits(ids: ids)) {
            tiledError = $0 as? KVCacheError
        }
        XCTAssertThrowsError(try sequential.lastPositionLogits(ids: ids)) {
            sequentialError = $0 as? KVCacheError
        }
        XCTAssertEqual(tiledError, .contextFull(position: 6, maxContext: 6))
        XCTAssertEqual(tiledError, sequentialError)
        XCTAssertTrue(tiled.cachedTokens.isEmpty,
                      "tiled path validates up front — no partial cache")
    }

    /// A prompt exactly at the context boundary succeeds and gates.
    func testPromptAtContextBoundaryMatchesOracle() throws {
        let ids = [1, 2, 3, 4, 5, 6]
        let tiled = try makeTiledModel(maxContext: 6, chunkSize: 4)
        let cpu = try makeCPUModel(maxSequenceLength: 6)
        assertFullStack(
            try tiled.lastPositionLogits(ids: ids),
            try cpu.lastPositionLogits(ids: ids),
            "prompt at the 6-token boundary (chunks 4+2)")
    }

    func testTokenOutOfRangeErrorIdenticalOnTiledPath() throws {
        let tiled = try makeTiledModel()
        let sequential = try makeSequentialModel()
        var tiledPayload: (Int, Int)?
        var sequentialPayload: (Int, Int)?
        XCTAssertThrowsError(try tiled.lastPositionLogits(ids: [1, 99])) {
            if case ModelError.tokenIdOutOfRange(let id, let vocab) = $0 {
                tiledPayload = (id, vocab)
            }
        }
        XCTAssertThrowsError(try sequential.lastPositionLogits(ids: [1, 99])) {
            if case ModelError.tokenIdOutOfRange(let id, let vocab) = $0 {
                sequentialPayload = (id, vocab)
            }
        }
        XCTAssertEqual(tiledPayload?.0, 99)
        XCTAssertEqual(tiledPayload?.1, 32)
        XCTAssertEqual(tiledPayload?.0, sequentialPayload?.0)
        XCTAssertEqual(tiledPayload?.1, sequentialPayload?.1)
    }

    /// Load-time rejects: tiled + naive kernel path, and out-of-range
    /// chunk sizes, fail loudly at load (never mid-prompt). The bf16
    /// initializer exposes no prefill option at all — its models report
    /// `.sequential` (spec D5: bf16 keeps sequential permanently).
    func testTiledLoadValidationRejectsAndBF16StaysSequential() throws {
        let context = try makeContextOrSkip()
        let packed = try makePackedCheckpoint()

        XCTAssertThrowsError(
            try GPUModel(
                packed: packed, config: try tinyConfig(), context: context,
                maxContext: 16, kernelPath: .naive, prefillPath: .tiled)
        ) { error in
            guard case ModelError.badInput(let detail) = error else {
                return XCTFail("expected badInput, got \(error)")
            }
            XCTAssertTrue(detail.contains("fused"), detail)
        }
        for badChunk in [0, -1, 17] {
            XCTAssertThrowsError(
                try GPUModel(
                    packed: packed, config: try tinyConfig(), context: context,
                    maxContext: 16, prefillPath: .tiled,
                    prefillChunkSize: badChunk)
            ) { error in
                guard case ModelError.badInput(let detail) = error else {
                    return XCTFail("expected badInput, got \(error)")
                }
                XCTAssertTrue(detail.contains("prefillChunkSize"), detail)
            }
        }

        let source = try makeSourceFile(tensors: tinyTensors())
        let bf16 = try GPUModel(
            checkpoint: try SafetensorsFile(path: source),
            config: try tinyConfig(), context: context, maxContext: 16)
        XCTAssertEqual(bf16.prefillPath, .sequential)

        // P5-4 (spec D5): TILED is the packed-pipeline default. An
        // unspecified chunk size resolves to min(default C, maxContext) so
        // the scratch is never sized beyond what a prompt can occupy; the
        // resolved value is what `prefillChunkSize` reports (rows record
        // it, spec D2).
        let defaulted = try GPUModel(
            packed: packed, config: try tinyConfig(), context: context,
            maxContext: 16)
        XCTAssertEqual(defaulted.prefillPath, .tiled,
                       "P5-4: tiled is the packed-pipeline default")
        XCTAssertEqual(defaulted.prefillChunkSize, 16,
                       "unspecified C resolves to min(512, maxContext)")
        XCTAssertNotNil(defaulted.prefillScratch,
                        "the default (tiled) model preallocates its scratch")
        let wide = try GPUModel(
            packed: packed, config: try tinyConfig(), context: context,
            maxContext: 1024)
        XCTAssertEqual(wide.prefillChunkSize, GPUModel.defaultPrefillChunkSize,
                       "a context ≥ 512 keeps the measured default C")

        // The naive kernel arm (kept for the P4-5/P5-5 A/B rows) supports
        // sequential prefill only, so an UNSPECIFIED prefill path resolves
        // to `.sequential` there — the naive arm keeps loading. Asking for
        // tiled + naive explicitly still fails at load (pinned above).
        let naiveDefault = try GPUModel(
            packed: packed, config: try tinyConfig(), context: context,
            maxContext: 16, kernelPath: .naive)
        XCTAssertEqual(naiveDefault.prefillPath, .sequential)
        XCTAssertNil(naiveDefault.prefillScratch,
                     "sequential models allocate no prefill scratch")

        // Explicit sequential on the fused arm stays selectable (D5: the
        // P5-5 in-session before/after row).
        XCTAssertEqual(try makeSequentialModel().prefillPath, .sequential)
    }

    // MARK: - Edge 11: scratch preallocation

    /// Multi-chunk prefills reuse the exact load-time buffers (identity),
    /// and a repeat prefill allocates NOTHING new on the device (the
    /// SDPA partial-state scratch is warmed by the first prefill and
    /// reused thereafter).
    func testScratchBuffersStableAndNoAllocationGrowthAcrossPrefills() throws {
        let context = try makeContextOrSkip()
        let model = try GPUModel(
            packed: try makePackedCheckpoint(), config: try tinyConfig(),
            context: context, maxContext: 16,
            prefillPath: .tiled, prefillChunkSize: 4)
        let scratch = try XCTUnwrap(model.prefillScratch)
        let before = scratch.allBuffers.map(ObjectIdentifier.init)

        let ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        _ = try model.lastPositionLogits(ids: ids)
        let warmed = context.device.currentAllocatedSize

        model.reset()
        _ = try model.lastPositionLogits(ids: ids)
        let after = try XCTUnwrap(model.prefillScratch).allBuffers
            .map(ObjectIdentifier.init)
        XCTAssertEqual(before, after, "scratch buffers are load-time and stable")
        XCTAssertEqual(
            context.device.currentAllocatedSize, warmed,
            "a repeat multi-chunk prefill must allocate no new device memory")
    }

    /// The spec D2 budget: prefill scratch at the PINNED model dims and
    /// the default C=512 is ≤ 64 MiB (exact byte pin so growth is loud).
    func testScratchBudgetAtPinnedDimsWithinSpecBudget() throws {
        let pinned = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        let bytes = GPUModel.prefillScratchBytes(
            config: pinned, chunkSize: GPUModel.defaultPrefillChunkSize)
        XCTAssertEqual(bytes, 35_653_632,
                       "512·(4 + 8·2048 + 6·2048 + 4·1024 + 6·6144) bytes")
        XCTAssertLessThanOrEqual(bytes, 64 * 1024 * 1024)
    }

    // MARK: - D1 span fields on the tiled path

    func testTiledPrefillSpanFieldsSane() throws {
        let model = try makeTiledModel(chunkSize: 4)
        _ = try model.lastPositionLogits(ids: [1, 2, 3, 4, 5])
        let span = try XCTUnwrap(model.lastCallSpan)
        XCTAssertEqual(span.stepCount, 5)
        XCTAssertGreaterThan(span.gpuSeconds, 0)
        XCTAssertGreaterThanOrEqual(span.wallSeconds, span.gpuSeconds)
        XCTAssertEqual(span.dispatchCount, 33)
    }

    /// A tiled call that throws in validation leaves NO stale span behind
    /// (the P5-1 contract — cleared at call entry, sequential parity).
    func testFailedTiledCallLeavesNoStaleSpan() throws {
        let model = try makeTiledModel(chunkSize: 4)
        _ = try model.lastPositionLogits(ids: [1, 2, 3, 4, 5])
        XCTAssertNotNil(model.lastCallSpan)
        model.reset()
        XCTAssertThrowsError(try model.lastPositionLogits(ids: [1, 99]))
        XCTAssertNil(model.lastCallSpan,
                     "a failed tiled call must not leave the previous span")
    }

    // MARK: - Edge 9: prefill-path toggle (P5-4)

    /// Both prefill paths load on the packed + fused pipeline — tiled via
    /// the DEFAULT, sequential via the explicit option — and both pass the
    /// shared full-stack spot check against the same CPU-quant oracle at
    /// the pre-committed constant (Tier-E-shape; spec D6: tiled-vs-
    /// sequential bitwise equality is NOT required). DispatchCounter tells
    /// the paths apart on the same prompt: sequential runs the per-token
    /// fused structure (4 × 8 no-logits steps + one 10-dispatch logits
    /// step = 42), tiled runs ONE chunk (gather 1 + 14 per layer + logits
    /// tail 3 = 18; measured pins, P2-5 rule — 27 at P5-3, before PF-1's
    /// one-dispatch batched SDPA).
    func testBothPrefillPathsLoadPassSpotCheckAndCounterDistinguishes() throws {
        let tiled = try makeDefaultModel()
        let sequential = try makeSequentialModel()
        XCTAssertEqual(tiled.prefillPath, .tiled)
        XCTAssertEqual(sequential.prefillPath, .sequential)
        XCTAssertEqual(tiled.kernelPath, sequential.kernelPath,
                       "the toggle changes prompt processing only")

        let ids = [1, 2, 3, 4, 5]
        let ref = try makeCPUModel().lastPositionLogits(ids: ids)
        let tiledLogits = try tiled.lastPositionLogits(ids: ids)
        let sequentialLogits = try sequential.lastPositionLogits(ids: ids)
        assertFullStack(tiledLogits, ref, "tiled default vs CPU-quant")
        assertFullStack(sequentialLogits, ref, "sequential vs CPU-quant")

        let tiledSpan = try XCTUnwrap(tiled.lastCallSpan)
        let sequentialSpan = try XCTUnwrap(sequential.lastCallSpan)
        XCTAssertEqual(tiledSpan.stepCount, 5)
        XCTAssertEqual(sequentialSpan.stepCount, 5)
        XCTAssertEqual(sequentialSpan.dispatchCount, 4 * 8 + 10)
        XCTAssertEqual(tiledSpan.dispatchCount, 18)
        XCTAssertNotEqual(tiledSpan.dispatchCount, sequentialSpan.dispatchCount,
                          "DispatchCounter distinguishes the paths")

        // Decode after the prompt is the SAME Phase 4 step on both paths.
        _ = try tiled.lastPositionLogits(ids: ids + [6])
        _ = try sequential.lastPositionLogits(ids: ids + [6])
        XCTAssertEqual(tiled.lastStepDispatchCount, 10)
        XCTAssertEqual(sequential.lastStepDispatchCount, 10)
    }

    // MARK: - Edge 12: instrumentation parity through the production runner

    /// Both paths report the SAME fields through `BenchGenerationRunner`:
    /// the engine-measured prefill span (D1 metric of record — wall ≥ GPU,
    /// per-engine token accounting, no WARM PREFIX label on a cold
    /// generation) AND the legacy TTFT-style field, with per-token decode
    /// records following. Only the dispatch count differs.
    func testRunnerReportsIdenticalPrefillFieldsOnBothPaths() throws {
        let ids = [1, 2, 3, 4, 5]
        var spans: [GPUModel.PrefillPath: PrefillSpan] = [:]
        for model in [try makeDefaultModel(), try makeSequentialModel()] {
            let metrics = try BenchGenerationRunner(
                gpuModel: model, maxContext: 16, eosTokenIds: []
            ).run(promptIds: ids, maxNewTokens: 3).metrics
            let prefill = try XCTUnwrap(
                metrics.prefillSpan, "\(model.prefillPath) span missing")
            XCTAssertEqual(prefill.promptTokenCount, 5)
            XCTAssertEqual(prefill.span.stepCount, 5,
                           "\(model.prefillPath): the span covers the prompt call only")
            XCTAssertGreaterThan(prefill.span.gpuSeconds, 0)
            XCTAssertGreaterThanOrEqual(
                prefill.span.wallSeconds, prefill.span.gpuSeconds)
            XCTAssertFalse(prefill.summaryLine.contains("WARM PREFIX"))
            XCTAssertNotNil(metrics.prefillSeconds,
                            "\(model.prefillPath): legacy TTFT-style field exports")
            XCTAssertEqual(metrics.generatedTokenCount, 3)
            XCTAssertEqual(metrics.timing?.tokenCount, 3)
            spans[model.prefillPath] = prefill
        }
        XCTAssertEqual(spans[.tiled]?.span.dispatchCount, 19,
                       "selecting form: the single chunk gains the argmax dispatch")
        XCTAssertEqual(spans[.sequential]?.span.dispatchCount, 4 * 8 + 11)
    }
}
