import XCTest
@testable import QwenMetalEngine
import Metal

/// P2-3 attention-kernel tests (docs/phases/phase-2.md D4 + enumerated edge
/// cases 1-5): kv-append, attn-scores, softmax, attn-pv diffed against the
/// CPU reference BEFORE any optimization exists (hard rule 3).
///
/// Gates are the pre-committed Phase 2 tolerances (DECISIONS.md 2026-08-23):
///   Exact:   kv-append readback — bitwise.
///   Tier K:  |Δ| <= max(2⁻⁹·M, 2⁻¹¹), M = max|ref| over the compared slice.
/// Matmul-shaped reference work (q·Kᵀ, probs·V) routes through BLAS.sgemm
/// (hard rule 8); softmax reference replicates the CPU module's fp32 formula.
final class AttentionKernelTests: XCTestCase {

    // MARK: - Deterministic RNG (SgemmTests pattern)

    private struct SplitMix64: RandomNumberGenerator {
        private var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    private func randomHalfs(
        count: Int, range: ClosedRange<Float> = -1...1, rng: inout SplitMix64
    ) -> [Float16] {
        (0..<count).map { _ in Float16(Float.random(in: range, using: &rng)) }
    }

    private func randomFloats(
        count: Int, range: ClosedRange<Float> = -1...1, rng: inout SplitMix64
    ) -> [Float] {
        (0..<count).map { _ in Float.random(in: range, using: &rng) }
    }

    // MARK: - GPU harness helpers (DecodeKernelTests pattern)

    private func makeContextOrSkip() throws -> MetalContext {
        do {
            return try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    private func makeBuffer<T>(_ device: MTLDevice, values: [T]) throws -> MTLBuffer {
        let length = values.count * MemoryLayout<T>.stride
        guard let buffer = values.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: length,
                              options: .storageModeShared)
        }) else {
            throw MetalHarnessError.bufferAllocationFailed(length: length)
        }
        return buffer
    }

    private func makeOutputBuffer(
        _ device: MTLDevice, count: Int, elementStride: Int
    ) throws -> MTLBuffer {
        let length = count * elementStride
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared)
        else {
            throw MetalHarnessError.bufferAllocationFailed(length: length)
        }
        return buffer
    }

    private func readHalfs(_ buffer: MTLBuffer, count: Int) -> [Float16] {
        [Float16](UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float16.self, capacity: count),
            count: count))
    }

    private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        [Float](UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: count),
            count: count))
    }

    private func readBits(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
        [UInt16](UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: UInt16.self, capacity: count),
            count: count))
    }

    private func fillBits(_ buffer: MTLBuffer, _ bits: UInt16) {
        let count = buffer.length / 2
        let ptr = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
        for i in 0..<count { ptr[i] = bits }
    }

    private func fillFloatBits(_ buffer: MTLBuffer, _ bits: UInt32) {
        let count = buffer.length / 4
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: count)
        for i in 0..<count { ptr[i] = bits }
    }

    /// Writes fp16 values straight into a cache slot (test-side ground truth,
    /// independent of the append kernel).
    private func writeCacheSlot(
        _ cache: KVCache, layer: Int, component: KVCache.Component,
        head: Int, position: Int, values: [Float16]
    ) throws {
        let offset = try cache.elementOffset(
            layer: layer, component: component, head: head, position: position)
        let ptr = cache.buffer.contents()
            .bindMemory(to: Float16.self, capacity: offset + values.count)
        for (i, v) in values.enumerated() { ptr[offset + i] = v }
    }

    /// The CPU reference's exact softmax formula (Attention.swift): scale-free
    /// max-subtracted fp32 softmax over one row.
    private func cpuSoftmax(_ row: [Float]) -> [Float] {
        var rowMax = -Float.infinity
        for v in row { rowMax = max(rowMax, v) }
        var sum: Float = 0
        let exps = row.map { v -> Float in
            let e = expf(v - rowMax)
            sum += e
            return e
        }
        return exps.map { $0 / sum }
    }

    // MARK: - Gate assertion (pre-committed Tier K; never loosened)

    private func assertTierK(
        _ got: [Float], _ ref: [Float], _ surface: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.count, ref.count, "\(surface): count mismatch",
                       file: file, line: line)
        let m = ref.reduce(Float(0)) { max($0, abs($1)) }
        let gate = max(exp2(-9) * m, exp2(-11))
        var worst: Float = 0
        var worstIndex = 0
        for i in 0..<ref.count {
            let delta = abs(got[i] - ref[i])
            if delta > worst {
                worst = delta
                worstIndex = i
            }
        }
        XCTAssertLessThanOrEqual(
            worst, gate,
            "\(surface): max |Δ| \(worst) at [\(worstIndex)] exceeds Tier-K gate "
                + "\(gate) (M = \(m))",
            file: file, line: line)
    }

    // MARK: - Edge case 1: kv-append writes exactly the addressed slot (EXACT)

    func testKVAppendWritesExactlyTheAddressedSlotAndNothingElse() throws {
        let context = try makeContextOrSkip()
        let kernels = try AttentionKernels(context: context)
        let (layers, kvHeads, maxContext, headDim) = (2, 3, 4, 2)
        let cache = try KVCache(
            device: context.device, layers: layers, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        let sentinel: UInt16 = 0xABAB
        fillBits(cache.buffer, sentinel)

        // Three appends at distinct (layer, component, position) slots. The
        // first vector carries fp16 boundary bit patterns (±0, subnormal,
        // ±inf, NaN payload) — a pure copy must preserve every bit.
        let appends: [(layer: Int, component: KVCache.Component, position: Int, bits: [UInt16])] = [
            (0, .key, 0, [0x0000, 0x8000, 0x0001, 0x7C00, 0xFC00, 0x7E01]),
            (1, .value, 3, (0..<kvHeads * headDim).map { UInt16(0x2200 + $0) }),
            (1, .key, 2, (0..<kvHeads * headDim).map { UInt16(0x3300 + $0) }),
        ]
        for a in appends {
            XCTAssertEqual(a.bits.count, kvHeads * headDim, "test premise: full vector")
            let vector = try makeBuffer(context.device, values: a.bits)
            try context.timedDispatch { encoder in
                try kernels.encodeKVAppend(
                    into: encoder, cache: cache, layer: a.layer,
                    component: a.component, position: a.position, vector: vector)
            }
        }

        // Exhaustive expected map over the whole buffer (small dims): every
        // written slot has the input bits, every other element is sentinel.
        let totalElements = cache.byteCount / 2
        var expected = [UInt16](repeating: sentinel, count: totalElements)
        for a in appends {
            for h in 0..<kvHeads {
                let offset = try cache.elementOffset(
                    layer: a.layer, component: a.component, head: h, position: a.position)
                for d in 0..<headDim {
                    expected[offset + d] = a.bits[h * headDim + d]
                }
            }
        }
        XCTAssertEqual(readBits(cache.buffer, count: totalElements), expected,
                       "kv-append must be bitwise exact and touch nothing else")
    }

    // MARK: - Edge case 5: context limit — last append succeeds, next stops cleanly

    func testContextLimitLastAppendSucceedsAndNextThrowsWithoutWriting() throws {
        let context = try makeContextOrSkip()
        let kernels = try AttentionKernels(context: context)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: 1, maxContext: 4, headDim: 2)
        fillBits(cache.buffer, 0x0000)

        // Positions 0..3 all succeed — the small-dims analog of "the 4096th
        // append succeeds" (position maxContext-1).
        for p in 0..<cache.maxContext {
            let vector = try makeBuffer(
                context.device, values: [UInt16(0x1000 + p), UInt16(0x2000 + p)])
            try context.timedDispatch { encoder in
                try kernels.encodeKVAppend(
                    into: encoder, cache: cache, layer: 0, component: .key,
                    position: p, vector: vector)
            }
        }
        let snapshot = readBits(cache.buffer, count: cache.byteCount / 2)

        // The next decode step's append throws BEFORE any dispatch — the
        // clean context-limit stop; no out-of-bounds write is possible.
        let vector = try makeBuffer(context.device, values: [UInt16](repeating: 0xFFFF, count: 2))
        for badPosition in [4, 5] {
            var thrown: Error?
            try context.timedDispatch { encoder in
                do {
                    try kernels.encodeKVAppend(
                        into: encoder, cache: cache, layer: 0, component: .key,
                        position: badPosition, vector: vector)
                } catch { thrown = error }
            }
            XCTAssertEqual(
                thrown as? KVCacheError,
                .contextFull(position: badPosition, maxContext: 4))
        }
        var thrown: Error?
        try context.timedDispatch { encoder in
            do {
                try kernels.encodeKVAppend(
                    into: encoder, cache: cache, layer: 0, component: .key,
                    position: -1, vector: vector)
            } catch { thrown = error }
        }
        XCTAssertEqual(thrown as? KVCacheError,
                       .indexOutOfRange(name: "position", value: -1, bound: 4))

        XCTAssertEqual(readBits(cache.buffer, count: cache.byteCount / 2), snapshot,
                       "failed appends must leave the cache untouched")
    }

    // MARK: - Edge case 2: GQA mapping — KV head h serves Q heads {2h, 2h+1}

    func testGQAMappingServesConsecutiveQHeadPairs() throws {
        let context = try makeContextOrSkip()
        let kernels = try AttentionKernels(context: context)
        // kvHeads=3, numHeads=6 (groupSize 2); headDim=4 makes scale = 1/2
        // exact, so score values identify the KV head EXACTLY. A
        // repeat-interleave mistake (kvHead = qHead % 3) or an off-by-one
        // (kvHead = (qHead+1)/2) reads a different K row and fails.
        let (kvHeads, numHeads, headDim, maxContext) = (3, 6, 4, 2)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        fillBits(cache.buffer, 0x0000)
        // K[h, position 0] = [h+1, 0, 0, 0] — the first component names the head.
        for h in 0..<kvHeads {
            var row = [Float16](repeating: 0, count: headDim)
            row[0] = Float16(h + 1)
            try writeCacheSlot(cache, layer: 0, component: .key, head: h,
                               position: 0, values: row)
        }
        // q[qHead] = [1, 0, 0, 0] for every Q head.
        var q = [Float16](repeating: 0, count: numHeads * headDim)
        for h in 0..<numHeads { q[h * headDim] = 1 }
        let qBuffer = try makeBuffer(context.device, values: q)
        let scores = try makeOutputBuffer(
            context.device, count: numHeads * maxContext, elementStride: 4)
        fillFloatBits(scores, 0xDEAD_BEEF)

        try context.timedDispatch { encoder in
            try kernels.encodeAttentionScores(
                into: encoder, cache: cache, layer: 0, position: 0,
                query: qBuffer, numHeads: numHeads, scores: scores)
        }

        let got = readFloats(scores, count: numHeads * maxContext)
        for qHead in 0..<numHeads {
            let expectedKVHead = qHead / 2
            XCTAssertEqual(
                got[qHead * maxContext], Float(expectedKVHead + 1) / 2,
                "Q head \(qHead) must read KV head \(expectedKVHead)")
            // Columns beyond `position` stay untouched.
            XCTAssertEqual(got[qHead * maxContext + 1].bitPattern, 0xDEAD_BEEF,
                           "score column beyond position must be untouched")
        }
    }

    // MARK: - Edge case 4: empty cache (p = 0) == single-token attention

    func testEmptyCacheDecodeEqualsSingleTokenAttention() throws {
        let context = try makeContextOrSkip()
        let kernels = try AttentionKernels(context: context)
        let (kvHeads, numHeads, headDim, maxContext) = (2, 4, 6, 3)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        var rng = SplitMix64(seed: 70)
        let k0 = randomHalfs(count: kvHeads * headDim, rng: &rng)
        let v0 = randomHalfs(count: kvHeads * headDim, rng: &rng)
        let q = randomHalfs(count: numHeads * headDim, rng: &rng)

        let kBuffer = try makeBuffer(context.device, values: k0)
        let vBuffer = try makeBuffer(context.device, values: v0)
        let qBuffer = try makeBuffer(context.device, values: q)
        let scores = try makeOutputBuffer(
            context.device, count: numHeads * maxContext, elementStride: 4)
        let probs = try makeOutputBuffer(
            context.device, count: numHeads * maxContext, elementStride: 4)
        let out = try makeOutputBuffer(
            context.device, count: numHeads * headDim, elementStride: 2)

        try context.timedDispatch { encoder in
            try kernels.encodeKVAppend(
                into: encoder, cache: cache, layer: 0, component: .key,
                position: 0, vector: kBuffer)
            try kernels.encodeKVAppend(
                into: encoder, cache: cache, layer: 0, component: .value,
                position: 0, vector: vBuffer)
            try kernels.encodeAttentionScores(
                into: encoder, cache: cache, layer: 0, position: 0,
                query: qBuffer, numHeads: numHeads, scores: scores)
            try kernels.encodeSoftmaxRows(
                into: encoder, input: scores, rows: numHeads, count: 1,
                rowStride: maxContext, output: probs)
            try kernels.encodeAttentionPV(
                into: encoder, cache: cache, layer: 0, position: 0,
                probs: probs, numHeads: numHeads, output: out)
        }

        // Softmax over one score is exactly 1.0, and 1.0 · V[0] round-trips
        // fp32→fp16 bit-exactly: the output IS the mapped V row.
        let gotProbs = readFloats(probs, count: numHeads * maxContext)
        for qHead in 0..<numHeads {
            XCTAssertEqual(gotProbs[qHead * maxContext], 1.0,
                           "softmax over a single score must be exactly 1")
        }
        let gotBits = readHalfs(out, count: numHeads * headDim).map(\.bitPattern)
        var expectedBits = [UInt16]()
        for qHead in 0..<numHeads {
            let kvHead = qHead / (numHeads / kvHeads)
            expectedBits += v0[(kvHead * headDim)..<((kvHead + 1) * headDim)]
                .map(\.bitPattern)
        }
        XCTAssertEqual(gotBits, expectedBits,
                       "p=0 attention output must equal the V row bitwise")
    }

    // MARK: - Edge case 3: decode step at p > 0 vs CPU full recompute (chain)

    func testDecodeStepAtLaterPositionMatchesCPUFullRecompute() throws {
        let context = try makeContextOrSkip()
        let kernels = try AttentionKernels(context: context)
        // Layer 1 of 2 — a base-offset bug lands in layer 0's (sentinel) slab
        // and fails the diff. One command buffer for the whole chain, the
        // P2-4 shape (spec D5).
        let (layers, kvHeads, numHeads, headDim, maxContext) = (2, 2, 4, 8, 8)
        let groupSize = numHeads / kvHeads
        let position = 4 // decode step p, with positions 0..4 in the cache
        let cache = try KVCache(
            device: context.device, layers: layers, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        fillBits(cache.buffer, 0xABAB)

        var rng = SplitMix64(seed: 80)
        let ks = (0...position).map { _ in randomHalfs(count: kvHeads * headDim, rng: &rng) }
        let vs = (0...position).map { _ in randomHalfs(count: kvHeads * headDim, rng: &rng) }
        let q = randomHalfs(count: numHeads * headDim, rng: &rng)

        let kBuffers = try ks.map { try makeBuffer(context.device, values: $0) }
        let vBuffers = try vs.map { try makeBuffer(context.device, values: $0) }
        let qBuffer = try makeBuffer(context.device, values: q)
        let scores = try makeOutputBuffer(
            context.device, count: numHeads * maxContext, elementStride: 4)
        let probs = try makeOutputBuffer(
            context.device, count: numHeads * maxContext, elementStride: 4)
        let out = try makeOutputBuffer(
            context.device, count: numHeads * headDim, elementStride: 2)

        try context.timedDispatch { encoder in
            for j in 0...position {
                try kernels.encodeKVAppend(
                    into: encoder, cache: cache, layer: 1, component: .key,
                    position: j, vector: kBuffers[j])
                try kernels.encodeKVAppend(
                    into: encoder, cache: cache, layer: 1, component: .value,
                    position: j, vector: vBuffers[j])
            }
            try kernels.encodeAttentionScores(
                into: encoder, cache: cache, layer: 1, position: position,
                query: qBuffer, numHeads: numHeads, scores: scores)
            try kernels.encodeSoftmaxRows(
                into: encoder, input: scores, rows: numHeads, count: position + 1,
                rowStride: maxContext, output: probs)
            try kernels.encodeAttentionPV(
                into: encoder, cache: cache, layer: 1, position: position,
                probs: probs, numHeads: numHeads, output: out)
        }

        // CPU reference: full recompute of the causal attention row at p from
        // the same fp16 inputs, fp32 throughout, sgemm for the matmul-shaped
        // parts (hard rule 8), the reference softmax formula in between.
        let scale = 1 / Float(headDim).squareRoot()
        var ref = [Float](repeating: 0, count: numHeads * headDim)
        for qHead in 0..<numHeads {
            let kvHead = qHead / groupSize
            let qRow = q[(qHead * headDim)..<((qHead + 1) * headDim)].map(Float.init)
            var kRows = [Float]()
            var vRows = [Float]()
            for j in 0...position {
                kRows += ks[j][(kvHead * headDim)..<((kvHead + 1) * headDim)].map(Float.init)
                vRows += vs[j][(kvHead * headDim)..<((kvHead + 1) * headDim)].map(Float.init)
            }
            let rawScores = try BLAS.sgemm(
                a: qRow, b: kRows, m: 1, k: headDim, n: position + 1, transposeB: true)
            let probsRow = cpuSoftmax(rawScores.map { $0 * scale })
            let contextRow = try BLAS.sgemm(
                a: probsRow, b: vRows, m: 1, k: position + 1, n: headDim)
            ref.replaceSubrange(
                (qHead * headDim)..<((qHead + 1) * headDim), with: contextRow)
        }
        assertTierK(readHalfs(out, count: numHeads * headDim).map(Float.init), ref,
                    "decode step at position \(position), full kernel chain")
    }

    // MARK: - Tier K: attn-scores on odd shapes vs sgemm

    func testAttentionScoresMatchSgemmOnOddShapes() throws {
        let context = try makeContextOrSkip()
        let kernels = try AttentionKernels(context: context)
        let (layers, kvHeads, numHeads, headDim, maxContext) = (2, 3, 6, 19, 11)
        let position = 6
        let cache = try KVCache(
            device: context.device, layers: layers, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        fillBits(cache.buffer, 0xABAB)

        // K written straight into layer 1's slab (independent of the append
        // kernel — each kernel gets its own oracle).
        var rng = SplitMix64(seed: 90)
        var kRowsPerHead = [[Float16]]()
        for h in 0..<kvHeads {
            var rows = [Float16]()
            for p in 0...position {
                let row = randomHalfs(count: headDim, rng: &rng)
                try writeCacheSlot(cache, layer: 1, component: .key, head: h,
                                   position: p, values: row)
                rows += row
            }
            kRowsPerHead.append(rows)
        }
        let q = randomHalfs(count: numHeads * headDim, rng: &rng)
        let qBuffer = try makeBuffer(context.device, values: q)
        let scores = try makeOutputBuffer(
            context.device, count: numHeads * maxContext, elementStride: 4)
        fillFloatBits(scores, 0xDEAD_BEEF)

        try context.timedDispatch { encoder in
            try kernels.encodeAttentionScores(
                into: encoder, cache: cache, layer: 1, position: position,
                query: qBuffer, numHeads: numHeads, scores: scores)
        }

        let scale = 1 / Float(headDim).squareRoot()
        let got = readFloats(scores, count: numHeads * maxContext)
        var gotAsserted = [Float]()
        var ref = [Float]()
        for qHead in 0..<numHeads {
            let kvHead = qHead / (numHeads / kvHeads)
            let qRow = q[(qHead * headDim)..<((qHead + 1) * headDim)].map(Float.init)
            ref += try BLAS.sgemm(
                a: qRow, b: kRowsPerHead[kvHead].map(Float.init),
                m: 1, k: headDim, n: position + 1, transposeB: true
            ).map { $0 * scale }
            gotAsserted += got[(qHead * maxContext)..<(qHead * maxContext + position + 1)]
            for j in (position + 1)..<maxContext {
                XCTAssertEqual(got[qHead * maxContext + j].bitPattern, 0xDEAD_BEEF,
                               "score column \(j) beyond position must be untouched")
            }
        }
        assertTierK(gotAsserted, ref, "attn-scores \(numHeads)×\(position + 1), headDim \(headDim)")
    }

    // MARK: - Tier K: softmax rows vs the CPU formula

    func testSoftmaxRowsMatchesCPUAndLeavesTailUntouched() throws {
        let context = try makeContextOrSkip()
        let kernels = try AttentionKernels(context: context)
        let (rows, count, rowStride) = (5, 7, 9)
        var rng = SplitMix64(seed: 100)
        // ±8 exercises the max-subtraction across a wide dynamic range.
        var input = [Float](repeating: 0, count: rows * rowStride)
        for r in 0..<rows {
            let row = randomFloats(count: count, range: -8...8, rng: &rng)
            input.replaceSubrange(
                (r * rowStride)..<(r * rowStride + count), with: row)
        }
        let inputBuffer = try makeBuffer(context.device, values: input)
        let output = try makeOutputBuffer(
            context.device, count: rows * rowStride, elementStride: 4)
        fillFloatBits(output, 0xDEAD_BEEF)

        try context.timedDispatch { encoder in
            try kernels.encodeSoftmaxRows(
                into: encoder, input: inputBuffer, rows: rows, count: count,
                rowStride: rowStride, output: output)
        }

        let got = readFloats(output, count: rows * rowStride)
        var gotAsserted = [Float]()
        var ref = [Float]()
        for r in 0..<rows {
            let base = r * rowStride
            ref += cpuSoftmax(Array(input[base..<(base + count)]))
            gotAsserted += got[base..<(base + count)]
            for j in count..<rowStride {
                XCTAssertEqual(got[base + j].bitPattern, 0xDEAD_BEEF,
                               "softmax output beyond count must be untouched")
            }
            // Each row sums to ~1 — a direct sanity of the normalization.
            let rowSum = got[base..<(base + count)].reduce(0, +)
            XCTAssertEqual(rowSum, 1.0, accuracy: 1e-5)
        }
        assertTierK(gotAsserted, ref, "softmax \(rows)×\(count) stride \(rowStride)")
    }

    // MARK: - Tier K: attn-pv on odd shapes vs sgemm

    func testAttentionPVMatchesSgemmOnOddShapes() throws {
        let context = try makeContextOrSkip()
        let kernels = try AttentionKernels(context: context)
        let (kvHeads, numHeads, headDim, maxContext) = (2, 4, 13, 9)
        let position = 5
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        fillBits(cache.buffer, 0xABAB)

        var rng = SplitMix64(seed: 110)
        var vRowsPerHead = [[Float16]]()
        for h in 0..<kvHeads {
            var rows = [Float16]()
            for p in 0...position {
                let row = randomHalfs(count: headDim, rng: &rng)
                try writeCacheSlot(cache, layer: 0, component: .value, head: h,
                                   position: p, values: row)
                rows += row
            }
            vRowsPerHead.append(rows)
        }
        // Arbitrary (not necessarily normalized) fp32 weights — the kernel is
        // a plain weighted sum; softmax semantics are tested separately.
        var probs = [Float](repeating: 0, count: numHeads * maxContext)
        for qHead in 0..<numHeads {
            let row = randomFloats(count: position + 1, range: 0...1, rng: &rng)
            probs.replaceSubrange(
                (qHead * maxContext)..<(qHead * maxContext + position + 1), with: row)
        }
        let probsBuffer = try makeBuffer(context.device, values: probs)
        let out = try makeOutputBuffer(
            context.device, count: numHeads * headDim, elementStride: 2)

        try context.timedDispatch { encoder in
            try kernels.encodeAttentionPV(
                into: encoder, cache: cache, layer: 0, position: position,
                probs: probsBuffer, numHeads: numHeads, output: out)
        }

        var ref = [Float]()
        for qHead in 0..<numHeads {
            let kvHead = qHead / (numHeads / kvHeads)
            let probsRow = Array(
                probs[(qHead * maxContext)..<(qHead * maxContext + position + 1)])
            ref += try BLAS.sgemm(
                a: probsRow, b: vRowsPerHead[kvHead].map(Float.init),
                m: 1, k: position + 1, n: headDim)
        }
        assertTierK(readHalfs(out, count: numHeads * headDim).map(Float.init), ref,
                    "attn-pv \(numHeads)×\(headDim), \(position + 1) positions")
    }

    // MARK: - Loud rejection of bad inputs (no dispatch reaches the GPU)

    func testAttentionEncodersRejectBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernels = try AttentionKernels(context: context)
        let cache = try KVCache(
            device: context.device, layers: 2, kvHeads: 2, maxContext: 4, headDim: 4)
        let smallHalf = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 2))
        let fullQuery = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 4 * 4))
        let fullScores = try makeOutputBuffer(
            context.device, count: 4 * 4, elementStride: 4)
        let smallFloat = try makeOutputBuffer(context.device, count: 2, elementStride: 4)

        func encodeExpectingError<E: Error & Equatable>(
            _ expected: E, _ body: @escaping (MTLComputeCommandEncoder) throws -> Void,
            file: StaticString = #filePath, line: UInt = #line
        ) throws {
            var thrown: Error?
            try context.timedDispatch { encoder in
                do { try body(encoder) } catch { thrown = error }
            }
            guard let error = thrown as? E else {
                return XCTFail("expected \(expected), got \(String(describing: thrown))",
                               file: file, line: line)
            }
            XCTAssertEqual(error, expected, file: file, line: line)
        }

        // GQA: numHeads must be a positive multiple of the cache's kvHeads.
        try encodeExpectingError(KVCacheError.gqaMismatch(numHeads: 5, kvHeads: 2)) {
            try kernels.encodeAttentionScores(
                into: $0, cache: cache, layer: 0, position: 0,
                query: fullQuery, numHeads: 5, scores: fullScores)
        }
        try encodeExpectingError(
            DecodeKernelError.nonPositiveDimension(name: "numHeads", value: 0)
        ) {
            try kernels.encodeAttentionPV(
                into: $0, cache: cache, layer: 0, position: 0,
                probs: fullScores, numHeads: 0, output: fullQuery)
        }
        // Layer / position bounds route through the cache's own checks.
        try encodeExpectingError(
            KVCacheError.indexOutOfRange(name: "layer", value: 2, bound: 2)
        ) {
            try kernels.encodeAttentionScores(
                into: $0, cache: cache, layer: 2, position: 0,
                query: fullQuery, numHeads: 4, scores: fullScores)
        }
        try encodeExpectingError(
            KVCacheError.indexOutOfRange(name: "position", value: 4, bound: 4)
        ) {
            try kernels.encodeAttentionPV(
                into: $0, cache: cache, layer: 0, position: 4,
                probs: fullScores, numHeads: 4, output: fullQuery)
        }
        // Short buffers are named.
        try encodeExpectingError(
            DecodeKernelError.bufferTooSmall(buffer: "vector", requiredBytes: 16, actualBytes: 4)
        ) {
            try kernels.encodeKVAppend(
                into: $0, cache: cache, layer: 0, component: .value,
                position: 0, vector: smallHalf)
        }
        try encodeExpectingError(
            DecodeKernelError.bufferTooSmall(buffer: "query", requiredBytes: 32, actualBytes: 4)
        ) {
            try kernels.encodeAttentionScores(
                into: $0, cache: cache, layer: 0, position: 0,
                query: smallHalf, numHeads: 4, scores: fullScores)
        }
        try encodeExpectingError(
            DecodeKernelError.bufferTooSmall(buffer: "probs", requiredBytes: 64, actualBytes: 8)
        ) {
            try kernels.encodeAttentionPV(
                into: $0, cache: cache, layer: 0, position: 0,
                probs: smallFloat, numHeads: 4, output: fullQuery)
        }
        // Softmax dimension checks.
        try encodeExpectingError(
            DecodeKernelError.nonPositiveDimension(name: "count", value: 0)
        ) {
            try kernels.encodeSoftmaxRows(
                into: $0, input: fullScores, rows: 4, count: 0, rowStride: 4,
                output: fullScores)
        }
        try encodeExpectingError(
            KVCacheError.rowCountExceedsStride(count: 5, rowStride: 4)
        ) {
            try kernels.encodeSoftmaxRows(
                into: $0, input: fullScores, rows: 4, count: 5, rowStride: 4,
                output: fullScores)
        }
    }
}
