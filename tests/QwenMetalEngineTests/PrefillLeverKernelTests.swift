import XCTest
@testable import QwenMetalEngine
import Metal

/// PF-1 lever kernels, diffed against the CPU oracle BEFORE they enter the
/// pipeline or any timing runs (hard rule 3):
/// - `PrefillSDPAKernel.encodeCausalSDPABatch` (D4 option 2, one dispatch
///   per layer) at the pre-committed attention-span constant
///   max(2⁻⁷·M, 2⁻¹¹) vs the sgemm-based oracle (hard rule 8), exact V-row
///   copy at depth 1, causality by poisoned-slot invariance, determinism,
///   GQA mapping, chunk-boundary basePosition, real headDim.
/// - `PrefillKernels.encodeRMSNormRows` (cooperative row norm) at the
///   norm-species constant max(2⁻⁸·M, 2⁻¹¹) vs the CPU RMSNorm module.
/// No constant is new; nothing loosened.
final class PrefillLeverKernelTests: XCTestCase {

    // MARK: - Deterministic RNG (AttentionKernelTests pattern)

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

    // MARK: - GPU harness helpers

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

    private func readBits(_ buffer: MTLBuffer, count: Int) -> [UInt16] {
        [UInt16](UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: UInt16.self, capacity: count),
            count: count))
    }

    /// Writes one head's row at `position` into the cache (fp16 element
    /// offset via the cache's own addressing).
    private func writeCache(
        _ cache: KVCache, layer: Int, component: KVCache.Component,
        head: Int, position: Int, values: [Float16]
    ) throws {
        let offset = try cache.elementOffset(
            layer: layer, component: component, head: head, position: position)
        let ptr = cache.buffer.contents()
            .bindMemory(to: Float16.self, capacity: offset + values.count)
        for (i, v) in values.enumerated() { ptr[offset + i] = v }
    }

    /// Fills K and V for positions 0..<depth of every kv head of `layer`
    /// with random fp16 rows; returns them as [position][kvHeads·headDim].
    private func fillCache(
        _ cache: KVCache, layer: Int, depth: Int, rng: inout SplitMix64
    ) throws -> (ks: [[Float16]], vs: [[Float16]]) {
        var ks: [[Float16]] = []
        var vs: [[Float16]] = []
        for position in 0..<depth {
            var kRow: [Float16] = []
            var vRow: [Float16] = []
            for head in 0..<cache.kvHeads {
                let k = randomHalfs(count: cache.headDim, rng: &rng)
                let v = randomHalfs(count: cache.headDim, rng: &rng)
                try writeCache(cache, layer: layer, component: .key, head: head,
                               position: position, values: k)
                try writeCache(cache, layer: layer, component: .value, head: head,
                               position: position, values: v)
                kRow += k
                vRow += v
            }
            ks.append(kRow)
            vs.append(vRow)
        }
        return (ks, vs)
    }

    // MARK: - CPU oracle (FusedSDPAKernelTests pattern, hard rule 8)

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

    /// Per query head: q·Kᵀ via sgemm → scale → reference softmax → probs·V
    /// via sgemm over cache positions 0...position.
    private func sdpaOracle(
        q: [Float16], ks: [[Float16]], vs: [[Float16]],
        numHeads: Int, kvHeads: Int, headDim: Int, position: Int
    ) throws -> [Float] {
        let groupSize = numHeads / kvHeads
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
        return ref
    }

    private func assertAttentionGate(
        _ got: [Float], _ ref: [Float], _ surface: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.count, ref.count, "\(surface): count mismatch",
                       file: file, line: line)
        let m = ref.reduce(Float(0)) { max($0, abs($1)) }
        let gate = max(exp2(-7) * m, exp2(-11))
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
            "\(surface): max |Δ| \(worst) at [\(worstIndex)] exceeds the "
                + "attention-span gate \(gate) (M = \(m))",
            file: file, line: line)
    }

    /// One batched causal SDPA dispatch over `batch` positions starting at
    /// `basePosition` (cache already filled to basePosition+batch).
    private func runBatched(
        context: MetalContext, kernel: PrefillSDPAKernel, cache: KVCache,
        layer: Int, basePosition: Int, q: [Float16], batch: Int, numHeads: Int
    ) throws -> MTLBuffer {
        let qBuffer = try makeBuffer(context.device, values: q)
        let out = try makeOutputBuffer(
            context.device, count: batch * numHeads * cache.headDim, elementStride: 2)
        try context.timedDispatch { encoder in
            try kernel.encodeCausalSDPABatch(
                into: encoder, cache: cache, layer: layer,
                basePosition: basePosition, batch: batch,
                query: qBuffer, numHeads: numHeads, output: out)
        }
        return out
    }

    /// Full oracle sweep: every chunk position vs the CPU reference at its
    /// own depth.
    private func assertBatchMatchesOracle(
        out: [Float16], q: [Float16], ks: [[Float16]], vs: [[Float16]],
        numHeads: Int, kvHeads: Int, headDim: Int, basePosition: Int,
        batch: Int, surface: String
    ) throws {
        let qDim = numHeads * headDim
        for p in 0..<batch {
            let qRow = Array(q[(p * qDim)..<((p + 1) * qDim)])
            let ref = try sdpaOracle(
                q: qRow, ks: ks, vs: vs, numHeads: numHeads, kvHeads: kvHeads,
                headDim: headDim, position: basePosition + p)
            let got = out[(p * qDim)..<((p + 1) * qDim)].map(Float.init)
            assertAttentionGate(got, ref, "\(surface) position \(basePosition + p)")
        }
    }

    // MARK: - Batched causal SDPA: oracle gates

    /// Small dims, one chunk from depth 0 through 12 positions (GQA 4→2):
    /// every position within the attention gate; the D4-option-2 kernel
    /// gates at the same constant as the per-position form (spec D6).
    func testBatchedCausalSDPAMatchesOracleFromEmptyCache() throws {
        let context = try makeContextOrSkip()
        let kernel = try PrefillSDPAKernel(context: context)
        let (numHeads, kvHeads, headDim, batch) = (4, 2, 16, 12)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: 32, headDim: headDim)
        var rng = SplitMix64(seed: 501)
        let (ks, vs) = try fillCache(cache, layer: 0, depth: batch, rng: &rng)
        let q = randomHalfs(count: batch * numHeads * headDim, rng: &rng)
        let out = try runBatched(
            context: context, kernel: kernel, cache: cache, layer: 0,
            basePosition: 0, q: q, batch: batch, numHeads: numHeads)
        try assertBatchMatchesOracle(
            out: readHalfs(out, count: batch * numHeads * headDim), q: q, ks: ks,
            vs: vs, numHeads: numHeads, kvHeads: kvHeads, headDim: headDim,
            basePosition: 0, batch: batch, surface: "batched SDPA (small dims)")
    }

    /// Chunk boundary (edge test 4 species): a second chunk starting at
    /// basePosition 9 reads the first chunk's cache correctly; real
    /// headDim 128 with the pinned GQA ratio (heads reduced 4→2 to keep the
    /// oracle cheap); depth crosses the 4-simdgroup stride many times.
    func testBatchedCausalSDPAChunkBoundaryAtRealHeadDim() throws {
        let context = try makeContextOrSkip()
        let kernel = try PrefillSDPAKernel(context: context)
        let (numHeads, kvHeads, headDim) = (4, 2, 128)
        let (basePosition, batch) = (9, 23)
        let cache = try KVCache(
            device: context.device, layers: 2, kvHeads: kvHeads,
            maxContext: 40, headDim: headDim)
        var rng = SplitMix64(seed: 502)
        // Layer 1 is the subject (layer 0 stays untouched — offsets must
        // resolve per layer).
        let (ks, vs) = try fillCache(
            cache, layer: 1, depth: basePosition + batch, rng: &rng)
        let q = randomHalfs(count: batch * numHeads * headDim, rng: &rng)
        let out = try runBatched(
            context: context, kernel: kernel, cache: cache, layer: 1,
            basePosition: basePosition, q: q, batch: batch, numHeads: numHeads)
        try assertBatchMatchesOracle(
            out: readHalfs(out, count: batch * numHeads * headDim), q: q, ks: ks,
            vs: vs, numHeads: numHeads, kvHeads: kvHeads, headDim: headDim,
            basePosition: basePosition, batch: batch,
            surface: "batched SDPA (headDim 128, chunk at 9)")
    }

    /// Depth-1 exactness (edge test 1 species): position 0's output is the
    /// mapped V row BITWISE, including -0.0, NaN payloads and subnormals;
    /// with finite V restored, every position gates against the oracle.
    func testBatchedCausalSDPADepthOneIsBitwiseVRowCopy() throws {
        let context = try makeContextOrSkip()
        let kernel = try PrefillSDPAKernel(context: context)
        let (numHeads, kvHeads, headDim, batch) = (4, 2, 8, 3)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: 8, headDim: headDim)
        var rng = SplitMix64(seed: 503)
        var (ks, vs) = try fillCache(cache, layer: 0, depth: batch, rng: &rng)
        let adversarial: [Float16] = [
            Float16(bitPattern: 0x8000),  // -0.0
            Float16(bitPattern: 0x7E01),  // NaN with payload
            Float16(bitPattern: 0x0001),  // smallest subnormal
            Float16(bitPattern: 0xFBFF),  // -max normal
            Float16(1.5), Float16(-2.25), Float16(bitPattern: 0x8001),
            Float16(0.125),
        ]
        for head in 0..<kvHeads {
            try writeCache(cache, layer: 0, component: .value, head: head,
                           position: 0, values: adversarial)
        }
        let q = randomHalfs(count: batch * numHeads * headDim, rng: &rng)
        let out = try runBatched(
            context: context, kernel: kernel, cache: cache, layer: 0,
            basePosition: 0, q: q, batch: batch, numHeads: numHeads)
        let bits = readBits(out, count: batch * numHeads * headDim)
        let expectedBits = adversarial.map(\.bitPattern)
        for head in 0..<numHeads {
            let row = Array(bits[(head * headDim)..<((head + 1) * headDim)])
            XCTAssertEqual(row, expectedBits,
                           "position 0 head \(head): not a bitwise V copy")
        }
        // Restore finite V at position 0 and gate the whole chunk (the NaN
        // row would propagate into positions 1 and 2 by design on both
        // sides, so the oracle sweep runs on finite data).
        let finite = randomHalfs(count: headDim, rng: &rng)
        for head in 0..<kvHeads {
            try writeCache(cache, layer: 0, component: .value, head: head,
                           position: 0, values: finite)
            vs[0].replaceSubrange((head * headDim)..<((head + 1) * headDim),
                                  with: finite)
        }
        ks = ks.map { $0 }
        let out2 = try runBatched(
            context: context, kernel: kernel, cache: cache, layer: 0,
            basePosition: 0, q: q, batch: batch, numHeads: numHeads)
        try assertBatchMatchesOracle(
            out: readHalfs(out2, count: batch * numHeads * headDim), q: q, ks: ks,
            vs: vs, numHeads: numHeads, kvHeads: kvHeads, headDim: headDim,
            basePosition: 0, batch: batch, surface: "batched SDPA after depth-1 check")
    }

    /// Causality (edge test 3): poisoning every slot at or beyond the chunk
    /// end with NaN, then the last chunk position's own slot, must leave
    /// every earlier position's output bitwise unchanged — position p
    /// reads 0...basePosition+p only.
    func testBatchedCausalSDPANeverReadsLaterSlots() throws {
        let context = try makeContextOrSkip()
        let kernel = try PrefillSDPAKernel(context: context)
        let (numHeads, kvHeads, headDim, batch) = (4, 2, 16, 6)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: 16, headDim: headDim)
        var rng = SplitMix64(seed: 504)
        _ = try fillCache(cache, layer: 0, depth: 16, rng: &rng)
        let q = randomHalfs(count: batch * numHeads * headDim, rng: &rng)
        let qDim = numHeads * headDim
        let clean = readBits(
            try runBatched(context: context, kernel: kernel, cache: cache, layer: 0,
                           basePosition: 2, q: q, batch: batch, numHeads: numHeads),
            count: batch * qDim)

        let nan = [Float16](repeating: Float16.nan, count: headDim)
        for position in 8..<16 {
            for head in 0..<kvHeads {
                try writeCache(cache, layer: 0, component: .key, head: head,
                               position: position, values: nan)
                try writeCache(cache, layer: 0, component: .value, head: head,
                               position: position, values: nan)
            }
        }
        let poisonedTail = readBits(
            try runBatched(context: context, kernel: kernel, cache: cache, layer: 0,
                           basePosition: 2, q: q, batch: batch, numHeads: numHeads),
            count: batch * qDim)
        XCTAssertEqual(poisonedTail, clean, "slots beyond the chunk end were read")

        for head in 0..<kvHeads {
            try writeCache(cache, layer: 0, component: .key, head: head,
                           position: 7, values: nan)
            try writeCache(cache, layer: 0, component: .value, head: head,
                           position: 7, values: nan)
        }
        let poisonedLast = readBits(
            try runBatched(context: context, kernel: kernel, cache: cache, layer: 0,
                           basePosition: 2, q: q, batch: batch, numHeads: numHeads),
            count: batch * qDim)
        XCTAssertEqual(Array(poisonedLast[0..<(5 * qDim)]), Array(clean[0..<(5 * qDim)]),
                       "an earlier position read a later slot")
    }

    /// Bitwise deterministic across runs (fixed-order merge), and exactly
    /// one dispatch per encode.
    func testBatchedCausalSDPADeterministicAndOneDispatch() throws {
        let context = try makeContextOrSkip()
        let kernel = try PrefillSDPAKernel(context: context)
        let counter = DispatchCounter()
        kernel.dispatchCounter = counter
        let (numHeads, kvHeads, headDim, batch) = (8, 4, 64, 20)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: 32, headDim: headDim)
        var rng = SplitMix64(seed: 505)
        _ = try fillCache(cache, layer: 0, depth: batch, rng: &rng)
        let q = randomHalfs(count: batch * numHeads * headDim, rng: &rng)
        let first = readBits(
            try runBatched(context: context, kernel: kernel, cache: cache, layer: 0,
                           basePosition: 0, q: q, batch: batch, numHeads: numHeads),
            count: batch * numHeads * headDim)
        XCTAssertEqual(counter.count, 1, "one dispatch per batched encode")
        for _ in 0..<3 {
            let again = readBits(
                try runBatched(context: context, kernel: kernel, cache: cache, layer: 0,
                               basePosition: 0, q: q, batch: batch, numHeads: numHeads),
                count: batch * numHeads * headDim)
            XCTAssertEqual(again, first, "batched SDPA must be bitwise deterministic")
        }
    }

    /// Both D4 forms gate against the same oracle, so their mutual
    /// difference is bounded by twice the attention constant (a derived
    /// bound, not a new one) — the per-position P4-7 loop vs one batched
    /// dispatch on the same cache and queries.
    func testBatchedMatchesPerPositionLoopWithinDerivedBound() throws {
        let context = try makeContextOrSkip()
        let batched = try PrefillSDPAKernel(context: context)
        let perPosition = try FusedSDPAKernel(context: context)
        let (numHeads, kvHeads, headDim, batch) = (4, 2, 32, 10)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: 16, headDim: headDim)
        var rng = SplitMix64(seed: 506)
        _ = try fillCache(cache, layer: 0, depth: batch, rng: &rng)
        let q = randomHalfs(count: batch * numHeads * headDim, rng: &rng)
        let qDim = numHeads * headDim
        let batchedOut = readHalfs(
            try runBatched(context: context, kernel: batched, cache: cache, layer: 0,
                           basePosition: 0, q: q, batch: batch, numHeads: numHeads),
            count: batch * qDim).map(Float.init)

        let qBuffer = try makeBuffer(context.device, values: q)
        let loopOut = try makeOutputBuffer(context.device, count: batch * qDim, elementStride: 2)
        try context.timedDispatch { encoder in
            for p in 0..<batch {
                try perPosition.encodeSDPA(
                    into: encoder, cache: cache, layer: 0, position: p,
                    query: qBuffer, queryByteOffset: p * qDim * 2,
                    numHeads: numHeads, output: loopOut, outputByteOffset: p * qDim * 2)
            }
        }
        let loop = readHalfs(loopOut, count: batch * qDim).map(Float.init)
        let m = loop.reduce(Float(0)) { max($0, abs($1)) }
        let bound = 2 * max(exp2(-7) * m, exp2(-11))
        var worst: Float = 0
        for i in 0..<loop.count { worst = max(worst, abs(batchedOut[i] - loop[i])) }
        XCTAssertLessThanOrEqual(worst, bound, "batched vs per-position loop: |Δ| \(worst)")
    }

    func testBatchedCausalSDPARejectsBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernel = try PrefillSDPAKernel(context: context)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: 2, maxContext: 8, headDim: 16)
        let q = try makeOutputBuffer(context.device, count: 8 * 4 * 16, elementStride: 2)
        let out = try makeOutputBuffer(context.device, count: 8 * 4 * 16, elementStride: 2)
        func encode(basePosition: Int, batch: Int, numHeads: Int = 4,
                    query: MTLBuffer? = nil) throws {
            try context.timedDispatch { encoder in
                try kernel.encodeCausalSDPABatch(
                    into: encoder, cache: cache, layer: 0,
                    basePosition: basePosition, batch: batch,
                    query: query ?? q, numHeads: numHeads, output: out)
            }
        }
        XCTAssertNoThrow(try encode(basePosition: 0, batch: 8))
        XCTAssertThrowsError(try encode(basePosition: 1, batch: 8), "chunk past maxContext")
        XCTAssertThrowsError(try encode(basePosition: 0, batch: 0), "empty batch")
        XCTAssertThrowsError(try encode(basePosition: -1, batch: 1), "negative base")
        XCTAssertThrowsError(try encode(basePosition: 0, batch: 2, numHeads: 3), "GQA mismatch")
        let small = try makeOutputBuffer(context.device, count: 4 * 16, elementStride: 2)
        XCTAssertThrowsError(try encode(basePosition: 0, batch: 2, query: small), "query too small")
    }

    // MARK: - Cooperative batched RMSNorm (norm-species gate vs the CPU module)

    private func randomBF16(count: Int, rng: inout SplitMix64) -> [UInt16] {
        (0..<count).map { _ in
            UInt16(truncatingIfNeeded: Float.random(in: -1...1, using: &rng).bitPattern >> 16)
        }
    }

    private func upcast(_ bits: [UInt16]) -> [Float] {
        bits.map { Float(bitPattern: UInt32($0) << 16) }
    }

    private func assertNormGate(
        _ got: [Float], _ ref: [Float], _ surface: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.count, ref.count, surface, file: file, line: line)
        let m = ref.reduce(Float(0)) { max($0, abs($1)) }
        let gate = max(exp2(-8) * m, exp2(-11))
        var worst: Float = 0
        var worstIndex = 0
        for i in 0..<ref.count where abs(got[i] - ref[i]) > worst {
            worst = abs(got[i] - ref[i])
            worstIndex = i
        }
        XCTAssertLessThanOrEqual(
            worst, gate,
            "\(surface): max |Δ| \(worst) at [\(worstIndex)] exceeds the norm-species "
                + "gate \(gate) (M = \(m))",
            file: file, line: line)
    }

    private func runRMSNormRows(rows: Int, dim: Int, seed: UInt64) throws {
        let context = try makeContextOrSkip()
        let kernels = try PrefillKernels(context: context)
        let counter = DispatchCounter()
        kernels.dispatchCounter = counter
        var rng = SplitMix64(seed: seed)
        let x = randomHalfs(count: rows * dim, rng: &rng)
        let weightBits = randomBF16(count: dim, rng: &rng)
        let eps: Float = 1e-6
        let xBuffer = try makeBuffer(context.device, values: x)
        let weightBuffer = try makeBuffer(context.device, values: weightBits)
        let outBuffer = try makeOutputBuffer(context.device, count: rows * dim, elementStride: 2)
        try context.timedDispatch { encoder in
            try kernels.encodeRMSNormRows(
                into: encoder, input: xBuffer, weight: weightBuffer,
                weightByteOffset: 0, rows: rows, dim: dim, eps: eps,
                output: outBuffer)
        }
        XCTAssertEqual(counter.count, 1)
        let ref = try RMSNorm(weight: upcast(weightBits), eps: eps)(x.map(Float.init))
        assertNormGate(readHalfs(outBuffer, count: rows * dim).map(Float.init), ref,
                       "rmsnorm rows=\(rows) dim=\(dim)")
        let first = readBits(outBuffer, count: rows * dim)
        try context.timedDispatch { encoder in
            try kernels.encodeRMSNormRows(
                into: encoder, input: xBuffer, weight: weightBuffer,
                weightByteOffset: 0, rows: rows, dim: dim, eps: eps,
                output: outBuffer)
        }
        XCTAssertEqual(readBits(outBuffer, count: rows * dim), first,
                       "cooperative norm must be bitwise deterministic")
    }

    /// dim < threads (idle lanes), dim not a multiple of the threadgroup,
    /// dim ≫ threads (many strided elements per thread), and the pinned
    /// hidden size over a chunk-sized batch.
    func testRMSNormRowsMatchCPUAcrossShapes() throws {
        try runRMSNormRows(rows: 3, dim: 67, seed: 20)
        try runRMSNormRows(rows: 5, dim: 300, seed: 21)
        try runRMSNormRows(rows: 2, dim: 6144, seed: 22)
        try runRMSNormRows(rows: 64, dim: 2048, seed: 23)
    }

    func testRMSNormRowsRejectsBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernels = try PrefillKernels(context: context)
        let x = try makeOutputBuffer(context.device, count: 4 * 64, elementStride: 2)
        let w = try makeOutputBuffer(context.device, count: 64, elementStride: 2)
        let out = try makeOutputBuffer(context.device, count: 4 * 64, elementStride: 2)
        func encode(rows: Int, dim: Int, weightByteOffset: Int = 0) throws {
            try context.timedDispatch { encoder in
                try kernels.encodeRMSNormRows(
                    into: encoder, input: x, weight: w,
                    weightByteOffset: weightByteOffset, rows: rows, dim: dim,
                    eps: 1e-6, output: out)
            }
        }
        XCTAssertNoThrow(try encode(rows: 4, dim: 64))
        XCTAssertThrowsError(try encode(rows: 0, dim: 64))
        XCTAssertThrowsError(try encode(rows: 5, dim: 64), "input too small")
        XCTAssertThrowsError(try encode(rows: 4, dim: 64, weightByteOffset: 1), "odd offset")
        XCTAssertThrowsError(try encode(rows: 4, dim: 128), "weight too small")
    }
}
