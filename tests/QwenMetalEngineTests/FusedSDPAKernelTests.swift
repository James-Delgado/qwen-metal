import XCTest
@testable import QwenMetalEngine
import Metal

/// P4-2 fused-SDPA kernel tests (docs/phases/phase-4.md D2 + enumerated edge
/// cases 1-5): the one-dispatch scores→softmax→PV replacement, diffed against
/// the CPU oracle BEFORE any optimization iteration (hard rule 3).
///
/// Gates are the pre-committed Phase 4 fused-span constants (DECISIONS.md
/// 2026-09-05 — reused Phase 2/3 species, no new constants):
///   Exact:            p=0 output == the mapped V row bitwise (weight is
///                     exactly 1.0; the P2-3 exactness carries).
///   Attention span:   |Δ| <= max(2⁻⁷·M, 2⁻¹¹), M = max|ref| over the slice.
/// Matmul-shaped reference work (q·Kᵀ, probs·V) routes through BLAS.sgemm
/// (hard rule 8); the softmax reference replicates the CPU module's formula.
final class FusedSDPAKernelTests: XCTestCase {

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

    // MARK: - GPU harness helpers (AttentionKernelTests pattern)

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

    private func fillBits(_ buffer: MTLBuffer, _ bits: UInt16) {
        let count = buffer.length / 2
        let ptr = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
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

    /// CPU full-recompute oracle for the fused span: per query head,
    /// q·Kᵀ via sgemm (hard rule 8) → scale → reference softmax → probs·V
    /// via sgemm, over the fp16 rows the kernel reads.
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

    // MARK: - Gate assertion (pre-committed attention-span constant; never loosened)

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

    /// Runs one fused-SDPA dispatch against cache state written directly by
    /// the test and returns the fp16 output rows.
    private func runFusedSDPA(
        context: MetalContext, kernels: FusedSDPAKernel, cache: KVCache,
        layer: Int, position: Int, q: [Float16], numHeads: Int
    ) throws -> [Float16] {
        let qBuffer = try makeBuffer(context.device, values: q)
        let out = try makeOutputBuffer(
            context.device, count: numHeads * cache.headDim, elementStride: 2)
        try context.timedDispatch { encoder in
            try kernels.encodeSDPA(
                into: encoder, cache: cache, layer: layer, position: position,
                query: qBuffer, numHeads: numHeads, output: out)
        }
        return readHalfs(out, count: numHeads * cache.headDim)
    }

    // MARK: - Edge case 1: p=0 — online-softmax weight exactly 1.0, output == V row bitwise

    func testP0OutputEqualsVRowBitwise() throws {
        let context = try makeContextOrSkip()
        let kernels = try FusedSDPAKernel(context: context)
        let (kvHeads, numHeads, headDim, maxContext) = (2, 4, 6, 3)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        fillBits(cache.buffer, 0xABAB)

        var rng = SplitMix64(seed: 40)
        let k0 = randomHalfs(count: kvHeads * headDim, rng: &rng)
        // V carries fp16 boundary values (±0, subnormal, ±max-normal, ±inf,
        // NaN payload) alongside randoms: at p=0 the kernel's weight-1.0
        // degenerate case is a pure copy of the V row, so every bit
        // pattern — including -0.0 and the NaN payload — must survive.
        var v0 = randomHalfs(count: kvHeads * headDim, rng: &rng)
        let boundary: [UInt16] = [
            0x0000, 0x8000, 0x0001, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E01,
        ]
        for (i, bits) in boundary.enumerated() where i < v0.count {
            v0[i] = Float16(bitPattern: bits)
        }
        let q = randomHalfs(count: numHeads * headDim, rng: &rng)
        for h in 0..<kvHeads {
            try writeCacheSlot(cache, layer: 0, component: .key, head: h,
                               position: 0,
                               values: Array(k0[(h * headDim)..<((h + 1) * headDim)]))
            try writeCacheSlot(cache, layer: 0, component: .value, head: h,
                               position: 0,
                               values: Array(v0[(h * headDim)..<((h + 1) * headDim)]))
        }

        let got = try runFusedSDPA(
            context: context, kernels: kernels, cache: cache, layer: 0,
            position: 0, q: q, numHeads: numHeads)

        var expectedBits = [UInt16]()
        for qHead in 0..<numHeads {
            let kvHead = qHead / (numHeads / kvHeads)
            expectedBits += v0[(kvHead * headDim)..<((kvHead + 1) * headDim)]
                .map(\.bitPattern)
        }
        XCTAssertEqual(got.map(\.bitPattern), expectedBits,
                       "p=0 fused SDPA output must equal the mapped V row bitwise")
    }

    // MARK: - Edge case 2: GQA mapping — each query head reads its own KV head

    func testGQAMappingReadsCorrectKVHeadThroughFusedPath() throws {
        let context = try makeContextOrSkip()
        let kernels = try FusedSDPAKernel(context: context)
        // kvHeads=3, numHeads=6 (groupSize 2), headDim=4 — the P2-3 exact
        // small-dims construction. Part A (exact): p=0 output IS the V row,
        // and V rows name their KV head, so a V-side mapping mistake fails
        // bitwise. Part B (gated): two positions whose K scores differ per
        // KV head — a K-side mapping mistake (e.g. kvHead = qHead % 3)
        // mixes the wrong softmax weights and lands far outside the gate.
        let (kvHeads, numHeads, headDim, maxContext) = (3, 6, 4, 2)
        let groupSize = numHeads / kvHeads
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        fillBits(cache.buffer, 0x0000)

        var ks = [[Float16]](); var vs = [[Float16]]()
        for j in 0..<2 {
            var kRow = [Float16](repeating: 0, count: kvHeads * headDim)
            var vRow = [Float16](repeating: 0, count: kvHeads * headDim)
            for h in 0..<kvHeads {
                // K[h, 0][0] = h+1, K[h, 1][0] = -(h+1): per-head distinct
                // score split across the two positions.
                kRow[h * headDim] = Float16(j == 0 ? Float(h + 1) : -Float(h + 1))
                // V[h, j][0] names (head, position): 8h + j + 1.
                vRow[h * headDim] = Float16(Float(8 * h + j + 1))
            }
            ks.append(kRow); vs.append(vRow)
            for h in 0..<kvHeads {
                let dims = (h * headDim)..<((h + 1) * headDim)
                try writeCacheSlot(cache, layer: 0, component: .key, head: h,
                                   position: j, values: Array(kRow[dims]))
                try writeCacheSlot(cache, layer: 0, component: .value, head: h,
                                   position: j, values: Array(vRow[dims]))
            }
        }
        var q = [Float16](repeating: 0, count: numHeads * headDim)
        for h in 0..<numHeads { q[h * headDim] = 1 }

        // Part A — p=0: output must be exactly V[qHead/groupSize, 0].
        let gotP0 = try runFusedSDPA(
            context: context, kernels: kernels, cache: cache, layer: 0,
            position: 0, q: q, numHeads: numHeads)
        for qHead in 0..<numHeads {
            let kvHead = qHead / groupSize
            XCTAssertEqual(
                gotP0[qHead * headDim], Float16(Float(8 * kvHead + 1)),
                "p=0: Q head \(qHead) must return KV head \(kvHead)'s V row exactly")
        }

        // Part B — p=1: gate vs the correct-mapping oracle, and prove teeth
        // by checking the wrong-mapping (qHead % kvHeads) value is far away.
        let got = try runFusedSDPA(
            context: context, kernels: kernels, cache: cache, layer: 0,
            position: 1, q: q, numHeads: numHeads
        ).map(Float.init)
        let ref = try sdpaOracle(
            q: q, ks: ks, vs: vs, numHeads: numHeads, kvHeads: kvHeads,
            headDim: headDim, position: 1)
        assertAttentionGate(got, ref, "GQA mapping, fused SDPA at p=1")

        let scale = 1 / Float(headDim).squareRoot()
        for qHead in 0..<numHeads {
            let wrongKVHead = qHead % kvHeads
            guard wrongKVHead != qHead / groupSize else { continue }
            let s0 = Float(wrongKVHead + 1) * scale
            let s1 = -Float(wrongKVHead + 1) * scale
            let w = cpuSoftmax([s0, s1])
            let wrong = w[0] * Float(8 * wrongKVHead + 1) + w[1] * Float(8 * wrongKVHead + 2)
            XCTAssertGreaterThan(
                abs(got[qHead * headDim] - wrong), 0.5,
                "test teeth: the wrong-mapping value must be clearly distinguishable")
        }
    }

    // MARK: - Edge case 3: cache boundary — p at the last valid slot (4095)

    func testCacheBoundaryLastValidSlotRunsInBounds() throws {
        let context = try makeContextOrSkip()
        let kernels = try FusedSDPAKernel(context: context)
        let attention = try AttentionKernels(context: context)
        // Full-depth maxContext=4096 at small dims (128 KiB cache): the fused
        // kernel streams every position 0...4095 — an off-by-one read past
        // the K slab lands in the V slab and fails the oracle diff.
        let (kvHeads, numHeads, headDim, maxContext) = (1, 2, 8, 4096)
        let position = maxContext - 1
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)

        var rng = SplitMix64(seed: 50)
        var ks = [[Float16]](); var vs = [[Float16]]()
        for j in 0...position {
            let kRow = randomHalfs(count: kvHeads * headDim, rng: &rng)
            let vRow = randomHalfs(count: kvHeads * headDim, rng: &rng)
            try writeCacheSlot(cache, layer: 0, component: .key, head: 0,
                               position: j, values: kRow)
            try writeCacheSlot(cache, layer: 0, component: .value, head: 0,
                               position: j, values: vRow)
            ks.append(kRow); vs.append(vRow)
        }
        let q = randomHalfs(count: numHeads * headDim, rng: &rng)

        let got = try runFusedSDPA(
            context: context, kernels: kernels, cache: cache, layer: 0,
            position: position, q: q, numHeads: numHeads
        ).map(Float.init)
        let ref = try sdpaOracle(
            q: q, ks: ks, vs: vs, numHeads: numHeads, kvHeads: kvHeads,
            headDim: headDim, position: position)
        assertAttentionGate(got, ref, "fused SDPA at the last valid slot p=4095")

        // The context-limit stop is unchanged: an SDPA encode at position ==
        // maxContext throws BEFORE any dispatch, and the next kv-append
        // throws contextFull with the cache untouched (P2-3 edge 5 carries).
        let qBuffer = try makeBuffer(context.device, values: q)
        let out = try makeOutputBuffer(
            context.device, count: numHeads * headDim, elementStride: 2)
        var sdpaThrown: Error?
        try context.timedDispatch { encoder in
            do {
                try kernels.encodeSDPA(
                    into: encoder, cache: cache, layer: 0, position: maxContext,
                    query: qBuffer, numHeads: numHeads, output: out)
            } catch { sdpaThrown = error }
        }
        XCTAssertEqual(
            sdpaThrown as? KVCacheError,
            .indexOutOfRange(name: "position", value: maxContext, bound: maxContext))

        let snapshot = readBits(cache.buffer, count: cache.byteCount / 2)
        let vector = try makeBuffer(
            context.device, values: [Float16](repeating: 1, count: kvHeads * headDim))
        var appendThrown: Error?
        try context.timedDispatch { encoder in
            do {
                try attention.encodeKVAppend(
                    into: encoder, cache: cache, layer: 0, component: .key,
                    position: maxContext, vector: vector)
            } catch { appendThrown = error }
        }
        XCTAssertEqual(appendThrown as? KVCacheError,
                       .contextFull(position: maxContext, maxContext: maxContext))
        XCTAssertEqual(readBits(cache.buffer, count: cache.byteCount / 2), snapshot,
                       "the failed append must leave the cache untouched")
    }

    // MARK: - Edge case 4: adversarial online-softmax orderings

    func testOnlineSoftmaxAdversarialOrderings() throws {
        let context = try makeContextOrSkip()
        let kernels = try FusedSDPAKernel(context: context)
        // Score profiles targeting the classic online-softmax failure modes:
        // running-max rescale when the max arrives first (never rescales),
        // last (rescales every step), a large-negative tail (exp underflow),
        // and all-equal ties (uniform weights). Scores are steered through
        // K[·][0] with q = [1, 0, ...]: score_j = K[j][0] · scale.
        let (kvHeads, numHeads, headDim, maxContext) = (2, 4, 8, 32)
        let position = maxContext - 1
        let scale = 1 / Float(headDim).squareRoot()
        let count = position + 1

        // Target score sequences (pre-scale K values; fp16-rounded before the
        // oracle reads them, so both sides see identical inputs).
        func kValues(_ profile: (Int) -> Float) -> [Float] {
            (0..<count).map { profile($0) / scale }
        }
        let profiles: [(name: String, values: [Float])] = [
            ("max-first-descending", kValues { 8 - Float($0) * 0.5 }),
            ("max-last-ascending", kValues { Float($0) * 0.5 - 8 }),
            ("large-negative-tail", kValues { $0 == 3 ? 5 : -60 }),
            ("all-equal-ties", kValues { _ in 2.5 }),
        ]

        for profile in profiles {
            let cache = try KVCache(
                device: context.device, layers: 1, kvHeads: kvHeads,
                maxContext: maxContext, headDim: headDim)
            var rng = SplitMix64(seed: 60)
            var ks = [[Float16]](); var vs = [[Float16]]()
            for j in 0...position {
                var kRow = [Float16](repeating: 0, count: kvHeads * headDim)
                for h in 0..<kvHeads {
                    // Both KV heads run the profile, offset so the heads
                    // still differ (h=1 shifts by one position).
                    kRow[h * headDim] = Float16(profile.values[(j + h) % count])
                }
                let vRow = randomHalfs(count: kvHeads * headDim, rng: &rng)
                for h in 0..<kvHeads {
                    let dims = (h * headDim)..<((h + 1) * headDim)
                    try writeCacheSlot(cache, layer: 0, component: .key, head: h,
                                       position: j, values: Array(kRow[dims]))
                    try writeCacheSlot(cache, layer: 0, component: .value, head: h,
                                       position: j, values: Array(vRow[dims]))
                }
                ks.append(kRow); vs.append(vRow)
            }
            var q = [Float16](repeating: 0, count: numHeads * headDim)
            for h in 0..<numHeads { q[h * headDim] = 1 }

            let got = try runFusedSDPA(
                context: context, kernels: kernels, cache: cache, layer: 0,
                position: position, q: q, numHeads: numHeads
            ).map(Float.init)
            let ref = try sdpaOracle(
                q: q, ks: ks, vs: vs, numHeads: numHeads, kvHeads: kvHeads,
                headDim: headDim, position: position)
            assertAttentionGate(got, ref, "adversarial ordering '\(profile.name)'")
        }
    }

    // MARK: - Edge case 5: window-depth accumulation at real head dims

    func testWindowDepthAccumulationAtRealHeadDim() throws {
        let context = try makeContextOrSkip()
        let kernels = try FusedSDPAKernel(context: context)
        // p = 511 ≈ the canonical-window depth, headDim = 128 (the pinned
        // model's), GQA 2:1 — the long-loop accumulator drift check.
        let (kvHeads, numHeads, headDim, maxContext) = (2, 4, 128, 512)
        let position = maxContext - 1
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)

        var rng = SplitMix64(seed: 70)
        var ks = [[Float16]](); var vs = [[Float16]]()
        for j in 0...position {
            let kRow = randomHalfs(count: kvHeads * headDim, rng: &rng)
            let vRow = randomHalfs(count: kvHeads * headDim, rng: &rng)
            for h in 0..<kvHeads {
                let dims = (h * headDim)..<((h + 1) * headDim)
                try writeCacheSlot(cache, layer: 0, component: .key, head: h,
                                   position: j, values: Array(kRow[dims]))
                try writeCacheSlot(cache, layer: 0, component: .value, head: h,
                                   position: j, values: Array(vRow[dims]))
            }
            ks.append(kRow); vs.append(vRow)
        }
        let q = randomHalfs(count: numHeads * headDim, rng: &rng)

        let got = try runFusedSDPA(
            context: context, kernels: kernels, cache: cache, layer: 0,
            position: position, q: q, numHeads: numHeads
        ).map(Float.init)
        let ref = try sdpaOracle(
            q: q, ks: ks, vs: vs, numHeads: numHeads, kvHeads: kvHeads,
            headDim: headDim, position: position)
        assertAttentionGate(got, ref, "window-depth p=\(position), headDim 128")
    }

    // MARK: - P4-7 split boundaries: every empty/partial/full chunk pattern

    /// The split-K structure partitions positions 0...p into NUM_SPLITS
    /// contiguous chunks (ceil division). Sweeping p = 0...2·NUM_SPLITS+1
    /// exercises every boundary pattern: p+1 < NUM_SPLITS (trailing chunks
    /// EMPTY — their partial state must not contribute), p+1 == NUM_SPLITS
    /// (all chunks singleton), and p+1 not a multiple of NUM_SPLITS (ragged
    /// final chunk). Odd headDim rides along (partially-filled lane stride).
    /// Each depth diffs against the CPU oracle at the verbatim
    /// attention-species gate; two encodes per depth pin determinism.
    func testSplitBoundaryDepthSweepMatchesOracle() throws {
        let context = try makeContextOrSkip()
        let kernels = try FusedSDPAKernel(context: context)
        let numSplits = FusedSDPAKernel.numSplits
        let maxContext = 2 * numSplits + 2
        let (kvHeads, numHeads, headDim) = (2, 4, 19)
        for position in 0...(2 * numSplits + 1) {
            let cache = try KVCache(
                device: context.device, layers: 1, kvHeads: kvHeads,
                maxContext: maxContext, headDim: headDim)
            fillBits(cache.buffer, 0xABAB)
            var rng = SplitMix64(seed: UInt64(100 + position))
            var ks = [[Float16]](); var vs = [[Float16]]()
            for j in 0...position {
                let kRow = randomHalfs(count: kvHeads * headDim, rng: &rng)
                let vRow = randomHalfs(count: kvHeads * headDim, rng: &rng)
                for h in 0..<kvHeads {
                    let dims = (h * headDim)..<((h + 1) * headDim)
                    try writeCacheSlot(cache, layer: 0, component: .key, head: h,
                                       position: j, values: Array(kRow[dims]))
                    try writeCacheSlot(cache, layer: 0, component: .value, head: h,
                                       position: j, values: Array(vRow[dims]))
                }
                ks.append(kRow); vs.append(vRow)
            }
            let q = randomHalfs(count: numHeads * headDim, rng: &rng)

            let first = try runFusedSDPA(
                context: context, kernels: kernels, cache: cache, layer: 0,
                position: position, q: q, numHeads: numHeads)
            let ref = try sdpaOracle(
                q: q, ks: ks, vs: vs, numHeads: numHeads, kvHeads: kvHeads,
                headDim: headDim, position: position)
            assertAttentionGate(
                first.map(Float.init), ref, "split-boundary sweep p=\(position)")
            let again = try runFusedSDPA(
                context: context, kernels: kernels, cache: cache, layer: 0,
                position: position, q: q, numHeads: numHeads)
            XCTAssertEqual(
                first.map(\.bitPattern), again.map(\.bitPattern),
                "split-boundary sweep p=\(position): bitwise determinism")
        }
    }

    /// P4-7 structure pin at the kernel level: one `encodeSDPA` call encodes
    /// exactly TWO dispatches (pass 1 partials + reduce), measured by the
    /// P2-5 counter at the dispatch call sites — never derived.
    func testEncodeSDPAEncodesTwoDispatches() throws {
        let context = try makeContextOrSkip()
        let kernels = try FusedSDPAKernel(context: context)
        let counter = DispatchCounter()
        kernels.dispatchCounter = counter
        let (kvHeads, numHeads, headDim, maxContext) = (2, 4, 8, 4)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        var rng = SplitMix64(seed: 110)
        for j in 0...1 {
            for h in 0..<kvHeads {
                try writeCacheSlot(cache, layer: 0, component: .key, head: h,
                                   position: j,
                                   values: randomHalfs(count: headDim, rng: &rng))
                try writeCacheSlot(cache, layer: 0, component: .value, head: h,
                                   position: j,
                                   values: randomHalfs(count: headDim, rng: &rng))
            }
        }
        let q = randomHalfs(count: numHeads * headDim, rng: &rng)
        for position in [0, 1] {
            counter.reset()
            _ = try runFusedSDPA(
                context: context, kernels: kernels, cache: cache, layer: 0,
                position: position, q: q, numHeads: numHeads)
            XCTAssertEqual(
                counter.count, 2,
                "p=\(position): the split-K SDPA is pass 1 + reduce, always")
        }
    }

    // MARK: - Odd shapes (lane striding: headDim not a multiple of the SIMD width)

    func testOddShapesMatchOracle() throws {
        let context = try makeContextOrSkip()
        let kernels = try FusedSDPAKernel(context: context)
        // headDim 19 exercises a partially-filled first stride; 67 exercises
        // uneven multi-stride lanes. Layer 1 of 2 catches base-offset bugs.
        let shapes: [(kvHeads: Int, numHeads: Int, headDim: Int, maxContext: Int, position: Int)] = [
            (3, 6, 19, 11, 6),
            (2, 4, 67, 16, 9),
        ]
        for shape in shapes {
            let cache = try KVCache(
                device: context.device, layers: 2, kvHeads: shape.kvHeads,
                maxContext: shape.maxContext, headDim: shape.headDim)
            fillBits(cache.buffer, 0xABAB)
            var rng = SplitMix64(seed: UInt64(80 + shape.headDim))
            var ks = [[Float16]](); var vs = [[Float16]]()
            for j in 0...shape.position {
                let kRow = randomHalfs(count: shape.kvHeads * shape.headDim, rng: &rng)
                let vRow = randomHalfs(count: shape.kvHeads * shape.headDim, rng: &rng)
                for h in 0..<shape.kvHeads {
                    let dims = (h * shape.headDim)..<((h + 1) * shape.headDim)
                    try writeCacheSlot(cache, layer: 1, component: .key, head: h,
                                       position: j, values: Array(kRow[dims]))
                    try writeCacheSlot(cache, layer: 1, component: .value, head: h,
                                       position: j, values: Array(vRow[dims]))
                }
                ks.append(kRow); vs.append(vRow)
            }
            let q = randomHalfs(count: shape.numHeads * shape.headDim, rng: &rng)

            let got = try runFusedSDPA(
                context: context, kernels: kernels, cache: cache, layer: 1,
                position: shape.position, q: q, numHeads: shape.numHeads
            ).map(Float.init)
            let ref = try sdpaOracle(
                q: q, ks: ks, vs: vs, numHeads: shape.numHeads,
                kvHeads: shape.kvHeads, headDim: shape.headDim,
                position: shape.position)
            assertAttentionGate(
                got, ref,
                "odd shape headDim \(shape.headDim), p=\(shape.position)")
        }
    }

    // MARK: - Determinism (the pipeline's bitwise incremental contract rides on it)

    func testFusedSDPAIsDeterministicAcrossRuns() throws {
        let context = try makeContextOrSkip()
        let kernels = try FusedSDPAKernel(context: context)
        let (kvHeads, numHeads, headDim, maxContext) = (2, 4, 64, 40)
        let position = 33
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        var rng = SplitMix64(seed: 90)
        for j in 0...position {
            for h in 0..<kvHeads {
                try writeCacheSlot(cache, layer: 0, component: .key, head: h,
                                   position: j,
                                   values: randomHalfs(count: headDim, rng: &rng))
                try writeCacheSlot(cache, layer: 0, component: .value, head: h,
                                   position: j,
                                   values: randomHalfs(count: headDim, rng: &rng))
            }
        }
        let q = randomHalfs(count: numHeads * headDim, rng: &rng)

        let first = try runFusedSDPA(
            context: context, kernels: kernels, cache: cache, layer: 0,
            position: position, q: q, numHeads: numHeads)
        for _ in 0..<3 {
            let again = try runFusedSDPA(
                context: context, kernels: kernels, cache: cache, layer: 0,
                position: position, q: q, numHeads: numHeads)
            XCTAssertEqual(
                first.map(\.bitPattern), again.map(\.bitPattern),
                "fused SDPA must be bitwise deterministic across runs")
        }
    }

    // MARK: - Loud rejection of bad inputs (no dispatch reaches the GPU)

    func testFusedSDPARejectsBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernels = try FusedSDPAKernel(context: context)
        let cache = try KVCache(
            device: context.device, layers: 2, kvHeads: 2, maxContext: 4, headDim: 4)
        let smallHalf = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 2))
        let fullQuery = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 4 * 4))
        let fullOut = try makeOutputBuffer(context.device, count: 4 * 4, elementStride: 2)

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

        try encodeExpectingError(KVCacheError.gqaMismatch(numHeads: 5, kvHeads: 2)) {
            try kernels.encodeSDPA(
                into: $0, cache: cache, layer: 0, position: 0,
                query: fullQuery, numHeads: 5, output: fullOut)
        }
        try encodeExpectingError(
            DecodeKernelError.nonPositiveDimension(name: "numHeads", value: 0)
        ) {
            try kernels.encodeSDPA(
                into: $0, cache: cache, layer: 0, position: 0,
                query: fullQuery, numHeads: 0, output: fullOut)
        }
        try encodeExpectingError(
            KVCacheError.indexOutOfRange(name: "layer", value: 2, bound: 2)
        ) {
            try kernels.encodeSDPA(
                into: $0, cache: cache, layer: 2, position: 0,
                query: fullQuery, numHeads: 4, output: fullOut)
        }
        try encodeExpectingError(
            KVCacheError.indexOutOfRange(name: "position", value: 4, bound: 4)
        ) {
            try kernels.encodeSDPA(
                into: $0, cache: cache, layer: 0, position: 4,
                query: fullQuery, numHeads: 4, output: fullOut)
        }
        try encodeExpectingError(
            DecodeKernelError.bufferTooSmall(buffer: "query", requiredBytes: 32, actualBytes: 4)
        ) {
            try kernels.encodeSDPA(
                into: $0, cache: cache, layer: 0, position: 0,
                query: smallHalf, numHeads: 4, output: fullOut)
        }
        try encodeExpectingError(
            DecodeKernelError.bufferTooSmall(buffer: "output", requiredBytes: 32, actualBytes: 4)
        ) {
            try kernels.encodeSDPA(
                into: $0, cache: cache, layer: 0, position: 0,
                query: fullQuery, numHeads: 4, output: smallHalf)
        }

        // headDim beyond the fused kernel's register budget is refused loudly.
        let bigCache = try KVCache(
            device: context.device, layers: 1, kvHeads: 1, maxContext: 2, headDim: 130)
        let bigQuery = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 2 * 130))
        let bigOut = try makeOutputBuffer(context.device, count: 2 * 130, elementStride: 2)
        try encodeExpectingError(
            KVCacheError.headDimExceedsFusedLimit(headDim: 130, limit: 128)
        ) {
            try kernels.encodeSDPA(
                into: $0, cache: bigCache, layer: 0, position: 0,
                query: bigQuery, numHeads: 2, output: bigOut)
        }
    }
}

/// Real-artifact fused-path smoke (P3-2 smoke precedent, GPU fused side):
/// skips cleanly when the local-only packed artifact is absent. NOT the
/// binding Tier-M/E verification — that runs on the fused path at P4-4;
/// this pins that the fused wiring produces sane output at the pinned
/// model's real dims (28 layers, 16/8 heads, headDim 128) and stays
/// consistent with the naive path within a bound DERIVED from the
/// committed full-stack species constant (no new gate constant).
final class FusedPathRealArtifactSmokeTests: XCTestCase {

    static let packedURL = SharedCheckpoint.modelsDir
        .appendingPathComponent("qwen3-1.7b-70d244cc-q4g64.safetensors")

    override func setUpWithError() throws {
        guard FileManager.default.fileExists(atPath: Self.packedURL.path) else {
            throw XCTSkip(
                "packed artifact not present at \(Self.packedURL.path) "
                + "(local-only — produce it with `qwen-metal-cli pack`)")
        }
    }

    func testFusedSmokeDecodeAndNaiveConsistencyOnRealArtifact() throws {
        let context: MetalContext
        do {
            context = try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
        let packed = try PackedCheckpoint(
            path: Self.packedURL.path,
            expectedRevision: SharedCheckpoint.pinnedRevision)
        let config = try ModelConfig(
            jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
        let fused = try GPUModel(
            packed: packed, config: config, context: context, maxContext: 64,
            kernelPath: .fused)
        XCTAssertEqual(fused.kernelPath, .fused)
        let naive = try GPUModel(
            packed: packed, config: config, context: context, maxContext: 64,
            kernelPath: .naive)

        // Last-prompt-position logits, fused vs naive. Both paths gate
        // against the SAME live CPU-quant oracle at the full-stack species
        // max(2⁻⁵·M, 2⁻¹¹) (naive: GPUQuantSuiteTests today; fused: P4-4),
        // so triangle inequality bounds their mutual difference by twice
        // that — a derived sanity bound, not a new constant (spec D5:
        // bitwise naive-vs-fused equality is NOT required).
        let prompt = try SharedCheckpoint.promptFixture("short_english")
        let fusedLogits = try fused.lastPositionLogits(ids: prompt.inputIds)
        // P4-7 (edge test 8 species), real dims: the folded fused path
        // MEASURES 199 dispatches/token with logits (7/layer × 28 +
        // embedding + final norm + lm_head — the split-K SDPA is two
        // dispatches: pass 1 partials + reduce; was 171 at P4-6, 227 at
        // P4-3) — under the pre-committed ≤300 dispatch gate (DECISIONS.md
        // 2026-09-05; the on-device gate verdict is P4-11's).
        let fusedDispatches = try XCTUnwrap(fused.lastStepDispatchCount)
        XCTAssertEqual(fusedDispatches, 199)
        XCTAssertLessThanOrEqual(fusedDispatches, 300,
                                 "pre-committed Phase 4 dispatch gate")
        let naiveLogits = try naive.lastPositionLogits(ids: prompt.inputIds)
        XCTAssertEqual(fusedLogits.count, naiveLogits.count)
        let m = naiveLogits.map(abs).max() ?? 0
        let bound = 2 * max(exp2(-5) * m, exp2(-11))
        var worst: Float = 0
        var worstIndex = 0
        for i in 0..<naiveLogits.count {
            let delta = abs(fusedLogits[i] - naiveLogits[i])
            if delta > worst {
                worst = delta
                worstIndex = i
            }
        }
        XCTAssertLessThanOrEqual(
            worst, bound,
            "fused-vs-naive logits: max |Δ| \(worst) at [\(worstIndex)] exceeds "
                + "the derived 2×full-stack bound \(bound) (M = \(m))")

        // Free-running fused smoke: finite logits, in-vocab, non-degenerate.
        fused.reset()
        var nonFiniteStep: Int?
        let generated = try DecodeLoop(model: fused, maxContext: 64).generate(
            promptIds: prompt.inputIds,
            maxNewTokens: 8,
            eosTokenIds: [151645, 151643],
            onStep: { step, logits, _ in
                if nonFiniteStep == nil, !logits.allSatisfy(\.isFinite) {
                    nonFiniteStep = step
                }
            })
        XCTAssertNil(nonFiniteStep, "non-finite logits at step \(nonFiniteStep ?? -1)")
        XCTAssertFalse(generated.isEmpty)
        XCTAssertTrue(generated.allSatisfy { (0..<config.vocabSize).contains($0) },
                      "generated ids out of vocab range: \(generated)")
        print("FusedPathRealArtifactSmokeTests: prompt \(prompt.inputIds) -> \(generated)")
    }
}
