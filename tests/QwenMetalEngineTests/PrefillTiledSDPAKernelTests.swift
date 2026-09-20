import XCTest
@testable import QwenMetalEngine
import Metal

/// PF-2: the query-tiled causal SDPA (`PrefillTiledSDPAKernel`), diffed
/// against the CPU oracle BEFORE it enters the pipeline or any timing run
/// (hard rule 3) at the pre-committed attention-span constant
/// max(2⁻⁷·M, 2⁻¹¹) vs the sgemm-based oracle (hard rule 8) — the same
/// gate, oracle, and helper shapes as the PF-1 `PrefillLeverKernelTests`.
/// Adds the fragment-ownership PROBE the kernel's register-level softmax
/// rests on, multi-tile / multi-block / ragged chunks, every GQA tiling
/// (1, 2, 4 heads per tile and the fallback), exact depth-1 copy, two
/// bitwise causality invariants, determinism, one dispatch, the derived
/// bound vs the PF-1 kernel, and rejects. No constant is new; nothing
/// loosened.
final class PrefillTiledSDPAKernelTests: XCTestCase {

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

    private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        [Float](UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: count),
            count: count))
    }

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

    private func runTiled(
        context: MetalContext, kernel: PrefillTiledSDPAKernel, cache: KVCache,
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

    /// One oracle sweep of a random chunk on a fresh cache.
    private func checkOracle(
        numHeads: Int, kvHeads: Int, headDim: Int, layers: Int = 1, layer: Int = 0,
        maxContext: Int, basePosition: Int, batch: Int, seed: UInt64,
        surface: String
    ) throws {
        let context = try makeContextOrSkip()
        let kernel = try PrefillTiledSDPAKernel(context: context, headDim: headDim)
        let cache = try KVCache(
            device: context.device, layers: layers, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        var rng = SplitMix64(seed: seed)
        let (ks, vs) = try fillCache(
            cache, layer: layer, depth: basePosition + batch, rng: &rng)
        let q = randomHalfs(count: batch * numHeads * headDim, rng: &rng)
        let out = try runTiled(
            context: context, kernel: kernel, cache: cache, layer: layer,
            basePosition: basePosition, q: q, batch: batch, numHeads: numHeads)
        try assertBatchMatchesOracle(
            out: readHalfs(out, count: batch * numHeads * headDim), q: q, ks: ks,
            vs: vs, numHeads: numHeads, kvHeads: kvHeads, headDim: headDim,
            basePosition: basePosition, batch: batch, surface: surface)
    }

    // MARK: - Fragment-ownership probe (the kernel's in-register softmax rests on it)

    private static let probeSource = """
    #include <metal_stdlib>
    #include <metal_simdgroup_matrix>
    using namespace metal;
    kernel void probe(device const float *src  [[buffer(0)]],
                      device const half *srcH  [[buffer(1)]],
                      device const half *ident [[buffer(2)]],
                      device float *owned      [[buffer(3)]],
                      device float *mma        [[buffer(4)]],
                      uint lane [[thread_index_in_simdgroup]]) {
        simdgroup_float8x8 f;
        simdgroup_load(f, src, 8);
        const thread float2 &e = reinterpret_cast<const thread float2 &>(f.thread_elements());
        owned[lane * 2 + 0] = e[0];
        owned[lane * 2 + 1] = e[1];
        simdgroup_half8x8 h;
        simdgroup_load(h, srcH, 8);
        const thread half2 &he = reinterpret_cast<const thread half2 &>(h.thread_elements());
        owned[64 + lane * 2 + 0] = float(he[0]);
        owned[64 + lane * 2 + 1] = float(he[1]);
        simdgroup_half8x8 a, b;
        simdgroup_load(a, srcH, 8);
        simdgroup_load(b, ident, 8);
        simdgroup_float8x8 c = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_multiply_accumulate(c, a, b, c);
        simdgroup_store(c, mma, 8);
    }
    """

    /// Pins, on the running device, the three hardware facts the kernel
    /// assumes: (1) lane t of a `simdgroup_float8x8` holds row
    /// (t/4 & 4) + (t/2 % 4), columns (t/4 & 2)·2 + (t%2)·2 and +1; (2) the
    /// half fragment shares that ownership; (3) half × half → float
    /// `simdgroup_multiply_accumulate` is exact on an identity product. A
    /// device where any of these differ fails here, not silently in the
    /// gates.
    func testFragmentOwnershipProbeMatchesTheKernelsAssumption() throws {
        let context = try makeContextOrSkip()
        let library = try context.makeLibrary(source: Self.probeSource)
        let pipeline = try context.makeComputePipeline(library: library, function: "probe")
        XCTAssertEqual(pipeline.threadExecutionWidth, 32,
                       "the ownership map is a 32-wide SIMD fact")
        let src = try makeBuffer(context.device, values: (0..<64).map { Float($0) })
        let srcH = try makeBuffer(context.device, values: (0..<64).map { Float16($0) })
        let ident = try makeBuffer(
            context.device,
            values: (0..<64).map { Float16($0 % 8 == $0 / 8 ? 1 : 0) })
        let owned = try makeOutputBuffer(context.device, count: 128, elementStride: 4)
        let mma = try makeOutputBuffer(context.device, count: 64, elementStride: 4)
        try context.timedDispatch { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(src, offset: 0, index: 0)
            encoder.setBuffer(srcH, offset: 0, index: 1)
            encoder.setBuffer(ident, offset: 0, index: 2)
            encoder.setBuffer(owned, offset: 0, index: 3)
            encoder.setBuffer(mma, offset: 0, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        let got = readFloats(owned, count: 128)
        for lane in 0..<32 {
            let qid = lane / 4
            let fm = (qid & 4) + ((lane / 2) % 4)
            let fn = (qid & 2) * 2 + (lane % 2) * 2
            XCTAssertEqual(Int(got[lane * 2]), fm * 8 + fn,
                           "lane \(lane) element 0 ownership")
            XCTAssertEqual(Int(got[lane * 2 + 1]), fm * 8 + fn + 1,
                           "lane \(lane) element 1 ownership")
            XCTAssertEqual(got[64 + lane * 2], got[lane * 2],
                           "half fragment ownership differs at lane \(lane)")
            XCTAssertEqual(got[64 + lane * 2 + 1], got[lane * 2 + 1])
        }
        XCTAssertEqual(readFloats(mma, count: 64), (0..<64).map { Float($0) },
                       "half×half→float MMA against the identity must be exact")
    }

    // MARK: - Oracle gates

    /// Small dims, one ragged tile from depth 0 (GQA 4→2: 2 heads × 16
    /// positions per tile, 12 used): every position within the gate.
    func testTiledSDPAMatchesOracleFromEmptyCache() throws {
        try checkOracle(
            numHeads: 4, kvHeads: 2, headDim: 16, maxContext: 32,
            basePosition: 0, batch: 12, seed: 601,
            surface: "tiled SDPA (small dims, one ragged tile)")
    }

    /// Real headDim 128 on layer 1 of 2 (offsets resolve per layer): a
    /// chunk starting at 37 covering 45 positions — three tiles (16 + 16 +
    /// 13 ragged), key range crossing three 32-key blocks, per-row masks
    /// on the diagonal blocks, and rows whose earlier blocks are fully
    /// masked (edge test 4 species + the tile/block seams).
    func testTiledSDPAMultiTileMultiBlockAtRealHeadDim() throws {
        try checkOracle(
            numHeads: 4, kvHeads: 2, headDim: 128, layers: 2, layer: 1,
            maxContext: 96, basePosition: 37, batch: 45, seed: 602,
            surface: "tiled SDPA (headDim 128, chunk at 37)")
    }

    /// The pinned model's head shape (16 q / 8 kv / headDim 128): a chunk
    /// at 70 covering 75 positions — 5 tiles × 8 head groups, 5 key blocks.
    func testTiledSDPAPinnedHeadShape() throws {
        try checkOracle(
            numHeads: 16, kvHeads: 8, headDim: 128, maxContext: 160,
            basePosition: 70, batch: 75, seed: 603,
            surface: "tiled SDPA (pinned 16/8/128, chunk at 70)")
    }

    /// The smallest supported headDim (8: one fragment along headDim, so
    /// only two of the four simdgroups take part in the K-tile transpose
    /// load — the doc's headDim ∈ {8, …, 128} claim, exercised on the
    /// device rather than argued): two ragged tiles, two key blocks.
    func testTiledSDPASmallestHeadDim() throws {
        try checkOracle(
            numHeads: 4, kvHeads: 2, headDim: 8, maxContext: 48,
            basePosition: 3, batch: 30, seed: 604,
            surface: "tiled SDPA (headDim 8)")
    }

    /// Every tiling the GQA ratio can select: 1, 2 and 4 heads per tile
    /// (32, 16, 8 positions), the ratio-8 case tiling 4 of 8 heads, and
    /// a ratio the tile cannot share (3 → one head per tile).
    func testTiledSDPAAcrossGQATilings() throws {
        XCTAssertEqual(PrefillTiledSDPAKernel.headsPerTile(groupSize: 1), 1)
        XCTAssertEqual(PrefillTiledSDPAKernel.headsPerTile(groupSize: 2), 2)
        XCTAssertEqual(PrefillTiledSDPAKernel.headsPerTile(groupSize: 3), 1)
        XCTAssertEqual(PrefillTiledSDPAKernel.headsPerTile(groupSize: 4), 4)
        XCTAssertEqual(PrefillTiledSDPAKernel.headsPerTile(groupSize: 6), 2)
        XCTAssertEqual(PrefillTiledSDPAKernel.headsPerTile(groupSize: 8), 4)
        XCTAssertEqual(PrefillTiledSDPAKernel.positionsPerTile(groupSize: 2), 16)
        let cases: [(numHeads: Int, kvHeads: Int, seed: UInt64)] = [
            (2, 2, 611), (8, 2, 612), (8, 1, 613), (6, 2, 614),
        ]
        for c in cases {
            try checkOracle(
                numHeads: c.numHeads, kvHeads: c.kvHeads, headDim: 16,
                maxContext: 64, basePosition: 5, batch: 41, seed: c.seed,
                surface: "tiled SDPA (GQA \(c.numHeads)/\(c.kvHeads))")
        }
    }

    // MARK: - Exactness and causality

    /// Depth-1 exactness (edge test 1 species): position 0's output is the
    /// mapped V row BITWISE, including -0.0, NaN payloads and subnormals;
    /// with finite V restored, every position gates against the oracle.
    func testTiledSDPADepthOneIsBitwiseVRowCopy() throws {
        let context = try makeContextOrSkip()
        let (numHeads, kvHeads, headDim, batch) = (4, 2, 8, 3)
        let kernel = try PrefillTiledSDPAKernel(context: context, headDim: headDim)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: 8, headDim: headDim)
        var rng = SplitMix64(seed: 620)
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
        let out = try runTiled(
            context: context, kernel: kernel, cache: cache, layer: 0,
            basePosition: 0, q: q, batch: batch, numHeads: numHeads)
        let bits = readBits(out, count: batch * numHeads * headDim)
        let expectedBits = adversarial.map(\.bitPattern)
        for head in 0..<numHeads {
            let row = Array(bits[(head * headDim)..<((head + 1) * headDim)])
            XCTAssertEqual(row, expectedBits,
                           "position 0 head \(head): not a bitwise V copy")
        }
        let finite = randomHalfs(count: headDim, rng: &rng)
        for head in 0..<kvHeads {
            try writeCache(cache, layer: 0, component: .value, head: head,
                           position: 0, values: finite)
            vs[0].replaceSubrange((head * headDim)..<((head + 1) * headDim),
                                  with: finite)
        }
        ks = ks.map { $0 }
        let out2 = try runTiled(
            context: context, kernel: kernel, cache: cache, layer: 0,
            basePosition: 0, q: q, batch: batch, numHeads: numHeads)
        try assertBatchMatchesOracle(
            out: readHalfs(out2, count: batch * numHeads * headDim), q: q, ks: ks,
            vs: vs, numHeads: numHeads, kvHeads: kvHeads, headDim: headDim,
            basePosition: 0, batch: batch, surface: "tiled SDPA after depth-1 check")
    }

    /// Slots at or beyond the chunk end are never READ: poisoning them
    /// with NaN leaves every output bitwise unchanged (the tiles zero-fill
    /// past keyEnd, so a NaN there cannot reach the matrix unit).
    func testTiledSDPANeverReadsBeyondChunkEnd() throws {
        let context = try makeContextOrSkip()
        let (numHeads, kvHeads, headDim, batch) = (4, 2, 16, 6)
        let kernel = try PrefillTiledSDPAKernel(context: context, headDim: headDim)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: 16, headDim: headDim)
        var rng = SplitMix64(seed: 621)
        _ = try fillCache(cache, layer: 0, depth: 16, rng: &rng)
        let q = randomHalfs(count: batch * numHeads * headDim, rng: &rng)
        let qDim = numHeads * headDim
        let clean = readBits(
            try runTiled(context: context, kernel: kernel, cache: cache, layer: 0,
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
        let poisoned = readBits(
            try runTiled(context: context, kernel: kernel, cache: cache, layer: 0,
                         basePosition: 2, q: q, batch: batch, numHeads: numHeads),
            count: batch * qDim)
        XCTAssertEqual(poisoned, clean, "slots beyond the chunk end were read")
    }

    /// Causality INSIDE the chunk (edge test 3): a later position's slot
    /// sits in the same tiles and key blocks as earlier rows, so it is
    /// loaded — but masked keys weigh exactly zero (exp(−∞) = 0 → P = 0
    /// → 0·V), so replacing later slots with large finite values (scores
    /// that would dominate any softmax if unmasked; V that would swamp any
    /// output) leaves every earlier position's output BITWISE unchanged.
    /// Checked on a single-tile chunk (last position poisoned) and across
    /// tile/block seams (the last 20 of 40 positions poisoned).
    func testTiledSDPACausalWithinChunkIsBitwiseInvariant() throws {
        let context = try makeContextOrSkip()
        let (numHeads, kvHeads, headDim) = (4, 2, 16)
        let kernel = try PrefillTiledSDPAKernel(context: context, headDim: headDim)
        let qDim = numHeads * headDim
        let bigK = [Float16](repeating: Float16(64.0), count: headDim)
        let bigV = [Float16](repeating: Float16(1000.0), count: headDim)

        func poisonedInvariance(
            maxContext: Int, basePosition: Int, batch: Int,
            poison: Range<Int>, seed: UInt64
        ) throws {
            let cache = try KVCache(
                device: context.device, layers: 1, kvHeads: kvHeads,
                maxContext: maxContext, headDim: headDim)
            var rng = SplitMix64(seed: seed)
            _ = try fillCache(cache, layer: 0, depth: basePosition + batch, rng: &rng)
            let q = randomHalfs(count: batch * qDim, rng: &rng)
            let clean = readBits(
                try runTiled(context: context, kernel: kernel, cache: cache, layer: 0,
                             basePosition: basePosition, q: q, batch: batch,
                             numHeads: numHeads),
                count: batch * qDim)
            for position in poison {
                for head in 0..<kvHeads {
                    try writeCache(cache, layer: 0, component: .key, head: head,
                                   position: position, values: bigK)
                    try writeCache(cache, layer: 0, component: .value, head: head,
                                   position: position, values: bigV)
                }
            }
            let poisoned = readBits(
                try runTiled(context: context, kernel: kernel, cache: cache, layer: 0,
                             basePosition: basePosition, q: q, batch: batch,
                             numHeads: numHeads),
                count: batch * qDim)
            let unaffectedRows = poison.lowerBound - basePosition
            XCTAssertEqual(
                Array(poisoned[0..<(unaffectedRows * qDim)]),
                Array(clean[0..<(unaffectedRows * qDim)]),
                "an earlier position read a later slot (chunk at \(basePosition), "
                    + "\(batch) positions, poison \(poison))")
            // The poisoned positions themselves DO change (the poison is
            // in their own causal window) — proves the poison was live.
            XCTAssertNotEqual(
                Array(poisoned[(unaffectedRows * qDim)...]),
                Array(clean[(unaffectedRows * qDim)...]),
                "poison did not reach the positions that may read it")
        }
        try poisonedInvariance(maxContext: 16, basePosition: 2, batch: 6,
                               poison: 7..<8, seed: 622)
        try poisonedInvariance(maxContext: 64, basePosition: 0, batch: 40,
                               poison: 20..<40, seed: 623)
    }

    /// Bitwise deterministic across runs (fixed-order reductions), and
    /// exactly one dispatch per encode.
    func testTiledSDPADeterministicAndOneDispatch() throws {
        let context = try makeContextOrSkip()
        let (numHeads, kvHeads, headDim, batch) = (8, 4, 64, 20)
        let kernel = try PrefillTiledSDPAKernel(context: context, headDim: headDim)
        let counter = DispatchCounter()
        kernel.dispatchCounter = counter
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: 32, headDim: headDim)
        var rng = SplitMix64(seed: 630)
        _ = try fillCache(cache, layer: 0, depth: batch, rng: &rng)
        let q = randomHalfs(count: batch * numHeads * headDim, rng: &rng)
        let first = readBits(
            try runTiled(context: context, kernel: kernel, cache: cache, layer: 0,
                         basePosition: 0, q: q, batch: batch, numHeads: numHeads),
            count: batch * numHeads * headDim)
        XCTAssertEqual(counter.count, 1, "one dispatch per tiled encode")
        for _ in 0..<3 {
            let again = readBits(
                try runTiled(context: context, kernel: kernel, cache: cache, layer: 0,
                             basePosition: 0, q: q, batch: batch, numHeads: numHeads),
                count: batch * numHeads * headDim)
            XCTAssertEqual(again, first, "tiled SDPA must be bitwise deterministic")
        }
    }

    /// Both prefill attention kernels gate against the same oracle, so
    /// their mutual difference is bounded by twice the attention constant
    /// (a derived bound, not a new one).
    func testTiledMatchesPerPositionKernelWithinDerivedBound() throws {
        let context = try makeContextOrSkip()
        let (numHeads, kvHeads, headDim, batch) = (4, 2, 32, 10)
        let tiled = try PrefillTiledSDPAKernel(context: context, headDim: headDim)
        let perPosition = try PrefillSDPAKernel(context: context)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: kvHeads,
            maxContext: 16, headDim: headDim)
        var rng = SplitMix64(seed: 640)
        _ = try fillCache(cache, layer: 0, depth: batch, rng: &rng)
        let q = randomHalfs(count: batch * numHeads * headDim, rng: &rng)
        let qDim = numHeads * headDim
        let tiledOut = readHalfs(
            try runTiled(context: context, kernel: tiled, cache: cache, layer: 0,
                         basePosition: 0, q: q, batch: batch, numHeads: numHeads),
            count: batch * qDim).map(Float.init)
        let qBuffer = try makeBuffer(context.device, values: q)
        let batchedOut = try makeOutputBuffer(context.device, count: batch * qDim, elementStride: 2)
        try context.timedDispatch { encoder in
            try perPosition.encodeCausalSDPABatch(
                into: encoder, cache: cache, layer: 0, basePosition: 0, batch: batch,
                query: qBuffer, numHeads: numHeads, output: batchedOut)
        }
        let batched = readHalfs(batchedOut, count: batch * qDim).map(Float.init)
        let m = batched.reduce(Float(0)) { max($0, abs($1)) }
        let bound = 2 * max(exp2(-7) * m, exp2(-11))
        var worst: Float = 0
        for i in 0..<batched.count { worst = max(worst, abs(tiledOut[i] - batched[i])) }
        XCTAssertLessThanOrEqual(worst, bound, "tiled vs per-position kernel: |Δ| \(worst)")
    }

    func testTiledSDPARejectsBadInputs() throws {
        let context = try makeContextOrSkip()
        XCTAssertThrowsError(try PrefillTiledSDPAKernel(context: context, headDim: 12),
                             "headDim not a multiple of 8")
        XCTAssertThrowsError(try PrefillTiledSDPAKernel(context: context, headDim: 0))
        XCTAssertThrowsError(try PrefillTiledSDPAKernel(context: context, headDim: 136),
                             "headDim above the fragment budget")
        let kernel = try PrefillTiledSDPAKernel(context: context, headDim: 16)
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: 2, maxContext: 8, headDim: 16)
        let q = try makeOutputBuffer(context.device, count: 8 * 4 * 16, elementStride: 2)
        let out = try makeOutputBuffer(context.device, count: 8 * 4 * 16, elementStride: 2)
        func encode(basePosition: Int, batch: Int, numHeads: Int = 4,
                    query: MTLBuffer? = nil, cacheOverride: KVCache? = nil) throws {
            try context.timedDispatch { encoder in
                try kernel.encodeCausalSDPABatch(
                    into: encoder, cache: cacheOverride ?? cache, layer: 0,
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
        let other = try KVCache(
            device: context.device, layers: 1, kvHeads: 2, maxContext: 8, headDim: 32)
        XCTAssertThrowsError(try encode(basePosition: 0, batch: 2, cacheOverride: other),
                             "cache headDim differs from the specialized instance")
    }
}
