import XCTest
@testable import QwenMetalEngine
import Metal

/// P4-3 fold-set tests (docs/phases/phase-4.md D3/D5, edge tests 6-8; gates
/// pre-committed in DECISIONS.md 2026-09-05 — no new constants):
///
///   Matvec-only spans (matvec3 QKV concat, gate+up+SwiGLU, residual-folded
///   matvec) diff against the UNFUSED kernel chain at Tier K
///   max(2⁻⁹·M, 2⁻¹¹), on odd synthetic dims AND the pinned model's real
///   dims (spec edge test 7 — proves the folds changed structure, not
///   arithmetic).
///
///   The qk-norm/RoPE/append cluster diffs against the CPU fp32 reference
///   (the CPU-quant oracle's norm+rope arithmetic) at the norm-species gate
///   max(2⁻⁸·M, 2⁻¹¹), at RoPE positions {0, 1, large} (edge test 6). The
///   v-side append remains a pure copy and stays EXACT bitwise, adversarial
///   bit patterns included; the k-side write is a computed value gated at
///   the mapped tolerance (approved gates entry). Context-limit behavior
///   carries the encodeKVAppend contract: throw pre-dispatch, cache
///   untouched.
///
/// Hard rule 3: these tests exist WITH the kernels; every optimization
/// iteration must re-pass them.
final class FoldedKernelTests: XCTestCase {

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

    // MARK: - Synthetic triplets

    /// A q4g64 triplet already on the GPU. Both the unfused QuantKernels
    /// chain and the folded kernels bind the SAME buffers, so any output
    /// difference is kernel structure, never data.
    private struct GPUTriplet {
        let outDim: Int
        let inDim: Int
        let q: MTLBuffer
        let scales: MTLBuffer
        let biases: MTLBuffer

        var folded: FoldedKernels.Triplet {
            FoldedKernels.Triplet(
                q: q, qByteOffset: 0, scales: scales, scalesByteOffset: 0,
                biases: biases, biasesByteOffset: 0)
        }
    }

    /// Random codes/scales/biases directly (any valid triplet is a legal
    /// kernel input; tying codes to the pinned packing recipe is
    /// Q4PackerTests' job). Ranges keep every row dot well inside fp16.
    private func randomTriplet(
        outDim: Int, inDim: Int, device: MTLDevice, rng: inout SplitMix64
    ) throws -> GPUTriplet {
        let words = (0..<(outDim * inDim / 8)).map { _ in
            UInt32(truncatingIfNeeded: rng.next())
        }
        let groups = outDim * inDim / 64
        let scales = (0..<groups).map { _ in
            Float16(Float.random(in: 0.002...0.05, using: &rng))
        }
        let biases = (0..<groups).map { _ in
            Float16(Float.random(in: -0.4...0.4, using: &rng))
        }
        return GPUTriplet(
            outDim: outDim, inDim: inDim,
            q: try makeBuffer(device, values: words),
            scales: try makeBuffer(device, values: scales),
            biases: try makeBuffer(device, values: biases))
    }

    private func randomHalfs(
        _ count: Int, range: ClosedRange<Float> = -1...1, rng: inout SplitMix64
    ) -> [Float16] {
        (0..<count).map { _ in Float16(Float.random(in: range, using: &rng)) }
    }

    /// bf16 bit pattern of an EXACTLY representable fp32 value.
    private func bf16Bits(_ v: Float) -> UInt16 {
        let bits = UInt16(truncatingIfNeeded: v.bitPattern >> 16)
        precondition(Float(bitPattern: UInt32(bits) << 16) == v,
                     "test value \(v) is not bf16-exact")
        return bits
    }

    // MARK: - GPU harness helpers (QuantKernelTests pattern)

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

    // MARK: - Gate assertions (pre-committed; never loosened)

    /// Tier K max(2⁻⁹·M, 2⁻¹¹) — matvec/elementwise-only fused spans vs the
    /// unfused chain (spec D5 mapping rule).
    private func assertTierK(
        _ got: [Float16], _ ref: [Float16], _ surface: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        assertWithinGate(got.map(Float.init), ref.map(Float.init),
                         exponent: -9, surface: surface, file: file, line: line)
    }

    /// Norm-species max(2⁻⁸·M, 2⁻¹¹) — the qk-norm/rope/append cluster vs
    /// the CPU fp32 reference (spec D5 mapping rule).
    private func assertNormSpecies(
        _ got: [Float16], _ ref: [Float], _ surface: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        assertWithinGate(got.map(Float.init), ref, exponent: -8,
                         surface: surface, file: file, line: line)
    }

    private func assertWithinGate(
        _ got: [Float], _ ref: [Float], exponent: Int, surface: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.count, ref.count, "\(surface): count mismatch",
                       file: file, line: line)
        let m = ref.reduce(Float(0)) { max($0, abs($1)) }
        let gate = max(exp2(Float(exponent)) * m, exp2(-11))
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
                + "pre-committed gate \(gate) (M = \(m)): got "
                + "\(got[worstIndex]), ref \(ref[worstIndex])",
            file: file, line: line)
    }

    private func assertBitwiseEqualHalfs(
        _ got: [Float16], _ ref: [Float16], _ surface: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.count, ref.count, "\(surface): count mismatch",
                       file: file, line: line)
        for i in 0..<ref.count where got[i].bitPattern != ref[i].bitPattern {
            return XCTFail(
                "\(surface): first bitwise mismatch at [\(i)]: got "
                    + "0x\(String(got[i].bitPattern, radix: 16)), ref "
                    + "0x\(String(ref[i].bitPattern, radix: 16))",
                file: file, line: line)
        }
    }

    // MARK: - Edge test 7: matvec3 (QKV concat) vs the unfused chain

    private func runMatvec3Comparison(
        outA: Int, outB: Int, outC: Int, inDim: Int, seed: UInt64
    ) throws {
        let context = try makeContextOrSkip()
        let device = context.device
        var rng = SplitMix64(seed: seed)
        let a = try randomTriplet(outDim: outA, inDim: inDim, device: device, rng: &rng)
        let b = try randomTriplet(outDim: outB, inDim: inDim, device: device, rng: &rng)
        let c = try randomTriplet(outDim: outC, inDim: inDim, device: device, rng: &rng)
        let x = try makeBuffer(device, values: randomHalfs(inDim, rng: &rng))

        let quant = try QuantKernels(context: context)
        let folded = try FoldedKernels(context: context)

        // Unfused reference chain: three standalone matvecs.
        let refA = try makeOutputBuffer(device, count: outA, elementStride: 2)
        let refB = try makeOutputBuffer(device, count: outB, elementStride: 2)
        let refC = try makeOutputBuffer(device, count: outC, elementStride: 2)
        try context.timedDispatch { encoder in
            for (t, out) in [(a, refA), (b, refB), (c, refC)] {
                try quant.encodeMatvec(
                    into: encoder, q: t.q, qByteOffset: 0,
                    scales: t.scales, scalesByteOffset: 0,
                    biases: t.biases, biasesByteOffset: 0,
                    input: x, outDim: t.outDim, inDim: inDim, output: out)
            }
        }

        let out = try makeOutputBuffer(
            device, count: outA + outB + outC, elementStride: 2)
        try context.timedDispatch { encoder in
            try folded.encodeMatvec3(
                into: encoder, a: a.folded, outDimA: outA,
                b: b.folded, outDimB: outB, c: c.folded, outDimC: outC,
                inDim: inDim, input: x, output: out)
        }

        let got = readHalfs(out, count: outA + outB + outC)
        assertTierK(Array(got[0..<outA]), readHalfs(refA, count: outA),
                    "matvec3 segment A")
        assertTierK(Array(got[outA..<(outA + outB)]), readHalfs(refB, count: outB),
                    "matvec3 segment B")
        assertTierK(Array(got[(outA + outB)...]), readHalfs(refC, count: outC),
                    "matvec3 segment C")
    }

    func testMatvec3MatchesUnfusedChainOnOddDims() throws {
        try runMatvec3Comparison(outA: 67, outB: 33, outC: 45, inDim: 128, seed: 11)
    }

    /// Real QKV dims of the pinned model (2048 | 1024 | 1024 rows × 2048).
    func testMatvec3MatchesUnfusedChainOnRealDims() throws {
        try runMatvec3Comparison(
            outA: 2048, outB: 1024, outC: 1024, inDim: 2048, seed: 12)
    }

    // MARK: - Edge test 7: gate+up+SwiGLU vs the unfused chain

    private func runGateUpComparison(outDim: Int, inDim: Int, seed: UInt64) throws {
        let context = try makeContextOrSkip()
        let device = context.device
        var rng = SplitMix64(seed: seed)
        let gate = try randomTriplet(outDim: outDim, inDim: inDim, device: device, rng: &rng)
        let up = try randomTriplet(outDim: outDim, inDim: inDim, device: device, rng: &rng)
        let x = try makeBuffer(device, values: randomHalfs(inDim, rng: &rng))

        let quant = try QuantKernels(context: context)
        let decode = try DecodeKernels(context: context)
        let folded = try FoldedKernels(context: context)

        // Unfused reference chain: gate matvec, up matvec, standalone SwiGLU.
        let gateOut = try makeOutputBuffer(device, count: outDim, elementStride: 2)
        let upOut = try makeOutputBuffer(device, count: outDim, elementStride: 2)
        let ref = try makeOutputBuffer(device, count: outDim, elementStride: 2)
        try context.timedDispatch { encoder in
            for (t, out) in [(gate, gateOut), (up, upOut)] {
                try quant.encodeMatvec(
                    into: encoder, q: t.q, qByteOffset: 0,
                    scales: t.scales, scalesByteOffset: 0,
                    biases: t.biases, biasesByteOffset: 0,
                    input: x, outDim: outDim, inDim: inDim, output: out)
            }
            try decode.encodeSwiGLU(
                into: encoder, gate: gateOut, up: upOut, count: outDim,
                output: ref)
        }

        let out = try makeOutputBuffer(device, count: outDim, elementStride: 2)
        try context.timedDispatch { encoder in
            try folded.encodeGateUpSwiGLU(
                into: encoder, gate: gate.folded, up: up.folded,
                outDim: outDim, inDim: inDim, input: x, output: out)
        }
        assertTierK(readHalfs(out, count: outDim), readHalfs(ref, count: outDim),
                    "gate+up+SwiGLU fold")
    }

    func testGateUpSwiGLUMatchesUnfusedChainOnOddDims() throws {
        try runGateUpComparison(outDim: 51, inDim: 192, seed: 21)
    }

    /// Real MLP dims of the pinned model (6144 × 2048).
    func testGateUpSwiGLUMatchesUnfusedChainOnRealDims() throws {
        try runGateUpComparison(outDim: 6144, inDim: 2048, seed: 22)
    }

    // MARK: - Edge test 7: residual-folded matvec vs the unfused chain

    private func runMatvecResidualComparison(
        outDim: Int, inDim: Int, seed: UInt64
    ) throws {
        let context = try makeContextOrSkip()
        let device = context.device
        var rng = SplitMix64(seed: seed)
        let t = try randomTriplet(outDim: outDim, inDim: inDim, device: device, rng: &rng)
        let x = try makeBuffer(device, values: randomHalfs(inDim, rng: &rng))
        let res = try makeBuffer(device, values: randomHalfs(outDim, rng: &rng))

        let quant = try QuantKernels(context: context)
        let decode = try DecodeKernels(context: context)
        let folded = try FoldedKernels(context: context)

        // Unfused reference chain: standalone matvec, standalone residual add.
        let proj = try makeOutputBuffer(device, count: outDim, elementStride: 2)
        let ref = try makeOutputBuffer(device, count: outDim, elementStride: 2)
        try context.timedDispatch { encoder in
            try quant.encodeMatvec(
                into: encoder, q: t.q, qByteOffset: 0,
                scales: t.scales, scalesByteOffset: 0,
                biases: t.biases, biasesByteOffset: 0,
                input: x, outDim: outDim, inDim: inDim, output: proj)
            try decode.encodeResidualAdd(
                into: encoder, a: res, b: proj, count: outDim, output: ref)
        }

        let out = try makeOutputBuffer(device, count: outDim, elementStride: 2)
        try context.timedDispatch { encoder in
            try folded.encodeMatvecResidual(
                into: encoder, triplet: t.folded, outDim: outDim, inDim: inDim,
                input: x, residual: res, output: out)
        }
        assertTierK(readHalfs(out, count: outDim), readHalfs(ref, count: outDim),
                    "residual-folded matvec")
    }

    func testMatvecResidualMatchesUnfusedChainOnOddDims() throws {
        try runMatvecResidualComparison(outDim: 77, inDim: 64, seed: 31)
    }

    /// Real dims of the pinned model's two residual-folded matvecs:
    /// o_proj (2048 × 2048) and down_proj (2048 × 6144).
    func testMatvecResidualMatchesUnfusedChainOnRealDims() throws {
        try runMatvecResidualComparison(outDim: 2048, inDim: 2048, seed: 32)
        try runMatvecResidualComparison(outDim: 2048, inDim: 6144, seed: 33)
    }

    // MARK: - Edge test 6: the qk-norm/RoPE/append cluster

    /// Small Qwen3-shaped attention head geometry for cluster tests.
    private struct ClusterFixture {
        let context: MetalContext
        let kernels: FoldedKernels
        let cache: KVCache
        let numHeads: Int
        let headDim: Int
        let maxContext: Int
        let eps: Float
        let qNormValues: [Float]
        let kNormValues: [Float]
        let qNormBuffer: MTLBuffer
        let kNormBuffer: MTLBuffer
        let cosBuffer: MTLBuffer
        let sinBuffer: MTLBuffer
        let rope: RoPE

        var kvHeads: Int { cache.kvHeads }
        var totalHeads: Int { numHeads + 2 * cache.kvHeads }
    }

    private func makeClusterFixture(
        numHeads: Int = 4, kvHeads: Int = 2, headDim: Int = 16,
        maxContext: Int = 40, eps: Float = 1e-6,
        qNormValues: [Float]? = nil, kNormValues: [Float]? = nil
    ) throws -> ClusterFixture {
        let context = try makeContextOrSkip()
        // bf16-exact norm weights (1/32 steps), distinct q vs k so a
        // norm-vector mixup is visible.
        let qValues = qNormValues
            ?? (0..<headDim).map { Float(($0 * 5 % 37) - 18) * 0.03125 }
        let kValues = kNormValues
            ?? (0..<headDim).map { Float(($0 * 11 % 41) - 20) * 0.03125 }
        let rope = try RoPE(
            headDim: headDim, theta: 10000, positions: maxContext)
        return ClusterFixture(
            context: context,
            kernels: try FoldedKernels(context: context),
            cache: try KVCache(
                device: context.device, layers: 1, kvHeads: kvHeads,
                maxContext: maxContext, headDim: headDim),
            numHeads: numHeads, headDim: headDim, maxContext: maxContext,
            eps: eps, qNormValues: qValues, kNormValues: kValues,
            qNormBuffer: try makeBuffer(context.device, values: qValues.map(bf16Bits)),
            kNormBuffer: try makeBuffer(context.device, values: kValues.map(bf16Bits)),
            cosBuffer: try makeBuffer(context.device, values: rope.cosValues),
            sinBuffer: try makeBuffer(context.device, values: rope.sinValues),
            rope: rope)
    }

    /// Encodes the cluster over `input` at `position`, returning the roped
    /// q heads and the cache K/V slot contents at that position.
    private func runCluster(
        _ f: ClusterFixture, input: [Float16], position: Int
    ) throws -> (q: [Float16], k: [Float16], v: [Float16]) {
        let qkv = try makeBuffer(f.context.device, values: input)
        let qOut = try makeOutputBuffer(
            f.context.device, count: f.numHeads * f.headDim, elementStride: 2)
        try f.context.timedDispatch { encoder in
            try f.kernels.encodeQKNormRoPEAppend(
                into: encoder, qkv: qkv,
                qNormWeight: f.qNormBuffer, qNormByteOffset: 0,
                kNormWeight: f.kNormBuffer, kNormByteOffset: 0,
                cosTable: f.cosBuffer, sinTable: f.sinBuffer,
                position: position, positions: f.maxContext,
                numHeads: f.numHeads, eps: f.eps,
                cache: f.cache, layer: 0, qOut: qOut)
        }
        var k: [Float16] = []
        var v: [Float16] = []
        let cacheHalfs = readHalfs(
            f.cache.buffer, count: f.cache.buffer.length / 2)
        for component in [KVCache.Component.key, KVCache.Component.value] {
            for head in 0..<f.kvHeads {
                let offset = try f.cache.elementOffset(
                    layer: 0, component: component, head: head, position: position)
                let slot = Array(cacheHalfs[offset..<(offset + f.headDim)])
                if component == .key { k += slot } else { v += slot }
            }
        }
        return (readHalfs(qOut, count: f.numHeads * f.headDim), k, v)
    }

    /// CPU fp32 reference for one norm+rope head: the CPU-quant oracle's
    /// arithmetic (the RMSNorm module + the same fp32 RoPE tables the GPU
    /// consumes).
    private func cpuNormRope(
        row: ArraySlice<Float16>, normValues: [Float], eps: Float,
        rope: RoPE, position: Int
    ) throws -> [Float] {
        let norm = RMSNorm(weight: normValues, eps: eps)
        var out = try norm(row.map(Float.init))
        let half = out.count / 2
        for i in 0..<half {
            let c = rope.cosValues[position * half + i]
            let sn = rope.sinValues[position * half + i]
            let x1 = out[i]
            let x2 = out[half + i]
            out[i] = x1 * c - x2 * sn
            out[half + i] = x2 * c + x1 * sn
        }
        return out
    }

    /// Edge test 6: q output and k cache contents vs the CPU fp32 reference
    /// within the norm-species gate at RoPE positions {0, 1, large}; the v
    /// slot bitwise-equals its input rows at every position.
    func testClusterMatchesCPUReferenceAtBoundaryPositions() throws {
        let f = try makeClusterFixture()
        var rng = SplitMix64(seed: 41)
        for position in [0, 1, f.maxContext - 1] {
            let input = randomHalfs(f.totalHeads * f.headDim, rng: &rng)
            let got = try runCluster(f, input: input, position: position)

            var qRef: [Float] = []
            for head in 0..<f.numHeads {
                let base = head * f.headDim
                qRef += try cpuNormRope(
                    row: input[base..<(base + f.headDim)],
                    normValues: f.qNormValues, eps: f.eps,
                    rope: f.rope, position: position)
            }
            assertNormSpecies(got.q, qRef, "cluster q output @p=\(position)")

            var kRef: [Float] = []
            for head in 0..<f.kvHeads {
                let base = (f.numHeads + head) * f.headDim
                kRef += try cpuNormRope(
                    row: input[base..<(base + f.headDim)],
                    normValues: f.kNormValues, eps: f.eps,
                    rope: f.rope, position: position)
            }
            assertNormSpecies(got.k, kRef, "cluster k cache slot @p=\(position)")

            let vBase = (f.numHeads + f.kvHeads) * f.headDim
            assertBitwiseEqualHalfs(
                got.v, Array(input[vBase...]),
                "cluster v cache slot @p=\(position)")
        }
    }

    /// Edge test 6 head-mapping (P2-3 small-dims exact construction): rows
    /// of ±c with c a power of two and eps 0 make invRMS exactly 1/c, so
    /// each output row is exactly its norm-weight vector times the row's
    /// sign pattern (position 0 ⇒ the rotation is exact identity for these
    /// values). Distinct sign patterns per head and distinct q/k norm
    /// vectors make any head-mapping or norm-selection mixup a hard
    /// exact-value failure.
    func testClusterHeadMappingAndNormSelectionExact() throws {
        let headDim = 4
        let qNorm: [Float] = [1.0, 2.0, 0.5, 4.0]
        let kNorm: [Float] = [8.0, 0.25, 2.0, 1.0]
        let f = try makeClusterFixture(
            numHeads: 4, kvHeads: 2, headDim: headDim, maxContext: 8,
            eps: 0, qNormValues: qNorm, kNormValues: kNorm)

        // Sign patterns, distinct per role-head (4 q, 2 k, 2 v).
        let signs: [[Float]] = [
            [1, 1, 1, 1], [1, -1, 1, -1], [-1, 1, 1, -1], [-1, -1, -1, -1],
            [1, 1, -1, -1], [-1, 1, -1, 1],
            [1, -1, -1, 1], [-1, -1, 1, 1],
        ]
        let c: Float = 0.5
        let input = signs.flatMap { row in row.map { Float16($0 * c) } }
        let got = try runCluster(f, input: input, position: 0)

        let qExpected = (0..<4).flatMap { head in
            (0..<headDim).map { Float16(qNorm[$0] * signs[head][$0]) }
        }
        assertBitwiseEqualHalfs(got.q, qExpected, "head-mapped q output")
        let kExpected = (0..<2).flatMap { head in
            (0..<headDim).map { Float16(kNorm[$0] * signs[4 + head][$0]) }
        }
        assertBitwiseEqualHalfs(got.k, kExpected, "head-mapped k cache slot")
        let vExpected = (0..<2).flatMap { head in
            (0..<headDim).map { Float16(signs[6 + head][$0] * c) }
        }
        assertBitwiseEqualHalfs(got.v, vExpected, "head-mapped v cache slot")
    }

    /// The v-side append is a pure copy: adversarial fp16 bit patterns (NaN
    /// payloads, ±inf, subnormals, -0, boundary values) survive bitwise —
    /// the kv_append_f16 exactness claim carried to the fused cluster.
    func testClusterVSideAppendBitwiseExactAdversarialPatterns() throws {
        let f = try makeClusterFixture()
        let patterns: [UInt16] = [
            0x7C01, 0xFC01, 0x7FFF,         // NaN payloads
            0x7C00, 0xFC00,                 // ±inf
            0x0001, 0x8001, 0x03FF,         // subnormals
            0x8000, 0x0000,                 // ±0
            0x7BFF, 0xFBFF, 0x3C00,         // ±max normal, 1.0
        ]
        var rng = SplitMix64(seed: 51)
        var input = randomHalfs(f.totalHeads * f.headDim, rng: &rng)
        let vBase = (f.numHeads + f.kvHeads) * f.headDim
        for i in vBase..<input.count {
            input[i] = Float16(bitPattern: patterns[i % patterns.count])
        }
        let got = try runCluster(f, input: input, position: 3)
        assertBitwiseEqualHalfs(
            got.v, Array(input[vBase...]), "adversarial v cache slot")
    }

    /// The cluster is bitwise deterministic across runs (the pipeline's
    /// incremental-replay contract depends on it).
    func testClusterDeterministicAcrossRuns() throws {
        let f = try makeClusterFixture()
        var rng = SplitMix64(seed: 61)
        let input = randomHalfs(f.totalHeads * f.headDim, rng: &rng)
        let first = try runCluster(f, input: input, position: 7)
        let second = try runCluster(f, input: input, position: 7)
        assertBitwiseEqualHalfs(second.q, first.q, "determinism: q")
        assertBitwiseEqualHalfs(second.k, first.k, "determinism: k")
        assertBitwiseEqualHalfs(second.v, first.v, "determinism: v")
    }

    /// Edge test 3 contract carried over: encoding at position == maxContext
    /// throws `.contextFull` BEFORE any dispatch (even when the rope table
    /// is larger) and the cache bytes stay untouched.
    func testClusterContextFullThrowsPreDispatchCacheUntouched() throws {
        let context = try makeContextOrSkip()
        let kernels = try FoldedKernels(context: context)
        let headDim = 16
        let maxContext = 4
        let cache = try KVCache(
            device: context.device, layers: 1, kvHeads: 2,
            maxContext: maxContext, headDim: headDim)
        // Sentinel-fill the cache so "untouched" is a real byte comparison.
        let cacheCount = cache.buffer.length / 2
        cache.buffer.contents().withMemoryRebound(
            to: Float16.self, capacity: cacheCount
        ) { p in
            for i in 0..<cacheCount { p[i] = Float16(bitPattern: 0x5AA5) }
        }
        let before = readHalfs(cache.buffer, count: cacheCount)

        // Rope table larger than the cache: contextFull must still win.
        let rope = try RoPE(headDim: headDim, theta: 10000, positions: 8)
        let cosBuffer = try makeBuffer(context.device, values: rope.cosValues)
        let sinBuffer = try makeBuffer(context.device, values: rope.sinValues)
        let norm = try makeBuffer(
            context.device, values: (0..<headDim).map { _ in bf16Bits(1.0) })
        let qkv = try makeBuffer(
            context.device, values: [Float16](repeating: 0.5, count: 8 * headDim))
        let qOut = try makeOutputBuffer(
            context.device, count: 4 * headDim, elementStride: 2)

        XCTAssertThrowsError(
            try context.timedDispatch { encoder in
                try kernels.encodeQKNormRoPEAppend(
                    into: encoder, qkv: qkv,
                    qNormWeight: norm, qNormByteOffset: 0,
                    kNormWeight: norm, kNormByteOffset: 0,
                    cosTable: cosBuffer, sinTable: sinBuffer,
                    position: maxContext, positions: 8,
                    numHeads: 4, eps: 1e-6, cache: cache, layer: 0, qOut: qOut)
            }
        ) { error in
            XCTAssertEqual(
                error as? KVCacheError,
                .contextFull(position: maxContext, maxContext: maxContext))
        }
        assertBitwiseEqualHalfs(
            readHalfs(cache.buffer, count: cacheCount), before,
            "cache after refused append")
    }

    /// Wrapper validation: bad inputs are refused loudly before any dispatch.
    func testFoldedKernelsRejectBadInputs() throws {
        let context = try makeContextOrSkip()
        let device = context.device
        let kernels = try FoldedKernels(context: context)
        var rng = SplitMix64(seed: 71)
        let t = try randomTriplet(outDim: 8, inDim: 64, device: device, rng: &rng)
        let x = try makeBuffer(device, values: randomHalfs(64, rng: &rng))
        let out = try makeOutputBuffer(device, count: 24, elementStride: 2)

        func encodeExpecting<E: Error & Equatable>(
            _ expected: E, _ body: @escaping (MTLComputeCommandEncoder) throws -> Void,
            file: StaticString = #filePath, line: UInt = #line
        ) throws {
            XCTAssertThrowsError(
                try context.timedDispatch { encoder in try body(encoder) },
                file: file, line: line
            ) { error in
                XCTAssertEqual(error as? E, expected,
                               "got \(error)", file: file, line: line)
            }
        }

        // inDim not a multiple of the q4g64 group size.
        try encodeExpecting(QuantKernelError.inDimNotMultipleOfGroup(inDim: 60)) {
            try kernels.encodeMatvec3(
                into: $0, a: t.folded, outDimA: 8, b: t.folded, outDimB: 8,
                c: t.folded, outDimC: 8, inDim: 60, input: x, output: out)
        }
        // Output too small for the concatenated rows (3 × 8 = 24 halfs).
        let shortOut = try makeOutputBuffer(device, count: 20, elementStride: 2)
        try encodeExpecting(QuantKernelError.bufferTooSmall(
            buffer: "output", requiredBytes: 48, actualBytes: 40)) {
            try kernels.encodeMatvec3(
                into: $0, a: t.folded, outDimA: 8, b: t.folded, outDimB: 8,
                c: t.folded, outDimC: 8, inDim: 64, input: x, output: shortOut)
        }
        // Misaligned q byte offset on the gate triplet.
        let misaligned = FoldedKernels.Triplet(
            q: t.q, qByteOffset: 2, scales: t.scales, scalesByteOffset: 0,
            biases: t.biases, biasesByteOffset: 0)
        try encodeExpecting(QuantKernelError.misalignedOffset(
            buffer: "q", byteOffset: 2, alignment: 4)) {
            try kernels.encodeGateUpSwiGLU(
                into: $0, gate: misaligned, up: t.folded, outDim: 8,
                inDim: 64, input: x, output: out)
        }
        // Residual too small.
        let shortRes = try makeOutputBuffer(device, count: 4, elementStride: 2)
        try encodeExpecting(QuantKernelError.bufferTooSmall(
            buffer: "residual", requiredBytes: 16, actualBytes: 8)) {
            try kernels.encodeMatvecResidual(
                into: $0, triplet: t.folded, outDim: 8, inDim: 64,
                input: x, residual: shortRes, output: out)
        }

        // Cluster: odd headDim, rope table exceeded, misaligned norm offset.
        let oddCache = try KVCache(
            device: device, layers: 1, kvHeads: 2, maxContext: 8, headDim: 15)
        let rope = try RoPE(headDim: 16, theta: 10000, positions: 8)
        let cosBuffer = try makeBuffer(device, values: rope.cosValues)
        let sinBuffer = try makeBuffer(device, values: rope.sinValues)
        let norm = try makeBuffer(
            device, values: (0..<16).map { _ in bf16Bits(1.0) })
        let qkv = try makeBuffer(
            device, values: [Float16](repeating: 0.5, count: 8 * 16))
        try encodeExpecting(DecodeKernelError.oddHeadDim(headDim: 15)) {
            try kernels.encodeQKNormRoPEAppend(
                into: $0, qkv: qkv, qNormWeight: norm, qNormByteOffset: 0,
                kNormWeight: norm, kNormByteOffset: 0,
                cosTable: cosBuffer, sinTable: sinBuffer,
                position: 0, positions: 8, numHeads: 4, eps: 1e-6,
                cache: oddCache, layer: 0, qOut: out)
        }
        let evenCache = try KVCache(
            device: device, layers: 1, kvHeads: 2, maxContext: 8, headDim: 16)
        let qOut = try makeOutputBuffer(device, count: 4 * 16, elementStride: 2)
        try encodeExpecting(DecodeKernelError.positionOutOfRange(
            position: 5, positions: 4)) {
            try kernels.encodeQKNormRoPEAppend(
                into: $0, qkv: qkv, qNormWeight: norm, qNormByteOffset: 0,
                kNormWeight: norm, kNormByteOffset: 0,
                cosTable: cosBuffer, sinTable: sinBuffer,
                position: 5, positions: 4, numHeads: 4, eps: 1e-6,
                cache: evenCache, layer: 0, qOut: qOut)
        }
        try encodeExpecting(DecodeKernelError.misalignedWeightOffset(byteOffset: 1)) {
            try kernels.encodeQKNormRoPEAppend(
                into: $0, qkv: qkv, qNormWeight: norm, qNormByteOffset: 1,
                kNormWeight: norm, kNormByteOffset: 0,
                cosTable: cosBuffer, sinTable: sinBuffer,
                position: 0, positions: 8, numHeads: 4, eps: 1e-6,
                cache: evenCache, layer: 0, qOut: qOut)
        }
    }
}
