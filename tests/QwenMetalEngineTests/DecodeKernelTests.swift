import XCTest
@testable import QwenMetalEngine
import Metal

/// P2-2 Tier-K tests (docs/phases/phase-2.md D4): every naive decode kernel
/// diffed against the CPU reference on synthetic unit-scale inputs BEFORE any
/// optimization exists (hard rule 3).
///
/// Gates are the pre-committed Phase 2 tolerances (DECISIONS.md 2026-08-23):
///   Tier K:  |Δ| <= max(2⁻⁹·M, 2⁻¹¹), M = max|ref| over the compared slice.
///   Exact:   embedding-lookup — GPU fp16 == fp16(ref fp32) bitwise.
/// Matmul-shaped reference work routes through BLAS.sgemm (hard rule 8).
final class DecodeKernelTests: XCTestCase {

    // MARK: - Deterministic RNG (SgemmTests pattern; system RNGs are unseedable)

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

    /// Random bf16 bit patterns: fp32 draws truncated to their top 16 bits —
    /// exactly the checkpoint's storage; CPU upcast of these bits is exact.
    private func randomBF16(
        count: Int, range: ClosedRange<Float> = -1...1, rng: inout SplitMix64
    ) -> [UInt16] {
        (0..<count).map { _ in
            UInt16(truncatingIfNeeded: Float.random(in: range, using: &rng).bitPattern >> 16)
        }
    }

    private func upcast(_ bf16: [UInt16]) -> [Float] {
        bf16.map { Float(bitPattern: UInt32($0) << 16) }
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

    private func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        [Float](UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: count),
            count: count))
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

    // MARK: - matvec (Tier K; reference via BLAS.sgemm — hard rule 8)

    private func runMatvecCase(
        outDim: Int, inDim: Int, fp32Output: Bool,
        range: ClosedRange<Float> = -1...1, seed: UInt64
    ) throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        var rng = SplitMix64(seed: seed)
        let weightBits = randomBF16(count: outDim * inDim, range: range, rng: &rng)
        let x = randomHalfs(count: inDim, range: range, rng: &rng)

        let weightsBuffer = try makeBuffer(context.device, values: weightBits)
        let xBuffer = try makeBuffer(context.device, values: x)
        let outBuffer = try makeOutputBuffer(
            context.device, count: outDim, elementStride: fp32Output ? 4 : 2)

        try context.timedDispatch { encoder in
            try kernels.encodeMatvec(
                into: encoder, weights: weightsBuffer, weightByteOffset: 0,
                input: xBuffer, outDim: outDim, inDim: inDim,
                output: outBuffer, fp32Output: fp32Output)
        }

        // Reference: y[1, outDim] = x[1, inDim] · W[outDim, inDim]ᵀ in fp32.
        let ref = try BLAS.sgemm(
            a: x.map(Float.init), b: upcast(weightBits),
            m: 1, k: inDim, n: outDim, transposeB: true)
        let got = fp32Output
            ? readFloats(outBuffer, count: outDim)
            : readHalfs(outBuffer, count: outDim).map(Float.init)
        assertTierK(got, ref, "matvec \(outDim)×\(inDim) fp32Out=\(fp32Output)")
    }

    func testMatvecF16MatchesSgemmOnOddShapes() throws {
        try runMatvecCase(outDim: 67, inDim: 129, fp32Output: false, seed: 1)
        try runMatvecCase(outDim: 301, inDim: 257, fp32Output: false, seed: 2)
    }

    func testMatvecF32OutputMatchesSgemmOnOddShapes() throws {
        try runMatvecCase(outDim: 67, inDim: 129, fp32Output: true, seed: 3)
        try runMatvecCase(outDim: 301, inDim: 257, fp32Output: true, seed: 4)
    }

    func testMatvecNearZeroSliceIsHeldByAbsoluteFloor() throws {
        // Unit-scale gate would be ~0; the 2⁻¹¹ absolute floor governs.
        try runMatvecCase(outDim: 45, inDim: 67, fp32Output: false,
                          range: -0.001...0.001, seed: 5)
    }

    func testMatvecReadsWeightsAtNonzeroEvenByteOffset() throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        var rng = SplitMix64(seed: 6)
        let (outDim, inDim) = (19, 23)
        let padding: [UInt16] = [0xDEAD, 0xBEEF, 0x7FC1] // 6 bytes of junk
        let weightBits = randomBF16(count: outDim * inDim, rng: &rng)
        let x = randomHalfs(count: inDim, rng: &rng)

        let weightsBuffer = try makeBuffer(context.device, values: padding + weightBits)
        let xBuffer = try makeBuffer(context.device, values: x)
        let outBuffer = try makeOutputBuffer(context.device, count: outDim, elementStride: 2)

        try context.timedDispatch { encoder in
            try kernels.encodeMatvec(
                into: encoder, weights: weightsBuffer,
                weightByteOffset: padding.count * 2,
                input: xBuffer, outDim: outDim, inDim: inDim, output: outBuffer)
        }
        let ref = try BLAS.sgemm(
            a: x.map(Float.init), b: upcast(weightBits),
            m: 1, k: inDim, n: outDim, transposeB: true)
        assertTierK(readHalfs(outBuffer, count: outDim).map(Float.init), ref,
                    "matvec at byte offset \(padding.count * 2)")
    }

    func testMatvecRejectsBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let weights = try makeBuffer(context.device, values: [UInt16](repeating: 0, count: 8))
        let x = try makeBuffer(context.device, values: [Float16](repeating: 0, count: 4))
        let out = try makeOutputBuffer(context.device, count: 2, elementStride: 2)

        func encodeExpectingError(
            _ expected: DecodeKernelError, _ body: (MTLComputeCommandEncoder) throws -> Void
        ) throws {
            var thrown: Error?
            try context.timedDispatch { encoder in
                do { try body(encoder) } catch { thrown = error }
            }
            guard let error = thrown as? DecodeKernelError else {
                return XCTFail("expected DecodeKernelError, got \(String(describing: thrown))")
            }
            XCTAssertEqual(error, expected)
        }

        try encodeExpectingError(.nonPositiveDimension(name: "outDim", value: 0)) {
            try kernels.encodeMatvec(into: $0, weights: weights, weightByteOffset: 0,
                                     input: x, outDim: 0, inDim: 4, output: out)
        }
        try encodeExpectingError(.misalignedWeightOffset(byteOffset: 3)) {
            try kernels.encodeMatvec(into: $0, weights: weights, weightByteOffset: 3,
                                     input: x, outDim: 2, inDim: 4, output: out)
        }
        try encodeExpectingError(
            .bufferTooSmall(buffer: "weights", requiredBytes: 20, actualBytes: 16)
        ) {
            try kernels.encodeMatvec(into: $0, weights: weights, weightByteOffset: 0,
                                     input: x, outDim: 2, inDim: 5, output: out)
        }
        try encodeExpectingError(
            .bufferTooSmall(buffer: "output", requiredBytes: 16, actualBytes: 4)
        ) {
            try kernels.encodeMatvec(into: $0, weights: weights, weightByteOffset: 0,
                                     input: x, outDim: 4, inDim: 2, output: out,
                                     fp32Output: true)
        }
    }

    // MARK: - rmsnorm (Tier K vs the CPU reference module)

    private func runRMSNormCase(rows: Int, dim: Int, seed: UInt64) throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        var rng = SplitMix64(seed: seed)
        let x = randomHalfs(count: rows * dim, rng: &rng)
        let weightBits = randomBF16(count: dim, rng: &rng)
        let eps: Float = 1e-6

        let xBuffer = try makeBuffer(context.device, values: x)
        let weightBuffer = try makeBuffer(context.device, values: weightBits)
        let outBuffer = try makeOutputBuffer(context.device, count: rows * dim, elementStride: 2)

        try context.timedDispatch { encoder in
            try kernels.encodeRMSNorm(
                into: encoder, input: xBuffer, weight: weightBuffer,
                weightByteOffset: 0, rows: rows, dim: dim, eps: eps,
                output: outBuffer)
        }

        let cpuNorm = RMSNorm(weight: upcast(weightBits), eps: eps)
        let ref = try cpuNorm(x.map(Float.init))
        assertTierK(readHalfs(outBuffer, count: rows * dim).map(Float.init), ref,
                    "rmsnorm rows=\(rows) dim=\(dim)")
    }

    func testRMSNormSingleRowMatchesCPUOnOddDim() throws {
        try runRMSNormCase(rows: 1, dim: 67, seed: 10)
    }

    func testRMSNormPerHeadRowsMatchCPU() throws {
        // The QK-norm shape: several consecutive headDim rows, one weight.
        try runRMSNormCase(rows: 5, dim: 32, seed: 11)
    }

    func testRMSNormRejectsBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let x = try makeBuffer(context.device, values: [Float16](repeating: 0, count: 4))
        let weight = try makeBuffer(context.device, values: [UInt16](repeating: 0, count: 4))
        let out = try makeOutputBuffer(context.device, count: 4, elementStride: 2)

        var thrown: Error?
        try context.timedDispatch { encoder in
            do {
                try kernels.encodeRMSNorm(
                    into: encoder, input: x, weight: weight, weightByteOffset: 0,
                    rows: 1, dim: 0, eps: 1e-6, output: out)
            } catch { thrown = error }
        }
        XCTAssertEqual(thrown as? DecodeKernelError,
                       .nonPositiveDimension(name: "dim", value: 0))

        thrown = nil
        try context.timedDispatch { encoder in
            do {
                try kernels.encodeRMSNorm(
                    into: encoder, input: x, weight: weight, weightByteOffset: 0,
                    rows: 1, dim: 8, eps: 1e-6, output: out)
            } catch { thrown = error }
        }
        XCTAssertEqual(thrown as? DecodeKernelError,
                       .bufferTooSmall(buffer: "weight", requiredBytes: 16, actualBytes: 8))
    }

    // MARK: - rope (Tier K; position p oracle = CPU full recompute)

    func testRoPEAtPositionZeroMatchesCPU() throws {
        try runRoPECase(position: 0, seqLen: 1, seed: 20)
    }

    func testRoPEAtLaterPositionMatchesCPUFullRecompute() throws {
        // The targeted unit form of spec edge case 3: a decode step at p > 0
        // must use angle(p). CPU oracle recomputes the full sequence and the
        // test compares row p only.
        try runRoPECase(position: 9, seqLen: 10, seed: 21)
    }

    private func runRoPECase(position: Int, seqLen: Int, seed: UInt64) throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let (heads, headDim, positions) = (4, 16, 16)
        var rng = SplitMix64(seed: seed)
        // CPU input: full [seqLen, heads·headDim]; the GPU sees only row p.
        let fullInput = randomHalfs(count: seqLen * heads * headDim, rng: &rng)
        let tokenRow = Array(fullInput[((seqLen - 1) * heads * headDim)...])
        XCTAssertEqual(seqLen - 1, position, "test premise: token row is row p")

        let rope = try RoPE(headDim: headDim, theta: 10_000, positions: positions)
        let vectorBuffer = try makeBuffer(context.device, values: tokenRow)
        let cosBuffer = try makeBuffer(context.device, values: rope.cosValues)
        let sinBuffer = try makeBuffer(context.device, values: rope.sinValues)

        try context.timedDispatch { encoder in
            try kernels.encodeRoPE(
                into: encoder, vector: vectorBuffer, cosTable: cosBuffer,
                sinTable: sinBuffer, position: position, positions: positions,
                heads: heads, headDim: headDim)
        }

        var cpu = fullInput.map(Float.init)
        try rope.apply(to: &cpu, seqLen: seqLen, heads: heads)
        let ref = Array(cpu[(position * heads * headDim)...])
        assertTierK(
            readHalfs(vectorBuffer, count: heads * headDim).map(Float.init),
            ref, "rope position \(position)")
    }

    func testRoPERejectsBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let rope = try RoPE(headDim: 8, theta: 10_000, positions: 4)
        let vector = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 2 * 8))
        let cosBuffer = try makeBuffer(context.device, values: rope.cosValues)
        let sinBuffer = try makeBuffer(context.device, values: rope.sinValues)

        func encodeExpectingError(
            _ expected: DecodeKernelError, position: Int = 0, headDim: Int = 8
        ) throws {
            var thrown: Error?
            try context.timedDispatch { encoder in
                do {
                    try kernels.encodeRoPE(
                        into: encoder, vector: vector, cosTable: cosBuffer,
                        sinTable: sinBuffer, position: position, positions: 4,
                        heads: 2, headDim: headDim)
                } catch { thrown = error }
            }
            XCTAssertEqual(thrown as? DecodeKernelError, expected)
        }

        try encodeExpectingError(.oddHeadDim(headDim: 7), headDim: 7)
        try encodeExpectingError(
            .positionOutOfRange(position: 4, positions: 4), position: 4)
        try encodeExpectingError(
            .positionOutOfRange(position: -1, positions: 4), position: -1)
    }

    // MARK: - swiglu (Tier K vs the CPU MLP's elementwise formula)

    func testSwiGLUMatchesCPUElementwise() throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let count = 1001
        var rng = SplitMix64(seed: 30)
        // ±3 covers silu's negative tail and near-linear positive side.
        let gate = randomHalfs(count: count, range: -3...3, rng: &rng)
        let up = randomHalfs(count: count, range: -3...3, rng: &rng)

        let gateBuffer = try makeBuffer(context.device, values: gate)
        let upBuffer = try makeBuffer(context.device, values: up)
        let outBuffer = try makeOutputBuffer(context.device, count: count, elementStride: 2)

        try context.timedDispatch { encoder in
            try kernels.encodeSwiGLU(
                into: encoder, gate: gateBuffer, up: upBuffer, count: count,
                output: outBuffer)
        }

        // CPU MLP's exact elementwise form: (g / (1 + e^-g)) · u in fp32.
        var ref = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let g = Float(gate[i])
            ref[i] = (g / (1 + expf(-g))) * Float(up[i])
        }
        assertTierK(readHalfs(outBuffer, count: count).map(Float.init), ref, "swiglu")
    }

    // MARK: - residual add (Tier K)

    func testResidualAddMatchesCPU() throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let count = 517
        var rng = SplitMix64(seed: 40)
        let a = randomHalfs(count: count, rng: &rng)
        let b = randomHalfs(count: count, rng: &rng)

        let aBuffer = try makeBuffer(context.device, values: a)
        let bBuffer = try makeBuffer(context.device, values: b)
        let outBuffer = try makeOutputBuffer(context.device, count: count, elementStride: 2)

        try context.timedDispatch { encoder in
            try kernels.encodeResidualAdd(
                into: encoder, a: aBuffer, b: bBuffer, count: count, output: outBuffer)
        }
        let ref = (0..<count).map { Float(a[$0]) + Float(b[$0]) }
        assertTierK(readHalfs(outBuffer, count: count).map(Float.init), ref, "residual add")
    }

    func testElementwiseKernelsRejectBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let small = try makeBuffer(context.device, values: [Float16](repeating: 0, count: 2))
        let out = try makeOutputBuffer(context.device, count: 8, elementStride: 2)

        var thrown: Error?
        try context.timedDispatch { encoder in
            do {
                try kernels.encodeSwiGLU(into: encoder, gate: small, up: small,
                                         count: 0, output: out)
            } catch { thrown = error }
        }
        XCTAssertEqual(thrown as? DecodeKernelError,
                       .nonPositiveDimension(name: "count", value: 0))

        thrown = nil
        try context.timedDispatch { encoder in
            do {
                try kernels.encodeResidualAdd(into: encoder, a: small, b: small,
                                              count: 8, output: out)
            } catch { thrown = error }
        }
        XCTAssertEqual(thrown as? DecodeKernelError,
                       .bufferTooSmall(buffer: "a", requiredBytes: 16, actualBytes: 4))
    }

    // MARK: - embedding lookup (pre-committed EXACT gate — bitwise)

    func testEmbeddingLookupIsExactIncludingBoundaryPatterns() throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let (vocab, hidden) = (7, 9)
        var rng = SplitMix64(seed: 50)
        var tableBits = randomBF16(count: vocab * hidden, rng: &rng)
        // Row 3 = fp16-conversion boundary patterns: ±0, bf16 subnormal
        // (underflows fp16 to 0), values overflowing fp16 to ±inf, the fp16
        // min-normal and a subnormal-result value, and +inf. The row gather +
        // upcast + fp16 store must be exact for ALL of them.
        let boundaryRow: [UInt16] = [
            0x3F80, 0x8000, 0x0000, 0x0001, 0x4780, 0xC780, 0x3880, 0x3800, 0x7F80,
        ]
        XCTAssertEqual(boundaryRow.count, hidden, "test premise: one full row")
        tableBits.replaceSubrange((3 * hidden)..<(4 * hidden), with: boundaryRow)

        let tableBuffer = try makeBuffer(context.device, values: tableBits)
        let refTable = upcast(tableBits)

        for tokenId in [0, 3, vocab - 1] {
            let outBuffer = try makeOutputBuffer(context.device, count: hidden, elementStride: 2)
            try context.timedDispatch { encoder in
                try kernels.encodeEmbeddingLookup(
                    into: encoder, table: tableBuffer, tableByteOffset: 0,
                    vocabSize: vocab, hiddenSize: hidden, tokenId: tokenId,
                    output: outBuffer)
            }
            let ref = refTable[(tokenId * hidden)..<((tokenId + 1) * hidden)]
                .map { Float16($0).bitPattern }
            XCTAssertEqual(
                readHalfs(outBuffer, count: hidden).map(\.bitPattern), ref,
                "embedding row \(tokenId) must be bitwise exact")
        }
    }

    func testEmbeddingLookupAtNonzeroTableOffsetAndIdBounds() throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let (vocab, hidden) = (3, 4)
        var rng = SplitMix64(seed: 51)
        let padding: [UInt16] = [0xFFFF, 0x1234]
        let tableBits = randomBF16(count: vocab * hidden, rng: &rng)
        let tableBuffer = try makeBuffer(context.device, values: padding + tableBits)
        let outBuffer = try makeOutputBuffer(context.device, count: hidden, elementStride: 2)

        try context.timedDispatch { encoder in
            try kernels.encodeEmbeddingLookup(
                into: encoder, table: tableBuffer, tableByteOffset: padding.count * 2,
                vocabSize: vocab, hiddenSize: hidden, tokenId: 1, output: outBuffer)
        }
        let ref = upcast(tableBits)[(1 * hidden)..<(2 * hidden)]
            .map { Float16($0).bitPattern }
        XCTAssertEqual(readHalfs(outBuffer, count: hidden).map(\.bitPattern), ref)

        // Out-of-range ids fail loudly before any dispatch.
        for badId in [-1, vocab] {
            var thrown: Error?
            try context.timedDispatch { encoder in
                do {
                    try kernels.encodeEmbeddingLookup(
                        into: encoder, table: tableBuffer,
                        tableByteOffset: padding.count * 2, vocabSize: vocab,
                        hiddenSize: hidden, tokenId: badId, output: outBuffer)
                } catch { thrown = error }
            }
            XCTAssertEqual(thrown as? DecodeKernelError,
                           .tokenIdOutOfRange(id: badId, vocabSize: vocab))
        }
    }

    // MARK: - composition sanity (the P2-4 shape: several kernels, one buffer)

    func testKernelsComposeWithinOneCommandBuffer() throws {
        // swiglu -> residual-add chained in a single command buffer, the way
        // P2-4 will encode a whole token (spec D5). Reference is the fp32
        // chain with the intermediate rounded to fp16, exactly like the GPU's
        // inter-kernel activation surface (spec D2).
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let count = 129
        var rng = SplitMix64(seed: 60)
        let gate = randomHalfs(count: count, range: -2...2, rng: &rng)
        let up = randomHalfs(count: count, range: -2...2, rng: &rng)
        let residual = randomHalfs(count: count, rng: &rng)

        let gateBuffer = try makeBuffer(context.device, values: gate)
        let upBuffer = try makeBuffer(context.device, values: up)
        let residualBuffer = try makeBuffer(context.device, values: residual)
        let midBuffer = try makeOutputBuffer(context.device, count: count, elementStride: 2)
        let outBuffer = try makeOutputBuffer(context.device, count: count, elementStride: 2)

        try context.timedDispatch { encoder in
            try kernels.encodeSwiGLU(
                into: encoder, gate: gateBuffer, up: upBuffer, count: count,
                output: midBuffer)
            try kernels.encodeResidualAdd(
                into: encoder, a: midBuffer, b: residualBuffer, count: count,
                output: outBuffer)
        }

        var ref = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let g = Float(gate[i])
            let mid = Float16((g / (1 + expf(-g))) * Float(up[i])) // fp16 surface
            ref[i] = Float(mid) + Float(residual[i])
        }
        assertTierK(readHalfs(outBuffer, count: count).map(Float.init), ref,
                    "swiglu → residual chain")
    }
}
