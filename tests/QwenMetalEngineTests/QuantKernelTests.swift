import XCTest
@testable import QwenMetalEngine
import Metal

/// P3-4 layered-oracle tests (docs/phases/phase-3.md D4/D5, gates
/// pre-committed in DECISIONS.md 2026-08-25 — no new constants):
///
///   Layer 1 (EXACT, bitwise): dequant-tile dump == CPU dequant of the same
///   packed bytes; embedding-gather fp16 store == fp16(exact fp32 dequant).
///   Covers spec edge cases 1-4 GPU-side (nibble order, group boundary,
///   degenerate, extreme-scale/negative-heavy groups).
///
///   Layer 2 (Tier K, Phase 2 constant reused): fused dequant-matvec
///   |Δ| <= max(2⁻⁹·M, 2⁻¹¹) vs the CPU-quant oracle — BLAS.sgemm over the
///   identical dequantized fp32 weights (hard rule 8).
///
/// Hard rule 3: these tests exist BEFORE any optimization of the fused
/// kernels; every optimization iteration must re-pass them.
final class QuantKernelTests: XCTestCase {

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

    // MARK: - Synthetic packed matrices

    /// A q4g64 triplet held in memory, plus the exact fp32 dequant reference
    /// computed with the pinned `Q4G64.dequant` — the same single-rounded
    /// arithmetic `PackedCheckpoint.dequantMatrix` applies, so "GPU == this"
    /// is the layer-1 exactness claim.
    private struct PackedMatrix {
        let outDim: Int
        let inDim: Int
        var words: [UInt32]
        var scales: [Float16]
        var biases: [Float16]

        var dequantReference: [Float] {
            var out = [Float](repeating: 0, count: outDim * inDim)
            for i in 0..<out.count {
                let wordIndex = i / Q4G64.codesPerWord
                let code = Q4G64.code(
                    in: words[wordIndex], lane: i % Q4G64.codesPerWord)
                let group = i / Q4G64.groupSize
                out[i] = Q4G64.dequant(
                    code: code, scale: scales[group], bias: biases[group])
            }
            return out
        }
    }

    /// Packs row-major fp32 values through the pinned recipe (Q4G64.packGroup).
    private func pack(_ values: [Float], outDim: Int, inDim: Int) throws -> PackedMatrix {
        precondition(values.count == outDim * inDim && inDim % Q4G64.groupSize == 0)
        var words: [UInt32] = []
        var scales: [Float16] = []
        var biases: [Float16] = []
        for groupStart in stride(from: 0, to: values.count, by: Q4G64.groupSize) {
            let group = try Q4G64.packGroup(
                values[groupStart..<(groupStart + Q4G64.groupSize)],
                tensor: "test", elementOffset: groupStart)
            words.append(contentsOf: Q4G64.packWords(group.codes))
            scales.append(group.scale)
            biases.append(group.bias)
        }
        return PackedMatrix(
            outDim: outDim, inDim: inDim, words: words, scales: scales, biases: biases)
    }

    /// Hand-built triplet from explicit codes/scales/biases (adversarial
    /// fixtures where the packer would never emit the pattern).
    private func handBuilt(
        codes: [UInt8], scales: [Float16], biases: [Float16], outDim: Int, inDim: Int
    ) -> PackedMatrix {
        precondition(codes.count == outDim * inDim)
        var words: [UInt32] = []
        for groupStart in stride(from: 0, to: codes.count, by: Q4G64.groupSize) {
            words.append(contentsOf: Q4G64.packWords(
                Array(codes[groupStart..<(groupStart + Q4G64.groupSize)])))
        }
        return PackedMatrix(
            outDim: outDim, inDim: inDim, words: words, scales: scales, biases: biases)
    }

    private func randomValues(
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

    /// Runs the tile dump over `matrix` (all offsets 0) and returns the fp32
    /// output.
    private func gpuTile(
        _ matrix: PackedMatrix, context: MetalContext, kernels: QuantKernels
    ) throws -> [Float] {
        let qBuffer = try makeBuffer(context.device, values: matrix.words)
        let scalesBuffer = try makeBuffer(context.device, values: matrix.scales)
        let biasesBuffer = try makeBuffer(context.device, values: matrix.biases)
        let outBuffer = try makeOutputBuffer(
            context.device, count: matrix.outDim * matrix.inDim, elementStride: 4)
        try context.timedDispatch { encoder in
            try kernels.encodeDequantTile(
                into: encoder, q: qBuffer, qByteOffset: 0,
                scales: scalesBuffer, scalesByteOffset: 0,
                biases: biasesBuffer, biasesByteOffset: 0,
                outDim: matrix.outDim, inDim: matrix.inDim, output: outBuffer)
        }
        return readFloats(outBuffer, count: matrix.outDim * matrix.inDim)
    }

    // MARK: - Gate assertions (pre-committed; never loosened)

    /// Layer 1: bitwise fp32 equality, mismatch reported by index and pattern.
    private func assertBitwiseEqual(
        _ got: [Float], _ ref: [Float], _ surface: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.count, ref.count, "\(surface): count mismatch",
                       file: file, line: line)
        for i in 0..<ref.count where got[i].bitPattern != ref[i].bitPattern {
            return XCTFail(
                "\(surface): first bitwise mismatch at [\(i)]: got \(got[i]) "
                    + "(0x\(String(got[i].bitPattern, radix: 16))), ref \(ref[i]) "
                    + "(0x\(String(ref[i].bitPattern, radix: 16)))",
                file: file, line: line)
        }
    }

    /// Layer 2: Phase 2 Tier K reused unchanged.
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

    // MARK: - Layer 1: dequant tile, spec edge case 1 (nibble-order pin)

    func testDequantTileMatchesHandComputedAsymmetricPattern() throws {
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        // One group, asymmetric code pattern: codes[i] = (i*7+3) % 16 — a
        // swapped-nibble read produces a hard value mismatch at lane 0/1.
        let codes = (0..<64).map { UInt8(($0 * 7 + 3) % 16) }
        let matrix = handBuilt(
            codes: codes, scales: [2.0], biases: [-3.0], outDim: 1, inDim: 64)

        let got = try gpuTile(matrix, context: context, kernels: kernels)
        // Hand-computed: q·2 − 3, exact in fp32 for these small integers.
        let handRef = codes.map { Float($0) * 2.0 - 3.0 }
        assertBitwiseEqual(got, handRef, "tile vs hand-computed values")
        assertBitwiseEqual(got, matrix.dequantReference, "tile vs Q4G64.dequant")
    }

    // MARK: - Layer 1: spec edge case 2 (group-boundary indexing)

    func testDequantTileGroupBoundaryElementsUseTheirOwnGroup() throws {
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        // 1×128: two groups with distinct scale/bias; every code 5, so any
        // group mix-up at elements 63/64/65 flips the dequant value.
        let matrix = handBuilt(
            codes: [UInt8](repeating: 5, count: 128),
            scales: [1.0, 100.0], biases: [0.5, -400.0], outDim: 1, inDim: 128)

        let got = try gpuTile(matrix, context: context, kernels: kernels)
        XCTAssertEqual(got[63], 5.5, "element 63 belongs to group 0")
        XCTAssertEqual(got[64], 100.0, "element 64 belongs to group 1")
        XCTAssertEqual(got[65], 100.0, "element 65 belongs to group 1")
        assertBitwiseEqual(got, matrix.dequantReference, "two-group boundary tile")
    }

    /// Same species along the OTHER axis: with multiple rows the group index
    /// must advance by groupsPerRow per row (a row-vs-flat indexing bug
    /// mismatches every row past the first).
    func testDequantTileRowIndexingUsesPerRowGroups() throws {
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        var rng = SplitMix64(seed: 7)
        let (outDim, inDim) = (3, 64)
        let matrix = try pack(
            randomValues(count: outDim * inDim, rng: &rng), outDim: outDim, inDim: inDim)
        XCTAssertNotEqual(matrix.scales[0], matrix.scales[2],
                          "premise: rows carry distinct scales")

        let got = try gpuTile(matrix, context: context, kernels: kernels)
        assertBitwiseEqual(got, matrix.dequantReference, "3-row tile")
    }

    // MARK: - Layer 1: spec edge cases 3+4 (degenerate, extreme, negative-heavy)

    func testDequantTileDegenerateAndExtremeGroupsAreExact() throws {
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        // Six adversarial groups (1×384), same species as the P3-1 CPU
        // fixtures: [0] degenerate scale=0 (dequant == bias exactly),
        // [1] fp16 max-normal scale/bias (products overflow fp16 but not
        // fp32), [2] fp16 min-normal scale, [3] SUBNORMAL scale and bias,
        // [4] negative-heavy (negative bias dominates), [5] subnormal bias
        // with unit scale. Codes sweep 0..15 in each group.
        let codes = (0..<384).map { UInt8($0 % 16) }
        let scales: [Float16] = [
            0, Float16(bitPattern: 0x7BFF), Float16(bitPattern: 0x0400),
            Float16(bitPattern: 0x0001), 0.25, 1.0,
        ]
        let biases: [Float16] = [
            42.0, Float16(bitPattern: 0xFBFF), 1.0,
            Float16(bitPattern: 0x8001), -8.0, Float16(bitPattern: 0x0003),
        ]
        let matrix = handBuilt(
            codes: codes, scales: scales, biases: biases, outDim: 1, inDim: 384)

        let got = try gpuTile(matrix, context: context, kernels: kernels)
        assertBitwiseEqual(got, matrix.dequantReference, "adversarial groups tile")
        // Spot-check the intent, not just the shared arithmetic: degenerate
        // group dequants to bias for every code.
        for i in 0..<64 {
            XCTAssertEqual(got[i], 42.0, "degenerate group must dequant to bias")
        }
    }

    func testDequantTileMatchesReferenceOnRandomPackedMatrix() throws {
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        var rng = SplitMix64(seed: 8)
        // Odd out-dim, multi-group rows, mixed magnitudes (packer-produced
        // scales/biases rather than hand-built ones).
        let (outDim, inDim) = (37, 192)
        let values = randomValues(count: outDim * inDim, range: -4...4, rng: &rng)
        let matrix = try pack(values, outDim: outDim, inDim: inDim)

        let got = try gpuTile(matrix, context: context, kernels: kernels)
        assertBitwiseEqual(got, matrix.dequantReference, "random 37×192 tile")
    }

    func testDequantTileReadsTripletAtNonzeroOffsets() throws {
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        var rng = SplitMix64(seed: 9)
        let (outDim, inDim) = (2, 64)
        let matrix = try pack(
            randomValues(count: outDim * inDim, rng: &rng), outDim: outDim, inDim: inDim)

        // Padding words ahead of each tensor — the whole-checkpoint buffer
        // situation (P3-5): q at a 4-byte-multiple offset, scales/biases at
        // 2-byte-multiple offsets.
        let qPad: [UInt32] = [0xDEAD_BEEF, 0xFFFF_FFFF, 0x0BAD_F00D]
        let groupPad: [Float16] = [.nan]
        let qBuffer = try makeBuffer(context.device, values: qPad + matrix.words)
        let scalesBuffer = try makeBuffer(context.device, values: groupPad + matrix.scales)
        let biasesBuffer = try makeBuffer(context.device, values: groupPad + matrix.biases)
        let outBuffer = try makeOutputBuffer(
            context.device, count: outDim * inDim, elementStride: 4)

        try context.timedDispatch { encoder in
            try kernels.encodeDequantTile(
                into: encoder, q: qBuffer, qByteOffset: qPad.count * 4,
                scales: scalesBuffer, scalesByteOffset: groupPad.count * 2,
                biases: biasesBuffer, biasesByteOffset: groupPad.count * 2,
                outDim: outDim, inDim: inDim, output: outBuffer)
        }
        assertBitwiseEqual(
            readFloats(outBuffer, count: outDim * inDim), matrix.dequantReference,
            "tile at nonzero offsets")
    }

    // MARK: - Layer 1: embedding gather (EXACT fp16 gate)

    func testEmbeddingGatherIsExactIncludingAdversarialRows() throws {
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        var rng = SplitMix64(seed: 10)
        let (vocab, hidden) = (5, 128)
        var values = randomValues(count: vocab * hidden, rng: &rng)
        // Row 2, group 0: constant (degenerate, scale 0). Row 2, group 1:
        // huge magnitudes whose dequant overflows fp16 to ±inf on store —
        // the fp16(exact fp32) rule must hold for them too.
        for i in 0..<64 { values[2 * hidden + i] = -0.375 }
        for i in 64..<128 { values[2 * hidden + i] = (i % 2 == 0) ? 60000.0 : -60000.0 }
        let matrix = try pack(values, outDim: vocab, inDim: hidden)
        let reference = matrix.dequantReference

        let qBuffer = try makeBuffer(context.device, values: matrix.words)
        let scalesBuffer = try makeBuffer(context.device, values: matrix.scales)
        let biasesBuffer = try makeBuffer(context.device, values: matrix.biases)

        for tokenId in [0, 2, vocab - 1] {
            let outBuffer = try makeOutputBuffer(
                context.device, count: hidden, elementStride: 2)
            try context.timedDispatch { encoder in
                try kernels.encodeEmbeddingGather(
                    into: encoder, q: qBuffer, qByteOffset: 0,
                    scales: scalesBuffer, scalesByteOffset: 0,
                    biases: biasesBuffer, biasesByteOffset: 0,
                    vocabSize: vocab, hiddenSize: hidden, tokenId: tokenId,
                    output: outBuffer)
            }
            let ref = reference[(tokenId * hidden)..<((tokenId + 1) * hidden)]
                .map { Float16($0).bitPattern }
            XCTAssertEqual(
                readHalfs(outBuffer, count: hidden).map(\.bitPattern), ref,
                "embedding row \(tokenId) must be fp16(exact fp32) bitwise")
        }

        // Out-of-range ids fail loudly before any dispatch.
        for badId in [-1, vocab] {
            var thrown: Error?
            let outBuffer = try makeOutputBuffer(
                context.device, count: hidden, elementStride: 2)
            try context.timedDispatch { encoder in
                do {
                    try kernels.encodeEmbeddingGather(
                        into: encoder, q: qBuffer, qByteOffset: 0,
                        scales: scalesBuffer, scalesByteOffset: 0,
                        biases: biasesBuffer, biasesByteOffset: 0,
                        vocabSize: vocab, hiddenSize: hidden, tokenId: badId,
                        output: outBuffer)
                } catch { thrown = error }
            }
            XCTAssertEqual(thrown as? QuantKernelError,
                           .tokenIdOutOfRange(id: badId, vocabSize: vocab))
        }
    }

    // MARK: - Layer 2: fused matvec (Tier K vs sgemm over dequant weights)

    private func runMatvecCase(
        outDim: Int, inDim: Int, fp32Output: Bool,
        range: ClosedRange<Float> = -1...1, seed: UInt64
    ) throws {
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        var rng = SplitMix64(seed: seed)
        let matrix = try pack(
            randomValues(count: outDim * inDim, range: range, rng: &rng),
            outDim: outDim, inDim: inDim)
        let x = (0..<inDim).map { _ in Float16(Float.random(in: range, using: &rng)) }

        let qBuffer = try makeBuffer(context.device, values: matrix.words)
        let scalesBuffer = try makeBuffer(context.device, values: matrix.scales)
        let biasesBuffer = try makeBuffer(context.device, values: matrix.biases)
        let xBuffer = try makeBuffer(context.device, values: x)
        let outBuffer = try makeOutputBuffer(
            context.device, count: outDim, elementStride: fp32Output ? 4 : 2)

        try context.timedDispatch { encoder in
            try kernels.encodeMatvec(
                into: encoder, q: qBuffer, qByteOffset: 0,
                scales: scalesBuffer, scalesByteOffset: 0,
                biases: biasesBuffer, biasesByteOffset: 0,
                input: xBuffer, outDim: outDim, inDim: inDim,
                output: outBuffer, fp32Output: fp32Output)
        }

        // CPU-quant oracle: y[1, outDim] = x[1, inDim] · dequant(W)ᵀ through
        // the validated sgemm wrapper (hard rule 8), identical dequant values
        // by the layer-1 exactness argument.
        let ref = try BLAS.sgemm(
            a: x.map(Float.init), b: matrix.dequantReference,
            m: 1, k: inDim, n: outDim, transposeB: true)
        let got = fp32Output
            ? readFloats(outBuffer, count: outDim)
            : readHalfs(outBuffer, count: outDim).map(Float.init)
        assertTierK(got, ref, "matvec_q4 \(outDim)×\(inDim) fp32Out=\(fp32Output)")
    }

    func testMatvecF16MatchesSgemmOverDequantWeights() throws {
        // Odd out-dims; in-dims are structurally multiples of 64 (schema).
        try runMatvecCase(outDim: 67, inDim: 128, fp32Output: false, seed: 20)
        try runMatvecCase(outDim: 301, inDim: 192, fp32Output: false, seed: 21)
    }

    func testMatvecF32OutputMatchesSgemmOverDequantWeights() throws {
        try runMatvecCase(outDim: 67, inDim: 128, fp32Output: true, seed: 22)
        try runMatvecCase(outDim: 301, inDim: 192, fp32Output: true, seed: 23)
    }

    func testMatvecNearZeroSliceIsHeldByAbsoluteFloor() throws {
        // Unit-scale gate would be ~0; the 2⁻¹¹ absolute floor governs.
        try runMatvecCase(outDim: 45, inDim: 64, fp32Output: false,
                          range: -0.001...0.001, seed: 24)
    }

    /// Spec edge case 10 (structural, P2-2 precedent): the tied embedding
    /// triplet, packed once as [vocab, hidden], IS the lm_head [out, in]
    /// matrix — the fp32-store matvec consumes it directly, no transpose,
    /// no second copy. Exercised by feeding the same buffers to both the
    /// gather (embedding role) and the fp32 matvec (lm_head role).
    func testTiedEmbeddingTripletServesLmHeadMatvecDirectly() throws {
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        var rng = SplitMix64(seed: 25)
        let (vocab, hidden) = (33, 128)
        let matrix = try pack(
            randomValues(count: vocab * hidden, rng: &rng), outDim: vocab, inDim: hidden)
        let x = (0..<hidden).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }

        let qBuffer = try makeBuffer(context.device, values: matrix.words)
        let scalesBuffer = try makeBuffer(context.device, values: matrix.scales)
        let biasesBuffer = try makeBuffer(context.device, values: matrix.biases)
        let xBuffer = try makeBuffer(context.device, values: x)
        let gatherOut = try makeOutputBuffer(context.device, count: hidden, elementStride: 2)
        let logitsOut = try makeOutputBuffer(context.device, count: vocab, elementStride: 4)

        try context.timedDispatch { encoder in
            try kernels.encodeEmbeddingGather(
                into: encoder, q: qBuffer, qByteOffset: 0,
                scales: scalesBuffer, scalesByteOffset: 0,
                biases: biasesBuffer, biasesByteOffset: 0,
                vocabSize: vocab, hiddenSize: hidden, tokenId: 4, output: gatherOut)
            try kernels.encodeMatvec(
                into: encoder, q: qBuffer, qByteOffset: 0,
                scales: scalesBuffer, scalesByteOffset: 0,
                biases: biasesBuffer, biasesByteOffset: 0,
                input: xBuffer, outDim: vocab, inDim: hidden,
                output: logitsOut, fp32Output: true)
        }

        let reference = matrix.dequantReference
        XCTAssertEqual(
            readHalfs(gatherOut, count: hidden).map(\.bitPattern),
            reference[(4 * hidden)..<(5 * hidden)].map { Float16($0).bitPattern },
            "gather side of the tied table")
        let ref = try BLAS.sgemm(
            a: x.map(Float.init), b: reference,
            m: 1, k: hidden, n: vocab, transposeB: true)
        assertTierK(readFloats(logitsOut, count: vocab), ref,
                    "lm_head over the tied packed table")
    }

    // MARK: - Host-wrapper rejects (loud, pre-dispatch)

    func testQuantKernelWrappersRejectBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        let q = try makeBuffer(
            context.device, values: [UInt32](repeating: 0, count: 8))
        let groups = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 1))
        let x = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 64))
        let out = try makeOutputBuffer(context.device, count: 64, elementStride: 4)

        func encodeExpectingError(
            _ expected: QuantKernelError,
            _ body: (MTLComputeCommandEncoder) throws -> Void
        ) throws {
            var thrown: Error?
            try context.timedDispatch { encoder in
                do { try body(encoder) } catch { thrown = error }
            }
            guard let error = thrown as? QuantKernelError else {
                return XCTFail("expected QuantKernelError, got \(String(describing: thrown))")
            }
            XCTAssertEqual(error, expected)
        }

        // Spec edge case 6, kernel side: non-multiple-of-64 in-dim.
        try encodeExpectingError(.inDimNotMultipleOfGroup(inDim: 96)) {
            try kernels.encodeMatvec(
                into: $0, q: q, qByteOffset: 0, scales: groups, scalesByteOffset: 0,
                biases: groups, biasesByteOffset: 0, input: x, outDim: 1, inDim: 96,
                output: out)
        }
        try encodeExpectingError(.nonPositiveDimension(name: "outDim", value: 0)) {
            try kernels.encodeMatvec(
                into: $0, q: q, qByteOffset: 0, scales: groups, scalesByteOffset: 0,
                biases: groups, biasesByteOffset: 0, input: x, outDim: 0, inDim: 64,
                output: out)
        }
        // Spec edge case 7, kernel side: misaligned .q offset (u32 loads).
        try encodeExpectingError(.misalignedOffset(buffer: "q", byteOffset: 2, alignment: 4)) {
            try kernels.encodeDequantTile(
                into: $0, q: q, qByteOffset: 2, scales: groups, scalesByteOffset: 0,
                biases: groups, biasesByteOffset: 0, outDim: 1, inDim: 64, output: out)
        }
        try encodeExpectingError(
            .misalignedOffset(buffer: "scales", byteOffset: 1, alignment: 2)
        ) {
            try kernels.encodeDequantTile(
                into: $0, q: q, qByteOffset: 0, scales: groups, scalesByteOffset: 1,
                biases: groups, biasesByteOffset: 0, outDim: 1, inDim: 64, output: out)
        }
        // Short .q buffer: 2 rows × 64 codes need 16 words, buffer holds 8.
        try encodeExpectingError(
            .bufferTooSmall(buffer: "q", requiredBytes: 64, actualBytes: 32)
        ) {
            try kernels.encodeDequantTile(
                into: $0, q: q, qByteOffset: 0, scales: groups, scalesByteOffset: 0,
                biases: groups, biasesByteOffset: 0, outDim: 2, inDim: 64, output: out)
        }
        // Short output: gather of 64 halfs into a 4-byte buffer.
        let tinyOut = try makeOutputBuffer(context.device, count: 1, elementStride: 4)
        try encodeExpectingError(
            .bufferTooSmall(buffer: "output", requiredBytes: 128, actualBytes: 4)
        ) {
            try kernels.encodeEmbeddingGather(
                into: $0, q: q, qByteOffset: 0, scales: groups, scalesByteOffset: 0,
                biases: groups, biasesByteOffset: 0, vocabSize: 1, hiddenSize: 64,
                tokenId: 0, output: tinyOut)
        }
    }

    // MARK: - Real packed artifact spot check (skips when absent)

    /// Ties the synthetic suites to the real thing: one full matrix of the
    /// pinned packed artifact dequants bitwise-identically on GPU vs
    /// `PackedCheckpoint.dequantMatrix` (through the mmap `GPUWeights`
    /// buffer at real file offsets), and the fused matvec holds Tier K
    /// against sgemm over those values at the real [2048, 2048] shape.
    func testRealArtifactTileAndMatvecSpotCheck() throws {
        let packedURL = SharedCheckpoint.modelsDir
            .appendingPathComponent("qwen3-1.7b-70d244cc-q4g64.safetensors")
        guard FileManager.default.fileExists(atPath: packedURL.path) else {
            throw XCTSkip(
                "packed artifact not present at \(packedURL.path) "
                + "(local-only; produce it with `qwen-metal-cli pack`)")
        }
        let context = try makeContextOrSkip()
        let kernels = try QuantKernels(context: context)
        let packed = try PackedCheckpoint(
            path: packedURL.path, expectedRevision: SharedCheckpoint.pinnedRevision)
        let weights = try GPUWeights(
            file: packed.file, context: context, residency: .mmap)

        let name = "model.layers.0.self_attn.q_proj.weight"
        let dims = try packed.dims(for: name)
        XCTAssertEqual(dims.outDim, 2048, "pinned q_proj shape")
        XCTAssertEqual(dims.inDim, 2048, "pinned q_proj shape")
        let qOffset = try weights.byteOffset(for: name + Q4G64.qSuffix)
        let scalesOffset = try weights.byteOffset(for: name + Q4G64.scalesSuffix)
        let biasesOffset = try weights.byteOffset(for: name + Q4G64.biasesSuffix)

        let tileOut = try makeOutputBuffer(
            context.device, count: dims.outDim * dims.inDim, elementStride: 4)
        var rng = SplitMix64(seed: 30)
        let x = (0..<dims.inDim).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
        let xBuffer = try makeBuffer(context.device, values: x)
        let matvecOut = try makeOutputBuffer(
            context.device, count: dims.outDim, elementStride: 2)

        try context.timedDispatch { encoder in
            try kernels.encodeDequantTile(
                into: encoder, q: weights.buffer, qByteOffset: qOffset,
                scales: weights.buffer, scalesByteOffset: scalesOffset,
                biases: weights.buffer, biasesByteOffset: biasesOffset,
                outDim: dims.outDim, inDim: dims.inDim, output: tileOut)
            try kernels.encodeMatvec(
                into: encoder, q: weights.buffer, qByteOffset: qOffset,
                scales: weights.buffer, scalesByteOffset: scalesOffset,
                biases: weights.buffer, biasesByteOffset: biasesOffset,
                input: xBuffer, outDim: dims.outDim, inDim: dims.inDim,
                output: matvecOut)
        }

        let cpuDequant = try packed.dequantMatrix(name)
        let gpuDequant = readFloats(tileOut, count: dims.outDim * dims.inDim)
        XCTAssertTrue(
            gpuDequant.withUnsafeBytes { gpu in
                cpuDequant.withUnsafeBytes { cpu in
                    memcmp(gpu.baseAddress!, cpu.baseAddress!, gpu.count) == 0
                }
            },
            "real-artifact q_proj tile must be bitwise identical to "
                + "PackedCheckpoint.dequantMatrix")

        let ref = try BLAS.sgemm(
            a: x.map(Float.init), b: cpuDequant,
            m: 1, k: dims.inDim, n: dims.outDim, transposeB: true)
        assertTierK(
            readHalfs(matvecOut, count: dims.outDim).map(Float.init), ref,
            "real-artifact q_proj matvec")
    }
}
