import XCTest
@testable import QwenMetalEngine
import Metal

/// P5-2 correctness suite for the tiled q4g64 dequant-GEMM kernel
/// (docs/phases/phase-5.md edge tests 1–2; gates pre-committed in
/// DECISIONS.md 2026-09-14 — NO new constants):
///
///   Edge test 1: odd synthetic shapes + real weight shapes (real-artifact
///   test below, skip-if-absent), ragged M (M % tile ≠ 0), M = 1
///   degenerate — Tier K max(2⁻⁹·M, 2⁻¹¹) vs the CPU-quant oracle:
///   BLAS.sgemm over the identical dequantized fp32 weights (hard rule 8).
///
///   Edge test 2: packed-layout adversarial reads through the GEMM path —
///   group boundaries crossing K-tiles, nibble order, the P3-1 fixture
///   species (degenerate/extreme/subnormal/negative-heavy groups). One-hot
///   activation rows make these EXACT (a single product term survives, so
///   the fp16 store must be fp16(exact fp32 dequant) bitwise — stronger
///   than Tier K, same species as the P3-4 layer-1 gates).
///
/// Hard rule 3: this suite exists at the kernel's first landing, BEFORE any
/// optimization iteration; every iteration must re-pass it.
final class QuantGemmKernelTests: XCTestCase {

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

    // MARK: - Synthetic packed matrices (QuantKernelTests pattern)

    private struct PackedMatrix {
        let outDim: Int
        let inDim: Int
        var words: [UInt32]
        var scales: [Float16]
        var biases: [Float16]

        /// Row-major [outDim, inDim] fp32 dequant via the pinned arithmetic —
        /// the value set `PackedCheckpoint.dequantMatrix` produces.
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

    /// Runs the tiled GEMM over an in-memory triplet (all offsets 0) and
    /// returns the fp16 output upcast to fp32.
    private func gpuGemm(
        _ matrix: PackedMatrix, a: [Float16], m: Int,
        context: MetalContext, kernel: QuantGemmKernel
    ) throws -> [Float] {
        precondition(a.count == m * matrix.inDim)
        let qBuffer = try makeBuffer(context.device, values: matrix.words)
        let scalesBuffer = try makeBuffer(context.device, values: matrix.scales)
        let biasesBuffer = try makeBuffer(context.device, values: matrix.biases)
        let aBuffer = try makeBuffer(context.device, values: a)
        let outBuffer = try makeOutputBuffer(
            context.device, count: m * matrix.outDim, elementStride: 2)
        try context.timedDispatch { encoder in
            try kernel.encodeGemm(
                into: encoder, q: qBuffer, qByteOffset: 0,
                scales: scalesBuffer, scalesByteOffset: 0,
                biases: biasesBuffer, biasesByteOffset: 0,
                input: aBuffer, batchM: m,
                outDim: matrix.outDim, inDim: matrix.inDim, output: outBuffer)
        }
        return readHalfs(outBuffer, count: m * matrix.outDim).map(Float.init)
    }

    // MARK: - Gate assertion (Tier K, reused verbatim — never loosened)

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

    /// One Tier K case: random packed W [n, k], random fp16 A [m, k], GPU
    /// GEMM vs sgemm over the dequantized weights.
    private func runGemmCase(
        m: Int, k: Int, n: Int,
        range: ClosedRange<Float> = -1...1, seed: UInt64,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let context = try makeContextOrSkip()
        let kernel = try QuantGemmKernel(context: context)
        var rng = SplitMix64(seed: seed)
        let matrix = try pack(
            randomValues(count: n * k, range: range, rng: &rng),
            outDim: n, inDim: k)
        let a = (0..<(m * k)).map { _ in
            Float16(Float.random(in: range, using: &rng))
        }

        let got = try gpuGemm(matrix, a: a, m: m, context: context, kernel: kernel)
        let ref = try BLAS.sgemm(
            a: a.map(Float.init), b: matrix.dequantReference,
            m: m, k: k, n: n, transposeB: true)
        assertTierK(got, ref, "gemm_q4 \(m)×\(k)×\(n)", file: file, line: line)
    }

    // MARK: - Edge test 1: odd shapes, ragged M, M = 1

    func testGemmMatchesSgemmOnOddShapes() throws {
        // Odd N, K spanning multiple 32-wide tiles and groups; M covers a
        // full tile (32), a ragged multi-simdgroup-row count (11), and a
        // multi-tile M (37).
        try runGemmCase(m: 32, k: 128, n: 67, seed: 40)
        try runGemmCase(m: 11, k: 192, n: 301, seed: 41)
        try runGemmCase(m: 37, k: 256, n: 45, seed: 42)
    }

    func testGemmRaggedAndDegenerateM() throws {
        // M = 1 (the degenerate matvec-shaped call), M = 8 (exactly one
        // simdgroup row), M = 33 (one full tile + 1).
        try runGemmCase(m: 1, k: 128, n: 67, seed: 43)
        try runGemmCase(m: 8, k: 64, n: 96, seed: 44)
        try runGemmCase(m: 33, k: 64, n: 33, seed: 45)
    }

    func testGemmNearZeroSliceIsHeldByAbsoluteFloor() throws {
        // Unit-scale gate would be ~0; the 2⁻¹¹ absolute floor governs.
        try runGemmCase(m: 9, k: 64, n: 45, range: -0.001...0.001, seed: 46)
    }

    // MARK: - Edge test 2: adversarial packed-layout reads, EXACT via one-hots

    /// A rows are one-hot at chosen k positions: out[row, n] = the single
    /// surviving product 1.0 · dequant(W[n, kPos]) accumulated in fp32 —
    /// exact — so the fp16 store must equal fp16(dequant value) bitwise.
    private func assertOneHotRowsExact(
        _ matrix: PackedMatrix, kPositions: [Int],
        context: MetalContext, kernel: QuantGemmKernel,
        _ surface: String, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let m = kPositions.count
        var a = [Float16](repeating: 0, count: m * matrix.inDim)
        for (row, k) in kPositions.enumerated() {
            a[row * matrix.inDim + k] = 1.0
        }
        let got = try gpuGemm(matrix, a: a, m: m, context: context, kernel: kernel)
        let reference = matrix.dequantReference
        for (row, k) in kPositions.enumerated() {
            for n in 0..<matrix.outDim {
                let expected = Float16(reference[n * matrix.inDim + k])
                let actual = Float16(got[row * matrix.outDim + n])
                XCTAssertEqual(
                    actual.bitPattern, expected.bitPattern,
                    "\(surface): one-hot k=\(k), out row \(row) col \(n): "
                        + "got \(actual), expected fp16(dequant) \(expected)",
                    file: file, line: line)
            }
        }
    }

    func testNibbleOrderPinThroughGemmPath() throws {
        let context = try makeContextOrSkip()
        let kernel = try QuantGemmKernel(context: context)
        // Asymmetric code pattern: a swapped-nibble read produces a hard
        // value mismatch at adjacent lanes (P3-4 fixture species).
        let codes = (0..<64).map { UInt8(($0 * 7 + 3) % 16) }
        let matrix = handBuilt(
            codes: codes, scales: [2.0], biases: [-3.0], outDim: 1, inDim: 64)
        try assertOneHotRowsExact(
            matrix, kPositions: [0, 1, 2, 7, 8, 62, 63],
            context: context, kernel: kernel, "nibble-order one-hots")
    }

    func testGroupBoundariesCrossingKTiles() throws {
        let context = try makeContextOrSkip()
        let kernel = try QuantGemmKernel(context: context)
        // K = 256: groups of 64 at cols [0,64), [64,128), [128,192),
        // [192,256) — but the kernel steps K in 32-wide tiles, so tiles
        // {0,1} share group 0, {2,3} share group 1, … Distinct scale/bias
        // per group + constant code 5 turns any tile→group indexing bug
        // into a hard value flip at the boundaries (P2-3 pattern).
        // Two rows with different group tables catch row-major indexing.
        let scales: [Float16] = [1.0, 100.0, 0.25, 8.0, 2.0, 50.0, 0.5, 16.0]
        let biases: [Float16] = [0.5, -400.0, 3.0, -1.0, 1.5, -200.0, 6.0, -2.0]
        let matrix = handBuilt(
            codes: [UInt8](repeating: 5, count: 2 * 256),
            scales: scales, biases: biases, outDim: 2, inDim: 256)
        try assertOneHotRowsExact(
            matrix, kPositions: [0, 31, 32, 63, 64, 65, 127, 128, 191, 192, 255],
            context: context, kernel: kernel, "group-boundary one-hots")

        // Belt-and-braces value spot check at the first boundary:
        // k=63 hits group 0 of each row, k=64 hits group 1.
        let a: [Float16] = {
            var v = [Float16](repeating: 0, count: 2 * 256)
            v[63] = 1.0        // row 0 one-hot at 63
            v[256 + 64] = 1.0  // row 1 one-hot at 64
            return v
        }()
        let got = try gpuGemm(matrix, a: a, m: 2, context: context, kernel: kernel)
        XCTAssertEqual(got[0], 5.5, "one-hot 63 → W row 0 group 0: 5·1+0.5")
        XCTAssertEqual(got[2], 100.0, "one-hot 64 → W row 0 group 1: 5·100−400")
    }

    func testAdversarialGroupSpeciesThroughGemmPath() throws {
        let context = try makeContextOrSkip()
        let kernel = try QuantGemmKernel(context: context)
        // The P3-1/P3-4 fixture species as six groups of one row (1×384):
        // [0] degenerate scale=0 (dequant == bias), [1] fp16 max-normal
        // scale/bias, [2] fp16 min-normal scale, [3] SUBNORMAL scale and
        // bias, [4] negative-heavy, [5] subnormal bias with unit scale.
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
        // One hit inside every group, including group-boundary neighbors.
        try assertOneHotRowsExact(
            matrix, kPositions: [0, 63, 64, 127, 128, 192, 256, 320, 383],
            context: context, kernel: kernel, "adversarial-group one-hots")
    }

    func testGemmReadsTripletAtNonzeroOffsets() throws {
        let context = try makeContextOrSkip()
        let kernel = try QuantGemmKernel(context: context)
        var rng = SplitMix64(seed: 47)
        let (m, k, n) = (5, 64, 34)
        let matrix = try pack(
            randomValues(count: n * k, rng: &rng), outDim: n, inDim: k)
        let a = (0..<(m * k)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }

        // Padding ahead of each tensor — the whole-checkpoint buffer
        // situation (P3-5): q at a 4-byte-multiple offset, scales/biases at
        // 2-byte-multiple offsets. NaN pad poisons any off-by-one read.
        let qPad: [UInt32] = [0xDEAD_BEEF, 0xFFFF_FFFF, 0x0BAD_F00D]
        let groupPad: [Float16] = [.nan]
        let qBuffer = try makeBuffer(context.device, values: qPad + matrix.words)
        let scalesBuffer = try makeBuffer(context.device, values: groupPad + matrix.scales)
        let biasesBuffer = try makeBuffer(context.device, values: groupPad + matrix.biases)
        let aBuffer = try makeBuffer(context.device, values: a)
        let outBuffer = try makeOutputBuffer(context.device, count: m * n, elementStride: 2)

        try context.timedDispatch { encoder in
            try kernel.encodeGemm(
                into: encoder, q: qBuffer, qByteOffset: qPad.count * 4,
                scales: scalesBuffer, scalesByteOffset: groupPad.count * 2,
                biases: biasesBuffer, biasesByteOffset: groupPad.count * 2,
                input: aBuffer, batchM: m, outDim: n, inDim: k, output: outBuffer)
        }
        let ref = try BLAS.sgemm(
            a: a.map(Float.init), b: matrix.dequantReference,
            m: m, k: k, n: n, transposeB: true)
        assertTierK(
            readHalfs(outBuffer, count: m * n).map(Float.init), ref,
            "gemm at nonzero offsets")
    }

    // MARK: - Ragged-M output isolation

    /// Rows past M must never be written: the output buffer carries poison
    /// rows beyond M×N and they must come back untouched (the zero-padded
    /// A rows are compute-only garbage-freeing, not license to store).
    func testGemmWritesExactlyMRows() throws {
        let context = try makeContextOrSkip()
        let kernel = try QuantGemmKernel(context: context)
        var rng = SplitMix64(seed: 48)
        let (m, k, n) = (3, 64, 40)
        let matrix = try pack(
            randomValues(count: n * k, rng: &rng), outDim: n, inDim: k)
        let a = (0..<(m * k)).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }

        let qBuffer = try makeBuffer(context.device, values: matrix.words)
        let scalesBuffer = try makeBuffer(context.device, values: matrix.scales)
        let biasesBuffer = try makeBuffer(context.device, values: matrix.biases)
        let aBuffer = try makeBuffer(context.device, values: a)
        // Poison an extra tile-height of rows past the real output.
        let poison = Float16(-1234.5)
        let paddedCount = (m + QuantGemmKernel.tileM) * n
        let outBuffer = try makeBuffer(
            context.device, values: [Float16](repeating: poison, count: paddedCount))

        try context.timedDispatch { encoder in
            try kernel.encodeGemm(
                into: encoder, q: qBuffer, qByteOffset: 0,
                scales: scalesBuffer, scalesByteOffset: 0,
                biases: biasesBuffer, biasesByteOffset: 0,
                input: aBuffer, batchM: m, outDim: n, inDim: k, output: outBuffer)
        }
        let all = readHalfs(outBuffer, count: paddedCount)
        let ref = try BLAS.sgemm(
            a: a.map(Float.init), b: matrix.dequantReference,
            m: m, k: k, n: n, transposeB: true)
        assertTierK(Array(all[0..<(m * n)]).map(Float.init), ref, "poison-guard rows")
        for i in (m * n)..<paddedCount {
            XCTAssertEqual(all[i].bitPattern, poison.bitPattern,
                           "row past M written at flat index \(i)")
        }
    }

    // MARK: - Host-wrapper rejects (loud, pre-dispatch) + dispatch counter

    func testGemmWrapperRejectsBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernel = try QuantGemmKernel(context: context)
        let q = try makeBuffer(
            context.device, values: [UInt32](repeating: 0, count: 8))
        let groups = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 1))
        let x = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 64))
        let out = try makeOutputBuffer(context.device, count: 64, elementStride: 2)

        func encodeExpectingError(
            _ expected: QuantKernelError,
            _ body: (MTLComputeCommandEncoder) throws -> Void,
            file: StaticString = #filePath, line: UInt = #line
        ) throws {
            var thrown: Error?
            try context.timedDispatch { encoder in
                do { try body(encoder) } catch { thrown = error }
            }
            guard let error = thrown as? QuantKernelError else {
                return XCTFail(
                    "expected QuantKernelError, got \(String(describing: thrown))",
                    file: file, line: line)
            }
            XCTAssertEqual(error, expected, file: file, line: line)
        }

        try encodeExpectingError(.nonPositiveDimension(name: "batchM", value: 0)) {
            try kernel.encodeGemm(
                into: $0, q: q, qByteOffset: 0, scales: groups, scalesByteOffset: 0,
                biases: groups, biasesByteOffset: 0, input: x, batchM: 0,
                outDim: 1, inDim: 64, output: out)
        }
        try encodeExpectingError(.inDimNotMultipleOfGroup(inDim: 96)) {
            try kernel.encodeGemm(
                into: $0, q: q, qByteOffset: 0, scales: groups, scalesByteOffset: 0,
                biases: groups, biasesByteOffset: 0, input: x, batchM: 1,
                outDim: 1, inDim: 96, output: out)
        }
        try encodeExpectingError(.misalignedOffset(buffer: "q", byteOffset: 2, alignment: 4)) {
            try kernel.encodeGemm(
                into: $0, q: q, qByteOffset: 2, scales: groups, scalesByteOffset: 0,
                biases: groups, biasesByteOffset: 0, input: x, batchM: 1,
                outDim: 1, inDim: 64, output: out)
        }
        // Input holds 64 halfs = one row of K=64; batchM 2 needs two.
        try encodeExpectingError(
            .bufferTooSmall(buffer: "input", requiredBytes: 256, actualBytes: 128)
        ) {
            try kernel.encodeGemm(
                into: $0, q: q, qByteOffset: 0, scales: groups, scalesByteOffset: 0,
                biases: groups, biasesByteOffset: 0, input: x, batchM: 2,
                outDim: 1, inDim: 64, output: out)
        }
        // Output too small for batchM × outDim halfs (triplet buffers sized
        // adequately so the output check is the one that fires).
        let q2 = try makeBuffer(
            context.device, values: [UInt32](repeating: 0, count: 16))
        let groups2 = try makeBuffer(
            context.device, values: [Float16](repeating: 0, count: 2))
        let tinyOut = try makeOutputBuffer(context.device, count: 1, elementStride: 2)
        try encodeExpectingError(
            .bufferTooSmall(buffer: "output", requiredBytes: 4, actualBytes: 2)
        ) {
            try kernel.encodeGemm(
                into: $0, q: q2, qByteOffset: 0, scales: groups2, scalesByteOffset: 0,
                biases: groups2, biasesByteOffset: 0, input: x, batchM: 1,
                outDim: 2, inDim: 64, output: tinyOut)
        }
    }

    func testGemmDispatchCounterCountsEncodes() throws {
        let context = try makeContextOrSkip()
        let kernel = try QuantGemmKernel(context: context)
        let counter = DispatchCounter()
        kernel.dispatchCounter = counter
        var rng = SplitMix64(seed: 49)
        let matrix = try pack(
            randomValues(count: 8 * 64, rng: &rng), outDim: 8, inDim: 64)
        _ = try gpuGemm(
            matrix, a: [Float16](repeating: 0.5, count: 2 * 64), m: 2,
            context: context, kernel: kernel)
        XCTAssertEqual(counter.count, 1, "one encode = one counted dispatch")
    }
}

/// Edge test 1's "every real weight shape" leg: the pinned artifact's five
/// distinct GEMM shapes vs sgemm over `PackedCheckpoint.dequantMatrix`,
/// through the mmap `GPUWeights` buffer at real file offsets. Skips cleanly
/// when the local-only artifact is absent (P3-4 precedent).
final class QuantGemmKernelRealArtifactTests: XCTestCase {

    override func setUpWithError() throws {
        guard FileManager.default.fileExists(
            atPath: SharedQuantModel.packedURL.path) else {
            throw XCTSkip(
                "packed artifact missing at \(SharedQuantModel.packedURL.path) "
                + "(local-only — produce it with `swift run qwen-metal-cli pack ...`)")
        }
        do {
            _ = try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    func testEveryDistinctRealShapeHoldsTierKIncludingRaggedM() throws {
        let context = try MetalContext()
        let kernel = try QuantGemmKernel(context: context)
        let packed = try PackedCheckpoint(
            path: SharedQuantModel.packedURL.path,
            expectedRevision: SharedCheckpoint.pinnedRevision)
        let weights = try GPUWeights(
            file: packed.file, context: context, residency: .mmap)

        // The five distinct [out, in] shapes of the pinned checkpoint:
        // q/o [2048, 2048], k/v [1024, 2048], gate/up [6144, 2048],
        // down [2048, 6144], embed (= tied lm_head) [151936, 2048].
        // M = 8 is the microbench gate point; the k_proj leg runs a ragged
        // M = 13 through a real shape.
        let shapes: [(name: String, outDim: Int, inDim: Int, m: Int)] = [
            ("model.layers.0.self_attn.q_proj.weight", 2048, 2048, 8),
            ("model.layers.0.self_attn.k_proj.weight", 1024, 2048, 13),
            ("model.layers.0.mlp.gate_proj.weight", 6144, 2048, 8),
            ("model.layers.0.mlp.down_proj.weight", 2048, 6144, 8),
            ("model.embed_tokens.weight", 151_936, 2048, 8),
        ]

        for shape in shapes {
            let dims = try packed.dims(for: shape.name)
            XCTAssertEqual(dims.outDim, shape.outDim, "\(shape.name) out dim")
            XCTAssertEqual(dims.inDim, shape.inDim, "\(shape.name) in dim")

            // Deterministic fp16-exact activations (microbench pattern).
            let a = (0..<(shape.m * shape.inDim)).map {
                Float16(QuantMatvecMicrobench.inputValue(at: $0))
            }
            let aBuffer = try XCTUnwrap(a.withUnsafeBytes {
                context.device.makeBuffer(
                    bytes: $0.baseAddress!, length: $0.count,
                    options: .storageModeShared)
            })
            let outBuffer = try XCTUnwrap(context.device.makeBuffer(
                length: shape.m * shape.outDim * 2, options: .storageModeShared))

            try context.timedDispatch { encoder in
                try kernel.encodeGemm(
                    into: encoder,
                    q: weights.buffer,
                    qByteOffset: try weights.byteOffset(for: shape.name + Q4G64.qSuffix),
                    scales: weights.buffer,
                    scalesByteOffset: try weights.byteOffset(
                        for: shape.name + Q4G64.scalesSuffix),
                    biases: weights.buffer,
                    biasesByteOffset: try weights.byteOffset(
                        for: shape.name + Q4G64.biasesSuffix),
                    input: aBuffer, batchM: shape.m,
                    outDim: shape.outDim, inDim: shape.inDim, output: outBuffer)
            }

            let ref = try BLAS.sgemm(
                a: a.map(Float.init), b: try packed.dequantMatrix(shape.name),
                m: shape.m, k: shape.inDim, n: shape.outDim, transposeB: true)
            let count = shape.m * shape.outDim
            let got = [Float16](UnsafeBufferPointer(
                start: outBuffer.contents().bindMemory(
                    to: Float16.self, capacity: count),
                count: count)).map(Float.init)

            let maxRef = ref.reduce(Float(0)) { max($0, abs($1)) }
            let gate = max(exp2(-9) * maxRef, exp2(-11))
            var worst: Float = 0
            for i in 0..<count { worst = max(worst, abs(got[i] - ref[i])) }
            XCTAssertLessThanOrEqual(
                worst, gate,
                "\(shape.name) (M=\(shape.m)): max |Δ| \(worst) exceeds Tier-K "
                    + "gate \(gate) (M = \(maxRef))")
        }
    }
}
