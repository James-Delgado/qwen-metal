import XCTest
@testable import QwenMetalEngine

/// P1-3 tests: the Accelerate sgemm wrapper (hard rule 8 — the single matmul
/// path for the whole CPU reference) diffed against a TEST-ONLY naive fp32
/// triple loop on odd-shaped, non-square, non-power-of-two matrices to flush
/// transpose/leading-dimension bugs. The gate — per element
/// |sgemm − tripleLoop| <= 2·γ_K·Σ|a||b|, γ_K = K·u/(1−K·u), u = 2^-24 —
/// was pre-committed in DECISIONS.md 2026-08-23 BEFORE these tests were
/// written, and never loosens. The triple loop lives ONLY here (phase-0-1.md
/// build step 3); the external HF fp32 oracle independently polices the
/// wrapper once the model pipeline exists.
final class SgemmTests: XCTestCase {

    /// Pre-committed gate basis — see DECISIONS.md 2026-08-23 (P1-3 entry).
    private static let unitRoundoff: Float = 0x1p-24

    /// Deterministic generator (SplitMix64), same reproducibility rationale as
    /// the P0B kernel tests; system RNGs are unseedable.
    private struct SplitMix64: RandomNumberGenerator {
        private var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }

    private func randomFloats(count: Int, rng: inout SplitMix64) -> [Float] {
        (0..<count).map { _ in Float.random(in: -1...1, using: &rng) }
    }

    /// The TEST-ONLY oracle: naive fp32 triple loop over row-major m×k · k×n,
    /// also accumulating Σ|a||b| per element (the gate's error-bound basis).
    private func cpuMatmul(
        a: [Float], b: [Float], m: Int, k: Int, n: Int
    ) -> (values: [Float], absSums: [Float]) {
        var c = [Float](repeating: 0, count: m * n)
        var s = [Float](repeating: 0, count: m * n)
        for i in 0..<m {
            for j in 0..<n {
                var acc: Float = 0
                var absAcc: Float = 0
                for p in 0..<k {
                    let product = a[i * k + p] * b[p * n + j]
                    acc += product
                    absAcc += abs(product)
                }
                c[i * n + j] = acc
                s[i * n + j] = absAcc
            }
        }
        return (c, s)
    }

    /// Test-side transpose so the oracle stays a plain row-major triple loop:
    /// the logical operands are generated untransposed, and the STORED array
    /// handed to the wrapper is transposed when the flag under test is set.
    private func transposed(_ x: [Float], rows: Int, cols: Int) -> [Float] {
        var t = [Float](repeating: 0, count: rows * cols)
        for r in 0..<rows {
            for c in 0..<cols {
                t[c * rows + r] = x[r * cols + c]
            }
        }
        return t
    }

    /// Asserts the pre-committed gate over every output element.
    private func assertWithinGate(
        result: [Float], ref: [Float], absSums: [Float], k: Int,
        label: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(result.count, ref.count, file: file, line: line)
        let ku = Float(k) * Self.unitRoundoff
        let gammaK = ku / (1 - ku)
        var worstExcess: Float = -.greatestFiniteMagnitude
        var worstIndex = 0
        for idx in 0..<ref.count {
            let diff = abs(result[idx] - ref[idx])
            let bound = 2 * gammaK * absSums[idx]
            if diff - bound > worstExcess {
                worstExcess = diff - bound
                worstIndex = idx
            }
        }
        XCTAssertLessThanOrEqual(
            worstExcess, 0,
            """
            sgemm diverged from the triple-loop reference beyond the \
            pre-committed gate (\(label)) at flat index \(worstIndex): \
            sgemm=\(result[worstIndex]) ref=\(ref[worstIndex])
            """,
            file: file, line: line
        )
    }

    // MARK: - Correctness vs triple loop (the P1-3 deliverable)

    func testSmallHandComputedIntegerCaseIsExact() throws {
        // 2x3 · 3x2, integer values: every product and partial sum is exactly
        // representable in fp32, so the wrapper must match bit-for-bit
        // (exactness carve-out in the DECISIONS.md gate entry).
        let a: [Float] = [1, 2, 3,
                          4, 5, 6]
        let b: [Float] = [7, 8,
                          9, 10,
                          11, 12]
        let c = try BLAS.sgemm(a: a, b: b, m: 2, k: 3, n: 2)
        XCTAssertEqual(c, [58, 64,
                           139, 154])
    }

    func testOddShapesMatchTripleLoopWithinPrecommittedGate() throws {
        // Odd in every dimension, non-square, non-power-of-two; m != k != n
        // makes any transpose or leading-dimension swap a loud failure.
        let shapes: [(m: Int, k: Int, n: Int)] = [(67, 129, 45), (301, 257, 173)]
        var rng = SplitMix64(seed: 0x5EED_C0DE)
        for shape in shapes {
            let a = randomFloats(count: shape.m * shape.k, rng: &rng)
            let b = randomFloats(count: shape.k * shape.n, rng: &rng)
            let c = try BLAS.sgemm(a: a, b: b, m: shape.m, k: shape.k, n: shape.n)
            let ref = cpuMatmul(a: a, b: b, m: shape.m, k: shape.k, n: shape.n)
            assertWithinGate(
                result: c, ref: ref.values, absSums: ref.absSums, k: shape.k,
                label: "shape \(shape)"
            )
        }
    }

    func testTransposeFlagCombinationsMatchTripleLoop() throws {
        // The wrapper's future callers hit every combination: projections use
        // (false, false), per-head QK^T and tied-embeddings lm_head use
        // transposeB, and transposeA guards against storage-order surprises.
        // Odd non-square shape so a wrong lda/ldb misreads memory loudly.
        let (m, k, n) = (35, 53, 29)
        var rng = SplitMix64(seed: 0x7A05_E5E7)
        let logicalA = randomFloats(count: m * k, rng: &rng)
        let logicalB = randomFloats(count: k * n, rng: &rng)
        let ref = cpuMatmul(a: logicalA, b: logicalB, m: m, k: k, n: n)
        for transposeA in [false, true] {
            for transposeB in [false, true] {
                let storedA = transposeA ? transposed(logicalA, rows: m, cols: k) : logicalA
                let storedB = transposeB ? transposed(logicalB, rows: k, cols: n) : logicalB
                let c = try BLAS.sgemm(
                    a: storedA, b: storedB, m: m, k: k, n: n,
                    transposeA: transposeA, transposeB: transposeB
                )
                assertWithinGate(
                    result: c, ref: ref.values, absSums: ref.absSums, k: k,
                    label: "transposeA=\(transposeA) transposeB=\(transposeB)"
                )
            }
        }
    }

    func testVectorShapedEdgesMatchTripleLoop() throws {
        // Degenerate matmul shapes: dot product, row·matrix, matrix·column,
        // and the k=1 outer product (a structurally distinct GEMM case —
        // single-term reduction with degenerate lda/ldb).
        let shapes: [(m: Int, k: Int, n: Int)] = [(1, 513, 1), (1, 33, 21), (19, 33, 1), (5, 1, 7)]
        var rng = SplitMix64(seed: 0xB1A5_F00D)
        for shape in shapes {
            let a = randomFloats(count: shape.m * shape.k, rng: &rng)
            let b = randomFloats(count: shape.k * shape.n, rng: &rng)
            let c = try BLAS.sgemm(a: a, b: b, m: shape.m, k: shape.k, n: shape.n)
            let ref = cpuMatmul(a: a, b: b, m: shape.m, k: shape.k, n: shape.n)
            assertWithinGate(
                result: c, ref: ref.values, absSums: ref.absSums, k: shape.k,
                label: "shape \(shape)"
            )
        }
    }

    func testInputsAreNotMutated() throws {
        let a: [Float] = [1, -2, 3, 4]
        let b: [Float] = [0.5, 0.25, -0.125, 8]
        _ = try BLAS.sgemm(a: a, b: b, m: 2, k: 2, n: 2)
        XCTAssertEqual(a, [1, -2, 3, 4])
        XCTAssertEqual(b, [0.5, 0.25, -0.125, 8])
    }

    // MARK: - Edge cases (explicit errors, not crashes)

    func testNonPositiveDimensionsThrow() {
        XCTAssertThrowsError(try BLAS.sgemm(a: [], b: [1], m: 0, k: 1, n: 1)) { error in
            guard case KernelInputError.nonPositiveDimensions(let m, let k, let n) = error else {
                return XCTFail("Expected nonPositiveDimensions, got \(error)")
            }
            XCTAssertEqual(m, 0)
            XCTAssertEqual(k, 1)
            XCTAssertEqual(n, 1)
        }
        XCTAssertThrowsError(try BLAS.sgemm(a: [1], b: [1], m: 1, k: 1, n: -3)) { error in
            guard case KernelInputError.nonPositiveDimensions = error else {
                return XCTFail("Expected nonPositiveDimensions, got \(error)")
            }
        }
    }

    func testElementCountMismatchThrows() {
        XCTAssertThrowsError(
            try BLAS.sgemm(a: [1, 2, 3], b: [1, 2], m: 2, k: 2, n: 1)
        ) { error in
            guard case KernelInputError.elementCountMismatch(let matrix, let expected, let actual) = error else {
                return XCTFail("Expected elementCountMismatch, got \(error)")
            }
            XCTAssertEqual(matrix, "A")
            XCTAssertEqual(expected, 4)
            XCTAssertEqual(actual, 3)
        }
        // Transpose flags change storage interpretation but not element counts.
        XCTAssertThrowsError(
            try BLAS.sgemm(a: [1, 2], b: [1, 2, 3], m: 2, k: 1, n: 2, transposeB: true)
        ) { error in
            guard case KernelInputError.elementCountMismatch(let matrix, let expected, let actual) = error else {
                return XCTFail("Expected elementCountMismatch, got \(error)")
            }
            XCTAssertEqual(matrix, "B")
            XCTAssertEqual(expected, 2)
            XCTAssertEqual(actual, 3)
        }
    }
}
