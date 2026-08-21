import XCTest
@testable import QwenMetalEngine
import Metal

/// P0B-3 tests: naive fp16 matmul kernel diffed against a CPU reference on odd
/// shapes (PRD 0b.4). The gate — per element |gpu − ref| <= max(2^-9, 2^-9·|ref|)
/// against the UNROUNDED fp32 triple-loop reference, fp16 inputs in [-1, 1] —
/// was pre-committed in DECISIONS.md 2026-08-21 BEFORE these tests were written,
/// and never loosens. The triple loop is TEST-ONLY (hard rule 8 binds the Phase 1
/// model reference, not this toy-kernel oracle — see the same DECISIONS entry).
final class NaiveMatmulKernelTests: XCTestCase {

    /// Pre-committed gate — see DECISIONS.md 2026-08-21 (P0B-3 entry).
    private static let absoluteTolerance: Float = 0x1p-9
    private static let relativeTolerance: Float = 0x1p-9

    /// Deterministic generator (SplitMix64), same reproducibility rationale as
    /// the saxpy tests; system RNGs are unseedable.
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

    private func makeKernelOrSkip() throws -> NaiveMatmulKernel {
        do {
            return try NaiveMatmulKernel(context: MetalContext())
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    private func randomHalves(count: Int, rng: inout SplitMix64) -> [Float16] {
        (0..<count).map { _ in Float16(Float.random(in: -1...1, using: &rng)) }
    }

    /// The TEST-ONLY oracle: naive fp32 triple loop over the same fp16 inputs,
    /// returning the unrounded fp32 accumulation (the gate's comparison basis).
    private func cpuMatmul(a: [Float16], b: [Float16], m: Int, k: Int, n: Int) -> [Float] {
        var c = [Float](repeating: 0, count: m * n)
        for i in 0..<m {
            for j in 0..<n {
                var acc: Float = 0
                for p in 0..<k {
                    acc += Float(a[i * k + p]) * Float(b[p * n + j])
                }
                c[i * n + j] = acc
            }
        }
        return c
    }

    /// Asserts the pre-committed gate over every output element.
    private func assertWithinGate(
        gpu: [Float16], ref: [Float], shape: (m: Int, k: Int, n: Int),
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(gpu.count, ref.count, file: file, line: line)
        var worstExcess: Float = -.greatestFiniteMagnitude
        var worstIndex = 0
        for idx in 0..<ref.count {
            let diff = abs(Float(gpu[idx]) - ref[idx])
            let bound = max(Self.absoluteTolerance, Self.relativeTolerance * abs(ref[idx]))
            if diff - bound > worstExcess {
                worstExcess = diff - bound
                worstIndex = idx
            }
        }
        XCTAssertLessThanOrEqual(
            worstExcess, 0,
            """
            GPU matmul diverged from CPU reference beyond the pre-committed gate \
            at flat index \(worstIndex) of shape \(shape): \
            gpu=\(Float(gpu[worstIndex])) ref=\(ref[worstIndex])
            """,
            file: file, line: line
        )
    }

    // MARK: - Correctness vs CPU (the P0B-3 deliverable)

    func testSmallHandComputedCaseIsExact() throws {
        let kernel = try makeKernelOrSkip()
        // 2x3 · 3x2, integer values: every product and partial sum is exactly
        // representable in fp16, so the GPU must match bit-for-bit.
        let a: [Float16] = [1, 2, 3,
                            4, 5, 6]
        let b: [Float16] = [7, 8,
                            9, 10,
                            11, 12]
        let result = try kernel.run(a: a, b: b, m: 2, k: 3, n: 2)
        XCTAssertEqual(result.values, [58, 64,
                                       139, 154])
    }

    func testOddShapesMatchCPUWithinPrecommittedTolerance() throws {
        let kernel = try makeKernelOrSkip()
        // Odd in every dimension, none a multiple of any threadgroup width, so
        // ragged 2D threadgroup edges and all three strides are exercised.
        // m != k != n also makes any transpose/stride swap a loud failure.
        let shapes: [(m: Int, k: Int, n: Int)] = [(67, 129, 45), (301, 257, 173)]
        var rng = SplitMix64(seed: 0xBEEFCAFE)
        for shape in shapes {
            let a = randomHalves(count: shape.m * shape.k, rng: &rng)
            let b = randomHalves(count: shape.k * shape.n, rng: &rng)
            let result = try kernel.run(a: a, b: b, m: shape.m, k: shape.k, n: shape.n)
            let ref = cpuMatmul(a: a, b: b, m: shape.m, k: shape.k, n: shape.n)
            assertWithinGate(gpu: result.values, ref: ref, shape: shape)
        }
    }

    func testVectorShapedEdgesMatchCPU() throws {
        let kernel = try makeKernelOrSkip()
        // Degenerate matmul shapes: dot product (1xK·Kx1), row·matrix, and
        // matrix·column — the m==1 / n==1 grid edges.
        let shapes: [(m: Int, k: Int, n: Int)] = [(1, 513, 1), (1, 33, 21), (19, 33, 1)]
        var rng = SplitMix64(seed: 0xDEADBEA7)
        for shape in shapes {
            let a = randomHalves(count: shape.m * shape.k, rng: &rng)
            let b = randomHalves(count: shape.k * shape.n, rng: &rng)
            let result = try kernel.run(a: a, b: b, m: shape.m, k: shape.k, n: shape.n)
            let ref = cpuMatmul(a: a, b: b, m: shape.m, k: shape.k, n: shape.n)
            assertWithinGate(gpu: result.values, ref: ref, shape: shape)
        }
    }

    func testInputsAreNotMutated() throws {
        let kernel = try makeKernelOrSkip()
        let a: [Float16] = [1, -2, 3, 4]
        let b: [Float16] = [0.5, 0.25, -0.125, 8]
        _ = try kernel.run(a: a, b: b, m: 2, k: 2, n: 2)
        XCTAssertEqual(a, [1, -2, 3, 4])
        XCTAssertEqual(b, [0.5, 0.25, -0.125, 8])
    }

    // MARK: - Edge cases (explicit errors, not crashes)

    func testNonPositiveDimensionsThrow() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(a: [], b: [1], m: 0, k: 1, n: 1)) { error in
            guard case KernelInputError.nonPositiveDimensions(let m, let k, let n) = error else {
                return XCTFail("Expected nonPositiveDimensions, got \(error)")
            }
            XCTAssertEqual(m, 0)
            XCTAssertEqual(k, 1)
            XCTAssertEqual(n, 1)
        }
        XCTAssertThrowsError(try kernel.run(a: [1], b: [1], m: 1, k: 1, n: -3)) { error in
            guard case KernelInputError.nonPositiveDimensions = error else {
                return XCTFail("Expected nonPositiveDimensions, got \(error)")
            }
        }
    }

    func testElementCountMismatchThrows() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(
            try kernel.run(a: [1, 2, 3], b: [1, 2], m: 2, k: 2, n: 1)
        ) { error in
            guard case KernelInputError.elementCountMismatch(let matrix, let expected, let actual) = error else {
                return XCTFail("Expected elementCountMismatch, got \(error)")
            }
            XCTAssertEqual(matrix, "A")
            XCTAssertEqual(expected, 4)
            XCTAssertEqual(actual, 3)
        }
        XCTAssertThrowsError(
            try kernel.run(a: [1, 2], b: [1, 2, 3], m: 2, k: 1, n: 2)
        ) { error in
            guard case KernelInputError.elementCountMismatch(let matrix, let expected, let actual) = error else {
                return XCTFail("Expected elementCountMismatch, got \(error)")
            }
            XCTAssertEqual(matrix, "B")
            XCTAssertEqual(expected, 2)
            XCTAssertEqual(actual, 3)
        }
    }

    // MARK: - Dual timing rides along (hard rule 7)

    func testRunReportsDualTiming() throws {
        let kernel = try makeKernelOrSkip()
        // 512^3 MACs is enough work that the GPU timestamp interval is
        // reliably nonzero (same reasoning as the harness timing test).
        let dim = 512
        var rng = SplitMix64(seed: 0x7141)
        let a = randomHalves(count: dim * dim, rng: &rng)
        let b = randomHalves(count: dim * dim, rng: &rng)
        let result = try kernel.run(a: a, b: b, m: dim, k: dim, n: dim)
        XCTAssertGreaterThan(result.timing.gpuDuration, 0)
        XCTAssertGreaterThanOrEqual(result.timing.wallDuration, result.timing.gpuDuration)
    }
}
