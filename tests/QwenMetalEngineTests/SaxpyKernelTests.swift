import XCTest
@testable import QwenMetalEngine
import Metal

/// P0B-2 tests: saxpy kernel diffed against a CPU reference (PRD 0b.3).
/// This is the first real dispatch -> readback -> diff loop; the tolerance
/// (max |Δ| <= 1e-6 on inputs in [-1, 1]) was pre-committed in DECISIONS.md
/// 2026-08-21 BEFORE these tests were written, and never loosens.
final class SaxpyKernelTests: XCTestCase {

    /// Pre-committed gate — see DECISIONS.md 2026-08-21 (P0B-2 entry).
    private static let tolerance: Float = 1e-6

    /// Deterministic generator (SplitMix64) so the diff test is reproducible
    /// run-to-run; system RNGs are unseedable.
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

    private func makeKernelOrSkip() throws -> SaxpyKernel {
        do {
            return try SaxpyKernel(context: MetalContext())
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    /// The hand-rolled CPU oracle. Elementwise, so hard rule 8 (BLAS wrapper)
    /// does not apply; obviously-correct loop over the definition.
    private func cpuSaxpy(a: Float, x: [Float], y: [Float]) -> [Float] {
        zip(x, y).map { a * $0 + $1 }
    }

    // MARK: - Correctness vs CPU (the P0B-2 deliverable)

    func testSmallHandComputedCaseIsExact() throws {
        let kernel = try makeKernelOrSkip()
        let result = try kernel.run(a: 2.0, x: [1, 2, 3], y: [10, 20, 30])
        // Values chosen so every product and sum is exactly representable:
        // no tolerance needed, the GPU must match bit-for-bit.
        XCTAssertEqual(result.values, [12, 24, 36])
    }

    func testLargeOddSizeMatchesCPUWithinPrecommittedTolerance() throws {
        let kernel = try makeKernelOrSkip()

        // Odd size, deliberately not a multiple of any threadgroup width, so
        // the ragged final threadgroup / bounds handling is exercised.
        let count = 1_000_003
        var rng = SplitMix64(seed: 0xC0FFEE)
        let x = (0..<count).map { _ in Float.random(in: -1...1, using: &rng) }
        let y = (0..<count).map { _ in Float.random(in: -1...1, using: &rng) }
        let a = Float.random(in: -1...1, using: &rng)

        let result = try kernel.run(a: a, x: x, y: y)
        let expected = cpuSaxpy(a: a, x: x, y: y)

        XCTAssertEqual(result.values.count, expected.count)
        var maxAbsDiff: Float = 0
        for i in 0..<count {
            maxAbsDiff = max(maxAbsDiff, abs(result.values[i] - expected[i]))
        }
        XCTAssertLessThanOrEqual(
            maxAbsDiff, Self.tolerance,
            "GPU saxpy diverged from CPU reference beyond the pre-committed gate"
        )
    }

    func testInputsAreNotMutated() throws {
        let kernel = try makeKernelOrSkip()
        let x: [Float] = [1, -2, 3]
        let y: [Float] = [0.5, 0.25, -0.125]
        _ = try kernel.run(a: -1.5, x: x, y: y)
        XCTAssertEqual(x, [1, -2, 3])
        XCTAssertEqual(y, [0.5, 0.25, -0.125])
    }

    // MARK: - Edge cases (explicit errors, not crashes)

    func testMismatchedLengthsThrow() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(a: 1, x: [1, 2, 3], y: [1, 2])) { error in
            guard case KernelInputError.lengthMismatch(let xCount, let yCount) = error else {
                return XCTFail("Expected lengthMismatch, got \(error)")
            }
            XCTAssertEqual(xCount, 3)
            XCTAssertEqual(yCount, 2)
        }
    }

    func testEmptyInputThrows() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(a: 1, x: [], y: [])) { error in
            guard case KernelInputError.emptyInput = error else {
                return XCTFail("Expected emptyInput, got \(error)")
            }
        }
    }

    // MARK: - Dual timing rides along (hard rule 7)

    func testRunReportsDualTiming() throws {
        let kernel = try makeKernelOrSkip()
        // Large enough that the GPU timestamp interval is reliably nonzero
        // (same reasoning as the harness timing test).
        let n = 1 << 24
        let x = [Float](repeating: 0.5, count: n)
        let y = [Float](repeating: 0.25, count: n)
        let result = try kernel.run(a: 3.0, x: x, y: y)
        XCTAssertGreaterThan(result.timing.gpuDuration, 0)
        XCTAssertGreaterThanOrEqual(result.timing.wallDuration, result.timing.gpuDuration)
    }
}
