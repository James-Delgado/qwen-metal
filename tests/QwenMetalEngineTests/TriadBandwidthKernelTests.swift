import XCTest
@testable import QwenMetalEngine
import Metal

/// P0B-4 tests: streaming triad bandwidth microbench (PRD 0a.6/0b.5).
/// Correctness gate (max |Δ| <= 1e-6 vs CPU on the deterministic init pattern)
/// and the measurement protocol (float4 triad, >= 1 GiB streamed/iteration,
/// sustained = median of measured iterations) were pre-committed in
/// DECISIONS.md 2026-08-21 BEFORE these tests were written.
final class TriadBandwidthKernelTests: XCTestCase {

    /// Pre-committed gate — see DECISIONS.md 2026-08-21 (P0B-4 entry).
    private static let tolerance: Float = 1e-6

    private func makeKernelOrSkip() throws -> TriadBandwidthKernel {
        do {
            return try TriadBandwidthKernel(context: MetalContext())
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    /// The hand-rolled CPU oracle. Elementwise, so hard rule 8 (BLAS wrapper)
    /// does not apply; obviously-correct evaluation of the definition.
    private func cpuTriad(at index: Int) -> Float {
        TriadBandwidthKernel.bValue(at: index)
            + TriadBandwidthKernel.scalar * TriadBandwidthKernel.cValue(at: index)
    }

    // MARK: - Correctness vs CPU (hard rule 3: diff test before any optimization)

    func testEveryElementMatchesCPUWithinPrecommittedGateOnSmallRun() throws {
        let kernel = try makeKernelOrSkip()
        let count = 4096
        let result = try kernel.run(
            elementCount: count, warmupIterations: 0, measuredIterations: 1,
            sampleIndices: Array(0..<count)
        )
        var maxAbsDiff: Float = 0
        for index in 0..<count {
            let gpu = try XCTUnwrap(result.outputSamples[index])
            maxAbsDiff = max(maxAbsDiff, abs(gpu - cpuTriad(at: index)))
        }
        XCTAssertLessThanOrEqual(
            maxAbsDiff, Self.tolerance,
            "GPU triad diverged from CPU reference beyond the pre-committed gate"
        )
    }

    // MARK: - The benchmark run (the P0B-4 deliverable)

    func testBenchmarkWorkingSetDefeatsSLC() {
        // Pinned protocol: bytes streamed per iteration must be >= 1 GiB so the
        // measured figure is DRAM, not any Apple SLC (PRD risk table).
        let n = TriadBandwidthKernel.benchmarkElementCount
        XCTAssertGreaterThanOrEqual(3 * n * 4, 1 << 30)
        XCTAssertEqual(n % 4, 0, "float4 kernel requires a multiple-of-4 count")
    }

    func testBenchmarkRunReportsSustainedGBpsWithDualTiming() throws {
        let kernel = try makeKernelOrSkip()
        // Full pinned working set (>= 1 GiB streamed) but fewer iterations than
        // the official protocol — this test checks the machinery, not the row.
        let sampleIndices = [0, TriadBandwidthKernel.benchmarkElementCount / 2,
                             TriadBandwidthKernel.benchmarkElementCount - 1]
        let result = try kernel.run(
            elementCount: TriadBandwidthKernel.benchmarkElementCount,
            warmupIterations: 1, measuredIterations: 5,
            sampleIndices: sampleIndices
        )

        XCTAssertEqual(result.elementCount, TriadBandwidthKernel.benchmarkElementCount)
        XCTAssertEqual(result.bytesPerIteration,
                       3 * TriadBandwidthKernel.benchmarkElementCount * 4)
        XCTAssertGreaterThanOrEqual(result.bytesPerIteration, 1 << 30)

        // Reports sustained GB/s: finite, positive, and the pinned median.
        XCTAssertEqual(result.measuredGBps.count, 5)
        for gbps in result.measuredGBps {
            XCTAssertGreaterThan(gbps, 0)
            XCTAssertTrue(gbps.isFinite)
        }
        let sorted = result.measuredGBps.sorted()
        XCTAssertEqual(result.sustainedGBps, sorted[2])
        XCTAssertLessThanOrEqual(result.minGBps, result.sustainedGBps)
        XCTAssertGreaterThanOrEqual(result.maxGBps, result.sustainedGBps)

        // Dual timing rides along on every iteration (hard rule 7).
        XCTAssertEqual(result.warmupTimings.count, 1)
        XCTAssertEqual(result.measuredTimings.count, 5)
        for timing in result.warmupTimings + result.measuredTimings {
            XCTAssertGreaterThan(timing.gpuDuration, 0)
            XCTAssertGreaterThanOrEqual(timing.wallDuration, timing.gpuDuration)
        }

        // The measured buffer really went through the triad (spot-check vs CPU).
        for index in sampleIndices {
            let gpu = try XCTUnwrap(result.outputSamples[index])
            XCTAssertLessThanOrEqual(abs(gpu - cpuTriad(at: index)), Self.tolerance)
        }
    }

    func testMedianIsAverageOfMiddleTwoForEvenIterationCounts() throws {
        let kernel = try makeKernelOrSkip()
        // Small working set: this checks the median arithmetic, not bandwidth.
        let result = try kernel.run(
            elementCount: 4096, warmupIterations: 0, measuredIterations: 4,
            sampleIndices: []
        )
        let sorted = result.measuredGBps.sorted()
        XCTAssertEqual(result.sustainedGBps, (sorted[1] + sorted[2]) / 2)
    }

    // MARK: - Edge cases (explicit errors, not crashes)

    func testNonPositiveElementCountThrows() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(
            elementCount: 0, warmupIterations: 0, measuredIterations: 1,
            sampleIndices: []
        )) { error in
            XCTAssertEqual(error as? KernelInputError,
                           .nonPositiveElementCount(count: 0))
        }
    }

    func testNonMultipleOfFourElementCountThrows() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(
            elementCount: 4098, warmupIterations: 0, measuredIterations: 1,
            sampleIndices: []
        )) { error in
            XCTAssertEqual(error as? KernelInputError,
                           .elementCountNotMultiple(of: 4, count: 4098))
        }
    }

    func testZeroMeasuredIterationsThrows() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(
            elementCount: 4096, warmupIterations: 2, measuredIterations: 0,
            sampleIndices: []
        )) { error in
            XCTAssertEqual(error as? KernelInputError,
                           .invalidIterations(warmup: 2, measured: 0))
        }
    }

    func testNegativeWarmupIterationsThrows() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(
            elementCount: 4096, warmupIterations: -1, measuredIterations: 1,
            sampleIndices: []
        )) { error in
            XCTAssertEqual(error as? KernelInputError,
                           .invalidIterations(warmup: -1, measured: 1))
        }
    }

    func testOutOfRangeSampleIndexThrows() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(
            elementCount: 4096, warmupIterations: 0, measuredIterations: 1,
            sampleIndices: [4096]
        )) { error in
            XCTAssertEqual(error as? KernelInputError,
                           .sampleIndexOutOfRange(index: 4096, count: 4096))
        }
    }

    // MARK: - Init pattern stays inside the gate's stated domain

    func testInitPatternIsDeterministicAndInRange() {
        for index in [0, 1, 4095, 4096, 1_000_003] {
            let b = TriadBandwidthKernel.bValue(at: index)
            let c = TriadBandwidthKernel.cValue(at: index)
            XCTAssertEqual(b, TriadBandwidthKernel.bValue(at: index))
            XCTAssertEqual(c, TriadBandwidthKernel.cValue(at: index))
            XCTAssertTrue((-1.0...1.0).contains(b))
            XCTAssertTrue((-1.0...1.0).contains(c))
        }
    }
}
