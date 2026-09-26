import XCTest
@testable import QwenMetalEngine
import Metal

/// BW-1 tests: read-only streaming bandwidth microbench (complements the
/// P0B-4 triad). Correctness gate (every simdgroup partial within 1e-6 of a
/// CPU oracle computed from the pure index pattern) and the measurement
/// protocol (three 384 MiB buffers = 1.125 GiB READ per iteration, float4
/// grid-stride, 64 float4 per thread per buffer, one float4 written per
/// simdgroup, sustained = median) were pre-committed in DECISIONS.md
/// 2026-09-26 BEFORE these tests were written.
final class ReadOnlyBandwidthKernelTests: XCTestCase {

    /// Pre-committed gate — DECISIONS.md 2026-09-26 (BW-1 entry).
    private static let tolerance: Float = 1e-6

    private func makeKernelOrSkip() throws -> ReadOnlyBandwidthKernel {
        do {
            return try ReadOnlyBandwidthKernel(context: MetalContext())
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    /// The hand-rolled CPU oracle, written directly from the pinned geometry
    /// (DECISIONS.md 2026-09-26): partial p covers lanes t in [32p, 32p+32)
    /// with t < T = N/256; lane t reads float4 t + k·T for k in 0..<64 from
    /// each of the three buffers. Independent of the engine's own oracle so
    /// the engine oracle is itself validated (see the test below).
    private func handRolledPartial(_ p: Int, elementCount n: Int) -> [Double] {
        let threads = n / (4 * 64)
        var sum = [Double](repeating: 0, count: 4)
        for lane in 0..<32 {
            let t = 32 * p + lane
            guard t < threads else { continue }
            for k in 0..<64 {
                let float4Index = t + k * threads
                for component in 0..<4 {
                    let element = 4 * float4Index + component
                    for buffer in 0..<3 {
                        sum[component] += Double(
                            ReadOnlyBandwidthKernel.inputValue(buffer: buffer, at: element))
                    }
                }
            }
        }
        return sum
    }

    // MARK: - Correctness vs CPU (hard rule 3: diff test before any optimization)

    func testEveryPartialMatchesCPUWithinPrecommittedGateOnSmallRun() throws {
        let kernel = try makeKernelOrSkip()
        // N = 300 × 256: T = 300 threads → 2 threadgroups of 256 → 16 partials.
        // Partial 9 is ragged (lanes 288–299 active, 300–319 inactive) and
        // partials 10–15 have no active lanes — both tail cases exercised.
        let count = 300 * 256
        let partials = ReadOnlyBandwidthKernel.partialCount(elementCount: count)
        XCTAssertEqual(partials, 16)
        let result = try kernel.run(
            elementCount: count, warmupIterations: 0, measuredIterations: 1,
            samplePartialIndices: Array(0..<partials)
        )
        var maxAbsDiff: Float = 0
        for p in 0..<partials {
            let gpu = try XCTUnwrap(result.partialSamples[p])
            let cpu = handRolledPartial(p, elementCount: count)
            for component in 0..<4 {
                maxAbsDiff = max(maxAbsDiff, abs(gpu[component] - Float(cpu[component])))
            }
        }
        XCTAssertLessThanOrEqual(
            maxAbsDiff, Self.tolerance,
            "GPU read-only reduce diverged from CPU reference beyond the pre-committed gate"
        )
        // Lanes beyond T contribute exactly zero.
        for p in 10..<16 {
            XCTAssertEqual(try XCTUnwrap(result.partialSamples[p]), SIMD4<Float>(repeating: 0))
        }
    }

    func testEngineOracleMatchesHandRolledOracle() {
        // CPU-only: the engine's oracle (used by the runtime gate in the CLI
        // and the app) must agree with the obviously-correct loop above.
        let count = 300 * 256
        for p in 0..<ReadOnlyBandwidthKernel.partialCount(elementCount: count) {
            let engine = ReadOnlyBandwidthKernel.cpuPartial(index: p, elementCount: count)
            let hand = handRolledPartial(p, elementCount: count)
            for component in 0..<4 {
                XCTAssertEqual(engine[component], Float(hand[component]))
            }
        }
    }

    func testPartialSumsStayExactlyRepresentable() {
        // The gate's rationale: every term is a multiple of 2^-11 with
        // |term| < 1, a partial holds 32 × 64 × 3 = 6144 terms per component,
        // so |partial| ≤ 6144 < 2^13 and every intermediate is exact in fp32.
        let termsPerComponent = 32 * ReadOnlyBandwidthKernel.float4sPerThreadPerBuffer
            * ReadOnlyBandwidthKernel.bufferCount
        XCTAssertEqual(termsPerComponent, 6144)
        XCTAssertLessThan(termsPerComponent, 1 << 13)
        for buffer in 0..<3 {
            for index in [0, 1, 4095, 4096, 1_000_003] {
                let v = ReadOnlyBandwidthKernel.inputValue(buffer: buffer, at: index)
                XCTAssertEqual((v * 2048).rounded(), v * 2048, "value not on the 2^-11 grid")
                XCTAssertGreaterThanOrEqual(v, -1)
                XCTAssertLessThan(v, 1)
            }
        }
    }

    // MARK: - The benchmark run (the BW-1 deliverable)

    func testBenchmarkWorkingSetDefeatsSLCAndWritesAreNegligible() {
        let n = ReadOnlyBandwidthKernel.benchmarkElementCount
        XCTAssertEqual(n, TriadBandwidthKernel.benchmarkElementCount,
                       "pinned: identical working set to the triad")
        XCTAssertGreaterThanOrEqual(3 * n * 4, 1 << 30)
        XCTAssertEqual(n % ReadOnlyBandwidthKernel.elementAlignment, 0)
        XCTAssertEqual(ReadOnlyBandwidthKernel.threadCount(elementCount: n), 393_216)
        let partials = ReadOnlyBandwidthKernel.partialCount(elementCount: n)
        XCTAssertEqual(partials, 12_288)
        let written = Double(partials * 16)
        XCTAssertLessThan(written / Double(3 * n * 4), 0.001,
                          "partial-sum writes must stay below 0.1% of the bytes read")
    }

    func testBenchmarkRunReportsSustainedGBpsWithDualTiming() throws {
        let kernel = try makeKernelOrSkip()
        let n = ReadOnlyBandwidthKernel.benchmarkElementCount
        let partials = ReadOnlyBandwidthKernel.partialCount(elementCount: n)
        let samples = [0, partials / 2, partials - 1]
        // Full pinned working set, fewer iterations than the official
        // protocol — this checks the machinery, not the row.
        let result = try kernel.run(
            elementCount: n, warmupIterations: 1, measuredIterations: 5,
            samplePartialIndices: samples
        )

        XCTAssertEqual(result.elementCount, n)
        XCTAssertEqual(result.threadCount, 393_216)
        XCTAssertEqual(result.partialCount, partials)
        XCTAssertEqual(result.bytesReadPerIteration, 3 * n * 4)
        XCTAssertEqual(result.bytesWrittenPerIteration, partials * 16)
        XCTAssertGreaterThanOrEqual(result.bytesReadPerIteration, 1 << 30)

        XCTAssertEqual(result.measuredGBps.count, 5)
        for gbps in result.measuredGBps {
            XCTAssertGreaterThan(gbps, 0)
            XCTAssertTrue(gbps.isFinite)
        }
        let sorted = result.measuredGBps.sorted()
        XCTAssertEqual(result.sustainedGBps, sorted[2])
        XCTAssertLessThanOrEqual(result.minGBps, result.sustainedGBps)
        XCTAssertGreaterThanOrEqual(result.maxGBps, result.sustainedGBps)

        XCTAssertEqual(result.warmupTimings.count, 1)
        XCTAssertEqual(result.measuredTimings.count, 5)
        for timing in result.warmupTimings + result.measuredTimings {
            XCTAssertGreaterThan(timing.gpuDuration, 0)
            XCTAssertGreaterThanOrEqual(timing.wallDuration, timing.gpuDuration)
        }

        // The measured dispatches really reduced the buffers (spot check vs CPU).
        for p in samples {
            let gpu = try XCTUnwrap(result.partialSamples[p])
            let cpu = ReadOnlyBandwidthKernel.cpuPartial(index: p, elementCount: n)
            for component in 0..<4 {
                XCTAssertLessThanOrEqual(abs(gpu[component] - cpu[component]), Self.tolerance)
            }
        }
    }

    func testMedianIsAverageOfMiddleTwoForEvenIterationCounts() throws {
        let kernel = try makeKernelOrSkip()
        let result = try kernel.run(
            elementCount: 4096, warmupIterations: 0, measuredIterations: 4,
            samplePartialIndices: []
        )
        let sorted = result.measuredGBps.sorted()
        XCTAssertEqual(result.sustainedGBps, (sorted[1] + sorted[2]) / 2)
    }

    // MARK: - Edge cases (explicit errors, not crashes)

    func testNonPositiveElementCountThrows() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(
            elementCount: 0, warmupIterations: 0, measuredIterations: 1,
            samplePartialIndices: []
        )) { error in
            XCTAssertEqual(error as? KernelInputError, .nonPositiveElementCount(count: 0))
        }
    }

    func testElementCountNotMultipleOf256Throws() throws {
        let kernel = try makeKernelOrSkip()
        // A multiple of 4 (float4-legal) that is not a multiple of the
        // 64-float4-per-thread geometry.
        XCTAssertThrowsError(try kernel.run(
            elementCount: 4100, warmupIterations: 0, measuredIterations: 1,
            samplePartialIndices: []
        )) { error in
            XCTAssertEqual(error as? KernelInputError,
                           .elementCountNotMultiple(of: 256, count: 4100))
        }
    }

    func testZeroMeasuredIterationsThrows() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(
            elementCount: 4096, warmupIterations: 2, measuredIterations: 0,
            samplePartialIndices: []
        )) { error in
            XCTAssertEqual(error as? KernelInputError, .invalidIterations(warmup: 2, measured: 0))
        }
    }

    func testNegativeWarmupIterationsThrows() throws {
        let kernel = try makeKernelOrSkip()
        XCTAssertThrowsError(try kernel.run(
            elementCount: 4096, warmupIterations: -1, measuredIterations: 1,
            samplePartialIndices: []
        )) { error in
            XCTAssertEqual(error as? KernelInputError, .invalidIterations(warmup: -1, measured: 1))
        }
    }

    func testOutOfRangePartialSampleIndexThrows() throws {
        let kernel = try makeKernelOrSkip()
        // N = 4096 → T = 16 → 1 threadgroup → 8 partials; index 8 is out.
        XCTAssertEqual(ReadOnlyBandwidthKernel.partialCount(elementCount: 4096), 8)
        XCTAssertThrowsError(try kernel.run(
            elementCount: 4096, warmupIterations: 0, measuredIterations: 1,
            samplePartialIndices: [8]
        )) { error in
            XCTAssertEqual(error as? KernelInputError, .sampleIndexOutOfRange(index: 8, count: 8))
        }
    }

    // MARK: - Diagnostic geometry (never a row; the pin stays 64)

    func testDiagnosticDepthStillMeetsGateOnEveryPartial() throws {
        // Depth 16: N = 300 × 64 → T = 300 again (same ragged tails), but each
        // lane reads 16 float4 per buffer; the oracle must follow the geometry.
        let depth = 16
        let kernel: ReadOnlyBandwidthKernel
        do {
            kernel = try ReadOnlyBandwidthKernel(context: MetalContext(), float4sPerThread: depth)
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
        let count = 300 * 4 * depth
        let partials = ReadOnlyBandwidthKernel.partialCount(elementCount: count, float4sPerThread: depth)
        XCTAssertEqual(partials, 16)
        let result = try kernel.run(
            elementCount: count, warmupIterations: 0, measuredIterations: 1,
            samplePartialIndices: Array(0..<partials))
        XCTAssertEqual(result.float4sPerThread, depth)
        XCTAssertEqual(result.threadCount, 300)
        var maxAbsDiff: Float = 0
        for p in 0..<partials {
            let gpu = try XCTUnwrap(result.partialSamples[p])
            let cpu = ReadOnlyBandwidthKernel.cpuPartial(
                index: p, elementCount: count, float4sPerThread: depth)
            for component in 0..<4 {
                maxAbsDiff = max(maxAbsDiff, abs(gpu[component] - cpu[component]))
            }
        }
        XCTAssertLessThanOrEqual(maxAbsDiff, Self.tolerance)
        // The pinned default is untouched by the diagnostic parameter.
        XCTAssertEqual(ReadOnlyBandwidthKernel.float4sPerThreadPerBuffer, 64)
        XCTAssertEqual(ReadOnlyBandwidthKernel.allowedFloat4sPerThread, 1...64)
    }

    func testDepthOutsideExactnessRangeThrows() throws {
        let context: MetalContext
        do { context = try MetalContext() } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
        for depth in [0, 65, -1] {
            XCTAssertThrowsError(try ReadOnlyBandwidthKernel(
                context: context, float4sPerThread: depth
            )) { error in
                XCTAssertEqual(error as? KernelInputError,
                               .float4sPerThreadOutOfRange(value: depth, allowed: 1...64))
            }
        }
    }

    func testDepthChangesAlignmentRequirement() throws {
        let kernel: ReadOnlyBandwidthKernel
        do {
            kernel = try ReadOnlyBandwidthKernel(context: MetalContext(), float4sPerThread: 16)
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
        // 4 × 16 = 64: 96 is float4-legal but not a multiple of 64.
        XCTAssertThrowsError(try kernel.run(
            elementCount: 96, warmupIterations: 0, measuredIterations: 1, samplePartialIndices: []
        )) { error in
            XCTAssertEqual(error as? KernelInputError, .elementCountNotMultiple(of: 64, count: 96))
        }
    }

    // MARK: - Init pattern stays inside the gate's stated domain

    func testInputPatternIsDeterministicDistinctPerBufferAndInRange() {
        for index in [0, 1, 4095, 4096, 1_000_003] {
            var seen: [Float] = []
            for buffer in 0..<3 {
                let v = ReadOnlyBandwidthKernel.inputValue(buffer: buffer, at: index)
                XCTAssertEqual(v, ReadOnlyBandwidthKernel.inputValue(buffer: buffer, at: index))
                XCTAssertTrue((-1.0..<1.0).contains(v))
                seen.append(v)
            }
            XCTAssertEqual(Set(seen).count, 3, "the three buffers carry distinct offsets")
        }
        // Buffers 0 and 1 reuse the triad's b/c patterns exactly (pinned).
        for index in [0, 7, 4095, 123_456] {
            XCTAssertEqual(ReadOnlyBandwidthKernel.inputValue(buffer: 0, at: index),
                           TriadBandwidthKernel.bValue(at: index))
            XCTAssertEqual(ReadOnlyBandwidthKernel.inputValue(buffer: 1, at: index),
                           TriadBandwidthKernel.cValue(at: index))
        }
    }
}
