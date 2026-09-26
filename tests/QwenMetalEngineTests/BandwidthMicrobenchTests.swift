import XCTest
@testable import QwenMetalEngine

/// BW-1: the shared bandwidth microbench harness behind
/// `qwen-metal-cli bandwidth --kernel triad|read-only` and the app's
/// microbench picker. Owns the pre-committed refuse-to-report gate (P0B-4
/// precedent) and the export text, so both surfaces print the same thing.
final class BandwidthMicrobenchTests: XCTestCase {

    private func makeBenchOrSkip() throws -> BandwidthMicrobench {
        do {
            return try BandwidthMicrobench(context: MetalContext())
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    // MARK: - Gate evaluation (pure; no GPU)

    func testGateToleranceIsThePrecommittedValue() {
        XCTAssertEqual(BandwidthMicrobench.tolerance, 1e-6)
    }

    func testGatePassesAndReportsMaxDeltaOnMatchingSamples() throws {
        // 2^-21 is exactly representable at magnitude 0.5 (ulp there is 2^-24),
        // so the reported max |Δ| must be bit-exact.
        let delta = Float(0x1p-21)
        let samples: [BandwidthMicrobench.GateSample] = [
            .init(index: 0, gpu: [1.0, 2.0], reference: [1.0, 2.0]),
            .init(index: 5, gpu: [0.5], reference: [0.5 + delta]),
        ]
        let maxDelta = try BandwidthMicrobench.checkGate(
            kernel: .readOnly, samples: samples, tolerance: 1e-6)
        XCTAssertEqual(maxDelta, delta)
    }

    func testGateRefusesOnAnyMismatchedComponent() {
        let samples: [BandwidthMicrobench.GateSample] = [
            .init(index: 0, gpu: [1.0, 2.0, 3.0, 4.0], reference: [1.0, 2.0, 3.0, 4.0]),
            .init(index: 7, gpu: [1.0, 2.0, 3.0, 4.0], reference: [1.0, 2.0, 3.5, 4.0]),
        ]
        XCTAssertThrowsError(try BandwidthMicrobench.checkGate(
            kernel: .triad, samples: samples, tolerance: 1e-6
        )) { error in
            XCTAssertEqual(
                error as? BandwidthMicrobenchError,
                .gateFailed(kernel: .triad, sampleIndex: 7, maxAbsDelta: 0.5, tolerance: 1e-6))
            XCTAssertTrue("\(error)".contains("WITHHELD"))
        }
    }

    func testGateRefusesOnNaNOutput() {
        let samples: [BandwidthMicrobench.GateSample] = [
            .init(index: 3, gpu: [.nan], reference: [0.25]),
        ]
        XCTAssertThrowsError(try BandwidthMicrobench.checkGate(
            kernel: .readOnly, samples: samples, tolerance: 1e-6))
    }

    func testGateRefusesOnComponentCountMismatch() {
        let samples: [BandwidthMicrobench.GateSample] = [
            .init(index: 0, gpu: [1.0, 2.0], reference: [1.0]),
        ]
        XCTAssertThrowsError(try BandwidthMicrobench.checkGate(
            kernel: .readOnly, samples: samples, tolerance: 1e-6))
    }

    // MARK: - Kernel selection parses the CLI/app vocabulary

    func testKernelRawValuesAreTheCLIVocabulary() {
        XCTAssertEqual(BandwidthKernel(rawValue: "triad"), .triad)
        XCTAssertEqual(BandwidthKernel(rawValue: "read-only"), .readOnly)
        XCTAssertNil(BandwidthKernel(rawValue: "readonly"))
        XCTAssertEqual(BandwidthKernel.allCases, [.triad, .readOnly])
    }

    // MARK: - Runs (small working set: machinery, not a row)

    func testReadOnlyRunProducesReportWithGateAndByteAccounting() throws {
        let bench = try makeBenchOrSkip()
        let n = 300 * 256
        let report = try bench.run(
            kernel: .readOnly, elementCount: n, warmupIterations: 1, measuredIterations: 3)
        XCTAssertEqual(report.kernel, .readOnly)
        XCTAssertEqual(report.bytesCountedPerIteration, 3 * n * 4)
        XCTAssertEqual(report.bytesWrittenPerIteration, 16 * 16)
        XCTAssertFalse(report.writesCounted)
        XCTAssertEqual(report.warmupCount, 1)
        XCTAssertEqual(report.measuredTimings.count, 3)
        XCTAssertEqual(report.measuredGBps.count, 3)
        XCTAssertEqual(report.gateSampleCount, 3)
        XCTAssertLessThanOrEqual(report.gateMaxAbsDelta, BandwidthMicrobench.tolerance)
        XCTAssertEqual(report.gateTolerance, BandwidthMicrobench.tolerance)
        XCTAssertEqual(report.sustainedGBps, BenchMath.median(report.measuredGBps))
    }

    func testTriadRunProducesReportCountingReadsAndWrites() throws {
        let bench = try makeBenchOrSkip()
        let n = 4096
        let report = try bench.run(
            kernel: .triad, elementCount: n, warmupIterations: 0, measuredIterations: 2)
        XCTAssertEqual(report.kernel, .triad)
        XCTAssertEqual(report.bytesCountedPerIteration, 3 * n * 4)
        XCTAssertEqual(report.bytesWrittenPerIteration, n * 4)
        XCTAssertTrue(report.writesCounted)
        XCTAssertEqual(report.gateSampleCount, 3)
        XCTAssertLessThanOrEqual(report.gateMaxAbsDelta, BandwidthMicrobench.tolerance)
        XCTAssertEqual(report.measuredGBps.count, 2)
    }

    func testDefaultIterationsArePinnedProtocol() throws {
        let bench = try makeBenchOrSkip()
        // Small N keeps this fast; the iteration defaults are the pinned 2 + 10.
        let report = try bench.run(kernel: .readOnly, elementCount: 4096)
        XCTAssertEqual(report.warmupCount, 2)
        XCTAssertEqual(report.measuredTimings.count, 10)
        XCTAssertEqual(BandwidthMicrobench.defaultElementCount,
                       ReadOnlyBandwidthKernel.benchmarkElementCount)
    }

    func testInvalidElementCountSurfacesAsKernelInputError() throws {
        let bench = try makeBenchOrSkip()
        XCTAssertThrowsError(try bench.run(
            kernel: .readOnly, elementCount: 4100, warmupIterations: 0, measuredIterations: 1
        )) { error in
            XCTAssertEqual(error as? KernelInputError,
                           .elementCountNotMultiple(of: 256, count: 4100))
        }
    }

    // MARK: - Diagnostic geometry is labeled, never silently a row

    func testNonPinnedDepthIsFlaggedDiagnosticInReportAndExport() throws {
        let bench = try makeBenchOrSkip()
        let report = try bench.run(
            kernel: .readOnly, elementCount: 300 * 64, warmupIterations: 0,
            measuredIterations: 1, float4sPerThread: 16)
        XCTAssertEqual(report.float4sPerThread, 16)
        XCTAssertTrue(report.isDiagnosticGeometry)
        let text = report.exportText(dateStamp: "d", deviceLabel: "m", osVersion: "o")
        XCTAssertTrue(text.contains("DIAGNOSTIC — non-pinned geometry (16 float4/thread/buffer; pin is 64) — NEVER a row"))
        XCTAssertTrue(text.contains("geometry: 16 float4/thread/buffer"))
    }

    func testPinnedDepthIsNotDiagnosticAndTriadCarriesNoDepth() throws {
        let bench = try makeBenchOrSkip()
        let readOnly = try bench.run(
            kernel: .readOnly, elementCount: 4096, warmupIterations: 0, measuredIterations: 1)
        XCTAssertEqual(readOnly.float4sPerThread, 64)
        XCTAssertFalse(readOnly.isDiagnosticGeometry)
        XCTAssertFalse(readOnly.exportText(dateStamp: "d", deviceLabel: "m", osVersion: "o")
            .contains("DIAGNOSTIC"))
        let triad = try bench.run(
            kernel: .triad, elementCount: 4096, warmupIterations: 0, measuredIterations: 1)
        XCTAssertNil(triad.float4sPerThread)
        XCTAssertFalse(triad.isDiagnosticGeometry)
    }

    // MARK: - Export text (the results.md row surface)

    func testExportTextCarriesEveryRowField() throws {
        let bench = try makeBenchOrSkip()
        let report = try bench.run(
            kernel: .readOnly, elementCount: 300 * 256, warmupIterations: 1, measuredIterations: 3)
        let text = report.exportText(
            dateStamp: "2026-09-26", deviceLabel: "iPhone16,1", osVersion: "iOS 26.7",
            batteryHealthNote: "100%", coldOrWarmNote: "warm")
        for needle in [
            "read-only streaming reduce (BW-1)",
            "date: 2026-09-26",
            "device: iPhone16,1 (iOS 26.7)",
            "battery health: 100%",
            "cold/warm: warm",
            "1 warmup discarded + 3 measured",
            "bytes read/iteration: 921600",
            "bytes written/iteration: 256",
            "excluded from GB/s",
            "correctness: 3 sampled partials within pre-committed gate",
            "iter  1:",
            "sustained (median):",
            "spread: min",
        ] {
            XCTAssertTrue(text.contains(needle), "export missing '\(needle)':\n\(text)")
        }
    }

    func testExportTextOmitsEmptyOperatorLinesAndLabelsTriad() throws {
        let bench = try makeBenchOrSkip()
        let report = try bench.run(
            kernel: .triad, elementCount: 4096, warmupIterations: 0, measuredIterations: 1)
        let text = report.exportText(
            dateStamp: "2026-09-26", deviceLabel: "Mac", osVersion: "macOS 26.5.1")
        XCTAssertTrue(text.contains("STREAM triad (P0B-4)"))
        XCTAssertTrue(text.contains("bytes moved/iteration (2R+1W): 49152"))
        XCTAssertTrue(text.contains("correctness: 3 sampled elements within pre-committed gate"))
        XCTAssertFalse(text.contains("battery health:"))
        XCTAssertFalse(text.contains("cold/warm:"))
    }
}
