import XCTest
@testable import QwenMetalEngine
import Metal

/// P0B-1 tests: Metal harness (device/queue/library setup) + dual timing utility.
/// Timing sanity per docs/phases/phase-0-1.md: nonzero duration, start <= end,
/// wall >= GPU. The saxpy dispatch->readback->diff loop is P0B-2, not here — the
/// kernel below is a test-only trivial fill used to exercise the harness itself.
final class MetalHarnessTests: XCTestCase {

    /// Test-only kernel source. Lives in the test target on purpose: the engine
    /// ships no kernels until P0B-2.
    private static let fillKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void fill_value(device float *out    [[buffer(0)]],
                           constant float &value [[buffer(1)]],
                           uint gid [[thread_position_in_grid]]) {
        out[gid] = value;
    }
    """

    private func makeContextOrSkip() throws -> MetalContext {
        do {
            return try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    // MARK: - Device / queue / library setup (PRD 0b.1)

    func testContextProvidesDeviceAndQueue() throws {
        let context = try makeContextOrSkip()
        XCTAssertFalse(context.device.name.isEmpty)
        // The queue must belong to the same device the context exposes.
        XCTAssertTrue(context.queue.device === context.device)
    }

    func testCompilesLibraryAndPipelineFromSource() throws {
        let context = try makeContextOrSkip()
        let library = try context.makeLibrary(source: Self.fillKernelSource)
        let pipeline = try context.makeComputePipeline(library: library, function: "fill_value")
        XCTAssertGreaterThan(pipeline.maxTotalThreadsPerThreadgroup, 0)
    }

    func testMalformedKernelSourceThrowsCompileError() throws {
        let context = try makeContextOrSkip()
        XCTAssertThrowsError(try context.makeLibrary(source: "kernel void broken(")) { error in
            guard case MetalHarnessError.libraryCompileFailed = error else {
                return XCTFail("Expected libraryCompileFailed, got \(error)")
            }
        }
    }

    func testMissingFunctionNameThrows() throws {
        let context = try makeContextOrSkip()
        let library = try context.makeLibrary(source: Self.fillKernelSource)
        XCTAssertThrowsError(
            try context.makeComputePipeline(library: library, function: "no_such_kernel")
        ) { error in
            guard case MetalHarnessError.functionNotFound(let name) = error else {
                return XCTFail("Expected functionNotFound, got \(error)")
            }
            XCTAssertEqual(name, "no_such_kernel")
        }
    }

    // MARK: - Dual timing utility (PRD 0b.2, hard rule 7)

    func testTimedDispatchRecordsSaneDualTiming() throws {
        let context = try makeContextOrSkip()
        let library = try context.makeLibrary(source: Self.fillKernelSource)
        let pipeline = try context.makeComputePipeline(library: library, function: "fill_value")

        // Large enough that GPU execution time is reliably nonzero.
        let count = 1 << 24 // 16M floats, 64 MB
        guard let buffer = context.device.makeBuffer(
            length: count * MemoryLayout<Float>.stride, options: .storageModeShared
        ) else {
            return XCTFail("Failed to allocate test buffer")
        }
        var fillValue: Float = 3.5

        let timing = try context.timedDispatch { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            encoder.setBytes(&fillValue, length: MemoryLayout<Float>.stride, index: 1)
            let threadsPerGroup = MTLSize(
                width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1
            )
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: threadsPerGroup
            )
        }

        // Spec sanity assertions: nonzero duration, start <= end, wall >= GPU.
        XCTAssertGreaterThan(timing.gpuDuration, 0, "GPU duration must be nonzero")
        XCTAssertGreaterThan(timing.wallDuration, 0, "Wall duration must be nonzero")
        XCTAssertLessThanOrEqual(timing.gpuStart, timing.gpuEnd)
        XCTAssertLessThanOrEqual(timing.wallStart, timing.wallEnd)
        XCTAssertGreaterThanOrEqual(
            timing.wallDuration, timing.gpuDuration,
            "Wall clock brackets encode/commit/schedule; it can never be under GPU time"
        )
        XCTAssertGreaterThanOrEqual(timing.dispatchOverhead, 0)
        XCTAssertEqual(
            timing.dispatchOverhead, timing.wallDuration - timing.gpuDuration,
            accuracy: 1e-12
        )

        // The dispatch actually ran: spot-check the fill result.
        let contents = buffer.contents().bindMemory(to: Float.self, capacity: count)
        XCTAssertEqual(contents[0], 3.5)
        XCTAssertEqual(contents[count / 2], 3.5)
        XCTAssertEqual(contents[count - 1], 3.5)
    }

    func testTimedDispatchPropagatesEncodeClosureError() throws {
        let context = try makeContextOrSkip()
        struct EncodeFailure: Error {}
        XCTAssertThrowsError(try context.timedDispatch { _ in throw EncodeFailure() }) { error in
            XCTAssertTrue(error is EncodeFailure)
        }
    }
}
