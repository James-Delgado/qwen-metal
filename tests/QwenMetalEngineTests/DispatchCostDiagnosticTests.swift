import XCTest
@testable import QwenMetalEngine
import Metal

/// P4-6 part 1 (phase-4.md 2026-09-12 addendum): the MEASURE-FIRST diagnosis
/// of the ~105 µs/dispatch elementwise anomaly the P4-5 on-device attribution
/// surfaced (norm+elementwise 8.86 ms over 84 dispatches — launch/latency-
/// bound, not byte-bound; the Mac rows show the same signature, which is what
/// makes a Mac-side diagnosis meaningful).
///
/// The harness times ONE command buffer holding `count` back-to-back tiny
/// dispatches (residual-add / rmsnorm at the pinned model's hidden size)
/// under controlled structural variations, so the per-dispatch GPU cost can
/// be attributed to a cause:
///
/// - **dependent vs independent**: consecutive dispatches chained through a
///   read-after-write hazard (the production decode shape) vs disjoint
///   buffer pairs. A large gap means the cost is hazard-barrier
///   serialization (pipeline drain), not raw launch overhead.
/// - **tracked vs untracked**: the same dependent chain on buffers allocated
///   with `.hazardTrackingModeUntracked` (timing-only — outputs may be
///   garbage without the barriers). A large gap would mean Metal's automatic
///   tracking inserts coarser barriers than the dependency needs, i.e. a
///   config-level fix could pay.
/// - **serial vs concurrent encoder**: independent dispatches through a
///   `.concurrent` dispatch-type encoder — the ceiling for overlap that
///   encoder-level configuration alone can buy.
/// - **size scaling**: the dependent chain at dim 2048 vs dim 64 — a
///   size-independent cost is latency, not bandwidth.
///
/// Dual timing rides along per hard rule 7 (wall brackets the whole batch;
/// GPU time from the command buffer's own timestamps). The full sweep is
/// opt-in via QWEN_DISPATCH_DIAG=1 (FreeRunReport precedent) and its numbers
/// are Mac PROVISIONAL diagnosis inputs, never benchmark rows; findings are
/// recorded in DECISIONS.md (P4-6 entry). The sanity test below always runs.
final class DispatchCostDiagnosticTests: XCTestCase {

    // MARK: - Harness

    private func makeContextOrSkip() throws -> MetalContext {
        do {
            return try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    private func makeHalfBuffer(
        _ device: MTLDevice, count: Int, fill: Float16,
        options: MTLResourceOptions = .storageModeShared
    ) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: count * 2, options: options)
        else {
            throw MetalHarnessError.bufferAllocationFailed(length: count * 2)
        }
        buffer.contents().withMemoryRebound(to: Float16.self, capacity: count) {
            for i in 0..<count { $0[i] = fill }
        }
        return buffer
    }

    /// bf16 bit pattern of an EXACTLY representable fp32 value.
    private func bf16Bits(_ v: Float) -> UInt16 {
        let bits = UInt16(truncatingIfNeeded: v.bitPattern >> 16)
        precondition(Float(bitPattern: UInt32(bits) << 16) == v,
                     "test value \(v) is not bf16-exact")
        return bits
    }

    /// Encodes dispatches into ONE command buffer through an encoder of the
    /// given dispatch type and runs it to completion — the `timedDispatch`
    /// contract with the dispatch type exposed (the diagnosis needs
    /// `.concurrent`; production stays `.serial`).
    @discardableResult
    private func runBatch(
        context: MetalContext, dispatchType: MTLDispatchType,
        encode: (MTLComputeCommandEncoder) throws -> Void
    ) throws -> DispatchTiming {
        let wallStart = CACurrentMediaTime()
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw MetalHarnessError.commandBufferCreationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder(
            dispatchType: dispatchType) else {
            throw MetalHarnessError.encoderCreationFailed
        }
        do {
            try encode(encoder)
        } catch {
            encoder.endEncoding()
            throw error
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let wallEnd = CACurrentMediaTime()
        guard commandBuffer.status == .completed else {
            throw MetalHarnessError.gpuExecutionFailed(
                status: commandBuffer.status, underlying: commandBuffer.error)
        }
        return DispatchTiming(
            wallStart: wallStart, wallEnd: wallEnd,
            gpuStart: commandBuffer.gpuStartTime,
            gpuEnd: commandBuffer.gpuEndTime)
    }

    private struct Experiment {
        let label: String
        let dispatches: Int
        /// Median over the measured iterations.
        let gpuPerDispatchMicros: Double
        let gpuMillis: Double
        let wallMillis: Double
    }

    /// Runs one configuration `warmup + iterations` times and reports the
    /// median GPU time (dual timing checked every run).
    private func measure(
        label: String, context: MetalContext, dispatches: Int,
        dispatchType: MTLDispatchType = .serial,
        warmup: Int = 2, iterations: Int = 9,
        encode: (MTLComputeCommandEncoder) throws -> Void
    ) throws -> Experiment {
        var gpuSeconds: [Double] = []
        var wallSeconds: [Double] = []
        for i in 0..<(warmup + iterations) {
            let timing = try runBatch(
                context: context, dispatchType: dispatchType, encode: encode)
            XCTAssertGreaterThan(timing.gpuDuration, 0, "\(label): GPU time")
            XCTAssertGreaterThanOrEqual(
                timing.wallDuration, timing.gpuDuration,
                "\(label): wall must bracket GPU (hard rule 7)")
            if i >= warmup {
                gpuSeconds.append(timing.gpuDuration)
                wallSeconds.append(timing.wallDuration)
            }
        }
        let gpuMedian = gpuSeconds.sorted()[gpuSeconds.count / 2]
        let wallMedian = wallSeconds.sorted()[wallSeconds.count / 2]
        return Experiment(
            label: label, dispatches: dispatches,
            gpuPerDispatchMicros: gpuMedian / Double(dispatches) * 1e6,
            gpuMillis: gpuMedian * 1e3, wallMillis: wallMedian * 1e3)
    }

    // MARK: - Configurations (all tiny elementwise at the pinned hidden size)

    /// Dependent chain: out = a + b ping-ponging through the SAME two
    /// buffers, so every dispatch has a read-after-write hazard on its
    /// predecessor — the production decode shape.
    private func dependentChain(
        context: MetalContext, kernels: DecodeKernels, dim: Int, count: Int,
        options: MTLResourceOptions, label: String
    ) throws -> Experiment {
        let a = try makeHalfBuffer(context.device, count: dim, fill: 0.25,
                                   options: options)
        let b = try makeHalfBuffer(context.device, count: dim, fill: 0.5,
                                   options: options)
        return try measure(label: label, context: context, dispatches: count) { encoder in
            for i in 0..<count {
                try kernels.encodeResidualAdd(
                    into: encoder, a: i % 2 == 0 ? a : b, b: i % 2 == 0 ? b : a,
                    count: dim, output: i % 2 == 0 ? b : a)
            }
        }
    }

    /// Independent dispatches: disjoint buffer triples, no hazards between
    /// consecutive dispatches.
    private func independent(
        context: MetalContext, kernels: DecodeKernels, dim: Int, count: Int,
        dispatchType: MTLDispatchType, label: String
    ) throws -> Experiment {
        var triples: [(MTLBuffer, MTLBuffer, MTLBuffer)] = []
        for _ in 0..<count {
            triples.append((
                try makeHalfBuffer(context.device, count: dim, fill: 0.25),
                try makeHalfBuffer(context.device, count: dim, fill: 0.5),
                try makeHalfBuffer(context.device, count: dim, fill: 0)))
        }
        return try measure(
            label: label, context: context, dispatches: count,
            dispatchType: dispatchType
        ) { encoder in
            for (a, b, out) in triples {
                try kernels.encodeResidualAdd(
                    into: encoder, a: a, b: b, count: dim, output: out)
            }
        }
    }

    /// Dependent rmsnorm chain — the actual norm kernel the fused layer
    /// dispatches, chained A→B→A. `rows: 1, dim: 2048` is the block-norm
    /// shape; `rows: 16, dim: 128` is the qk-norm shape (same element
    /// count, different redundancy/occupancy profile).
    private func dependentNormChain(
        context: MetalContext, kernels: DecodeKernels, rows: Int, dim: Int,
        count: Int, label: String
    ) throws -> Experiment {
        let a = try makeHalfBuffer(context.device, count: rows * dim, fill: 0.25)
        let b = try makeHalfBuffer(context.device, count: rows * dim, fill: 0.5)
        let weightBits = [UInt16](repeating: bf16Bits(1.0), count: dim)
        guard let weight = weightBits.withUnsafeBytes({
            context.device.makeBuffer(
                bytes: $0.baseAddress!, length: $0.count,
                options: .storageModeShared)
        }) else {
            throw MetalHarnessError.bufferAllocationFailed(length: dim * 2)
        }
        return try measure(label: label, context: context, dispatches: count) { encoder in
            for i in 0..<count {
                try kernels.encodeRMSNorm(
                    into: encoder, input: i % 2 == 0 ? a : b, weight: weight,
                    weightByteOffset: 0, rows: rows, dim: dim, eps: 1e-6,
                    output: i % 2 == 0 ? b : a)
            }
        }
    }

    // MARK: - Always-run sanity (the harness itself is tested)

    /// The harness produces sane dual timing on both encoder types and a
    /// positive per-dispatch cost — cheap, runs in every suite pass.
    func testDiagnosticHarnessSanity() throws {
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        let dependent = try dependentChain(
            context: context, kernels: kernels, dim: 256, count: 8,
            options: .storageModeShared, label: "sanity dependent")
        XCTAssertGreaterThan(dependent.gpuPerDispatchMicros, 0)
        XCTAssertGreaterThanOrEqual(dependent.wallMillis, dependent.gpuMillis)
        let concurrent = try independent(
            context: context, kernels: kernels, dim: 256, count: 8,
            dispatchType: .concurrent, label: "sanity concurrent")
        XCTAssertGreaterThan(concurrent.gpuPerDispatchMicros, 0)
    }

    // MARK: - The opt-in diagnosis sweep

    /// Set QWEN_DISPATCH_DIAG=1 to run. Prints the comparison table the
    /// DECISIONS.md P4-6 diagnosis entry records; asserts only sanity (the
    /// numbers are findings, not gates).
    func testDispatchCostDiagnosisSweep() throws {
        guard ProcessInfo.processInfo.environment["QWEN_DISPATCH_DIAG"] == "1" else {
            throw XCTSkip(
                "dispatch-cost diagnosis is opt-in: set QWEN_DISPATCH_DIAG=1 "
                + "(Mac PROVISIONAL diagnosis inputs, never benchmark rows)")
        }
        let context = try makeContextOrSkip()
        let kernels = try DecodeKernels(context: context)
        // 84 = the on-device elementwise class count (3/layer × 28) whose
        // 8.86 ms / ~105 µs-per-dispatch signature this diagnosis explains.
        let count = 84
        let dim = 2048

        var rows: [Experiment] = []
        rows.append(try dependentChain(
            context: context, kernels: kernels, dim: dim, count: count,
            options: .storageModeShared,
            label: "residual dependent tracked serial dim\(dim)"))
        rows.append(try dependentChain(
            context: context, kernels: kernels, dim: dim, count: count,
            options: [.storageModeShared, .hazardTrackingModeUntracked],
            label: "residual dependent UNTRACKED serial dim\(dim)"))
        rows.append(try independent(
            context: context, kernels: kernels, dim: dim, count: count,
            dispatchType: .serial,
            label: "residual independent tracked serial dim\(dim)"))
        rows.append(try independent(
            context: context, kernels: kernels, dim: dim, count: count,
            dispatchType: .concurrent,
            label: "residual independent tracked CONCURRENT dim\(dim)"))
        rows.append(try dependentChain(
            context: context, kernels: kernels, dim: 64, count: count,
            options: .storageModeShared,
            label: "residual dependent tracked serial dim64"))
        rows.append(try dependentNormChain(
            context: context, kernels: kernels, rows: 1, dim: dim, count: count,
            label: "rmsnorm  dependent BLOCK shape rows1 dim\(dim)"))
        rows.append(try dependentNormChain(
            context: context, kernels: kernels, rows: 16, dim: 128, count: count,
            label: "rmsnorm  dependent QKNORM shape rows16 dim128"))

        print("=== P4-6 dispatch-cost diagnosis (Mac PROVISIONAL, \(count) dispatches/buffer, median of 9) ===")
        for r in rows {
            let perDispatch = String(format: "%8.2f µs/dispatch", r.gpuPerDispatchMicros)
            let gpu = String(format: "GPU %7.3f ms", r.gpuMillis)
            let wall = String(format: "wall %7.3f ms", r.wallMillis)
            print("  \(r.label.padding(toLength: 52, withPad: " ", startingAt: 0)) \(perDispatch)  \(gpu)  \(wall)")
        }
        for r in rows {
            XCTAssertGreaterThan(r.gpuPerDispatchMicros, 0, r.label)
        }
    }
}
