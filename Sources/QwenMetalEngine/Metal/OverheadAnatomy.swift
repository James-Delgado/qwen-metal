import Foundation
import Metal
import QuartzCore

/// P4-9 (phase-4.md 2026-09-12 addendum): fine-grained anatomy of one
/// command buffer's wall−GPU overhead — the diagnostic record that splits
/// the fixed per-token cost the P4-5 affine fit surfaced (≈1.17 ms/token
/// on-device, ≈1.5 µs/dispatch slope) into its structural components:
///
///     wallStart ─[encode]─ wallEncoded ─[commit call]─ wallCommitted
///       ─[commit→GPU-start]─ gpuStart ─[GPU execution]─ gpuEnd
///       ─[completion wakeup]─ wallEnd
///
/// All fields are host-time seconds in the mach monotonic domain (the
/// `DispatchTiming` contract): wall samples from `CACurrentMediaTime()`,
/// GPU and scheduling timestamps from the command buffer itself, so every
/// span is directly subtractable, and the four non-GPU spans telescope to
/// wall − GPU exactly by construction.
///
/// DIAGNOSTIC ONLY (P4-1 invariance precedent): the production decode path
/// never produces this record and its numbers are never benchmark rows —
/// the production overhead metric remains `DispatchTiming.dispatchOverhead`.
public struct OverheadAnatomy: Sendable {
    /// Sampled immediately before command-buffer creation.
    public let wallStart: TimeInterval
    /// Sampled immediately after `endEncoding()` returned.
    public let wallEncoded: TimeInterval
    /// Sampled immediately after `commit()` returned.
    public let wallCommitted: TimeInterval
    /// `MTLCommandBuffer.kernelStartTime` — start of the driver's
    /// scheduling ("kernel") stage, inside the commit→GPU-start span.
    public let kernelStart: TimeInterval
    /// `MTLCommandBuffer.kernelEndTime` — end of the scheduling stage.
    public let kernelEnd: TimeInterval
    /// Host time at which the GPU began executing the command buffer.
    public let gpuStart: TimeInterval
    /// Host time at which the GPU finished executing the command buffer.
    public let gpuEnd: TimeInterval
    /// Sampled immediately after `waitUntilCompleted()` returned.
    public let wallEnd: TimeInterval

    public init(wallStart: TimeInterval, wallEncoded: TimeInterval,
                wallCommitted: TimeInterval, kernelStart: TimeInterval,
                kernelEnd: TimeInterval, gpuStart: TimeInterval,
                gpuEnd: TimeInterval, wallEnd: TimeInterval) {
        self.wallStart = wallStart
        self.wallEncoded = wallEncoded
        self.wallCommitted = wallCommitted
        self.kernelStart = kernelStart
        self.kernelEnd = kernelEnd
        self.gpuStart = gpuStart
        self.gpuEnd = gpuEnd
        self.wallEnd = wallEnd
    }

    /// CPU-side command-buffer creation + kernel encoding.
    public var encodeSeconds: TimeInterval { wallEncoded - wallStart }
    /// The `commit()` call itself.
    public var commitSeconds: TimeInterval { wallCommitted - wallEncoded }
    /// Scheduling latency: `commit()` returned → GPU began executing.
    public var commitToGPUStartSeconds: TimeInterval { gpuStart - wallCommitted }
    /// GPU execution time (the `DispatchTiming.gpuDuration` analogue).
    public var gpuSeconds: TimeInterval { gpuEnd - gpuStart }
    /// Completion wakeup: GPU finished → `waitUntilCompleted()` returned
    /// on the calling thread.
    public var wakeupSeconds: TimeInterval { wallEnd - gpuEnd }
    /// End-to-end wall time (the `DispatchTiming.wallDuration` analogue).
    public var wallSeconds: TimeInterval { wallEnd - wallStart }
    /// wall − GPU — the metric the ≤1.2 ms Phase 4 gate consumes. Identical
    /// (up to float rounding) to encode + commit + commit→GPU-start + wakeup.
    public var overheadSeconds: TimeInterval { wallSeconds - gpuSeconds }
    /// Duration of the driver's scheduling stage (subdivides the
    /// commit→GPU-start span; the stage can overlap the commit call).
    public var scheduleStageSeconds: TimeInterval { kernelEnd - kernelStart }

    /// The production-shaped dual-timing view (hard rule 7 field set).
    public var timing: DispatchTiming {
        DispatchTiming(wallStart: wallStart, wallEnd: wallEnd,
                       gpuStart: gpuStart, gpuEnd: gpuEnd)
    }
}

extension MetalContext {
    /// `timedDispatch`'s diagnostic twin: identical structure (create →
    /// encode → endEncoding → commit → waitUntilCompleted) with wall
    /// samples at the encode/commit boundaries and the command buffer's
    /// scheduling-stage timestamps captured — the P4-9 overhead-anatomy
    /// probe. Never called by the production decode path.
    ///
    /// `unretainedReferences: true` is the P4-9 "cheap submission
    /// experiment" variant: a command buffer that skips per-encoder
    /// resource retention (`makeCommandBufferWithUnretainedReferences`).
    /// Safe only when every referenced resource outlives the buffer —
    /// true for the engine's model-owned buffers — and used only to
    /// measure what retention costs, never as a production setting.
    @discardableResult
    public func anatomyDispatch(
        unretainedReferences: Bool = false,
        _ encode: (MTLComputeCommandEncoder) throws -> Void
    ) throws -> OverheadAnatomy {
        let wallStart = CACurrentMediaTime()
        let maybeBuffer = unretainedReferences
            ? queue.makeCommandBufferWithUnretainedReferences()
            : queue.makeCommandBuffer()
        guard let commandBuffer = maybeBuffer else {
            throw MetalHarnessError.commandBufferCreationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalHarnessError.encoderCreationFailed
        }
        do {
            try encode(encoder)
        } catch {
            encoder.endEncoding()
            throw error
        }
        encoder.endEncoding()
        let wallEncoded = CACurrentMediaTime()
        commandBuffer.commit()
        let wallCommitted = CACurrentMediaTime()
        commandBuffer.waitUntilCompleted()
        let wallEnd = CACurrentMediaTime()

        guard commandBuffer.status == .completed else {
            throw MetalHarnessError.gpuExecutionFailed(
                status: commandBuffer.status, underlying: commandBuffer.error)
        }
        return OverheadAnatomy(
            wallStart: wallStart,
            wallEncoded: wallEncoded,
            wallCommitted: wallCommitted,
            kernelStart: commandBuffer.kernelStartTime,
            kernelEnd: commandBuffer.kernelEndTime,
            gpuStart: commandBuffer.gpuStartTime,
            gpuEnd: commandBuffer.gpuEndTime,
            wallEnd: wallEnd)
    }
}
