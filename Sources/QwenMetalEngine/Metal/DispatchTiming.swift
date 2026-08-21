import Foundation

/// Timing record for one command-buffer batch.
///
/// Hard rule 7 (PLAN.md invariant 5): every timing measurement carries BOTH
/// GPU timestamps and wall clock — GPU-only timing is forbidden. The
/// wall − GPU delta is the CPU encode/commit/scheduling overhead metric that
/// Phase 4's dispatch-reduction exit criterion consumes.
///
/// All fields are host-time seconds in the mach monotonic time domain:
/// `gpuStart`/`gpuEnd` come from `MTLCommandBuffer.gpuStartTime`/`.gpuEndTime`,
/// and the wall fields are sampled from the same clock, so the values are
/// directly comparable.
public struct DispatchTiming: Sendable {
    /// Sampled immediately before command-buffer creation/encoding began.
    public let wallStart: TimeInterval
    /// Sampled immediately after `waitUntilCompleted` returned.
    public let wallEnd: TimeInterval
    /// Host time at which the GPU began executing the command buffer.
    public let gpuStart: TimeInterval
    /// Host time at which the GPU finished executing the command buffer.
    public let gpuEnd: TimeInterval

    public init(wallStart: TimeInterval, wallEnd: TimeInterval,
                gpuStart: TimeInterval, gpuEnd: TimeInterval) {
        self.wallStart = wallStart
        self.wallEnd = wallEnd
        self.gpuStart = gpuStart
        self.gpuEnd = gpuEnd
    }

    /// End-to-end batch time as the CPU experienced it: encode + commit +
    /// scheduling + GPU execution.
    public var wallDuration: TimeInterval { wallEnd - wallStart }

    /// Time the GPU spent executing the command buffer.
    public var gpuDuration: TimeInterval { gpuEnd - gpuStart }

    /// CPU-side overhead of the batch (encode/commit/schedule): wall − GPU.
    public var dispatchOverhead: TimeInterval { wallDuration - gpuDuration }
}
