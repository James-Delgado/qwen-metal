import Foundation

/// P4-1 (phase-4.md D1): the kernel classes the diagnostic attribution mode
/// reports GPU time against. The class of every dispatch is declared at its
/// encode call site in `GPUModel.encodeForward`, so a Phase 4 fusion that
/// restructures the pipeline re-classifies automatically.
public enum KernelClass: String, CaseIterable, Sendable {
    /// Weight-matrix matvecs in the layer stack (QKV, o_proj, gate/up/down).
    /// bf16 or fused-dequant q4 — the class is about pipeline role, not
    /// weight format (the report records the format separately).
    case matvec
    /// The attention chain over the KV cache (scores, softmax, PV — later
    /// the fused SDPA).
    case attention
    /// RMSNorm / RoPE / SwiGLU / residual / KV-append elementwise work.
    case normElementwise = "norm+elementwise"
    /// Embedding lookup, final norm, and the lm_head projection.
    case headTail = "head/tail"
    /// PF-1: the batched prefill projections through the P5-2 tiled
    /// dequant-GEMM (q/k/v, o, gate/up, down at M = chunk positions).
    /// Labeled distinctly from `matvec` so a prefill breakdown is never
    /// read as a decode one; always zero in decode attributions.
    case gemm
}

/// One diagnostic decode token's per-class GPU time, from per-class
/// command-buffer splits (spec D1). DIAGNOSTIC ONLY: produced by
/// `GPUModel.attributedStep`, never by the production one-command-buffer
/// path, and never a benchmark row.
public struct TokenAttribution: Sendable {
    /// One class-contiguous run of dispatches — one command buffer.
    /// GPU timestamps are the buffer's own (mach host-time seconds, the
    /// `DispatchTiming` domain).
    public struct Segment: Sendable {
        public let kernelClass: KernelClass
        public let gpuStart: TimeInterval
        public let gpuEnd: TimeInterval
        /// Dispatches encoded into this segment, measured by
        /// `DispatchCounter` snapshots at the segment boundaries.
        public let dispatchCount: Int

        public init(
            kernelClass: KernelClass, gpuStart: TimeInterval,
            gpuEnd: TimeInterval, dispatchCount: Int
        ) {
            self.kernelClass = kernelClass
            self.gpuStart = gpuStart
            self.gpuEnd = gpuEnd
            self.dispatchCount = dispatchCount
        }

        public var gpuSeconds: TimeInterval { gpuEnd - gpuStart }
    }

    /// Cache position the attributed token ran at (its attention depth).
    public let position: Int
    /// Wall time of the whole diagnostic step, sampled around the full
    /// encode → last-buffer-completed bracket (hard rule 7: wall rides
    /// along even in diagnostic mode; it bounds the GPU span from above).
    public let wallSeconds: Double
    /// Segments in encode order.
    public let segments: [Segment]

    public init(position: Int, wallSeconds: Double, segments: [Segment]) {
        self.position = position
        self.wallSeconds = wallSeconds
        self.segments = segments
    }

    /// Sum of GPU time over this class's segments (exact bookkeeping —
    /// the pre-committed sanity bounds test this against the segments).
    public func classGPUSeconds(_ kernelClass: KernelClass) -> Double {
        segments.lazy.filter { $0.kernelClass == kernelClass }
            .reduce(0) { $0 + $1.gpuSeconds }
    }

    public func classDispatchCount(_ kernelClass: KernelClass) -> Int {
        segments.lazy.filter { $0.kernelClass == kernelClass }
            .reduce(0) { $0 + $1.dispatchCount }
    }

    /// Total attributed GPU time (sum over all segments).
    public var gpuSecondsTotal: Double {
        segments.reduce(0) { $0 + $1.gpuSeconds }
    }

    /// Total dispatches across all segments — must equal the production
    /// pipeline's measured count for the same step shape.
    public var dispatchCountTotal: Int {
        segments.reduce(0) { $0 + $1.dispatchCount }
    }

    /// First segment's gpuStart → last segment's gpuEnd: the token's whole
    /// GPU window including inter-buffer scheduling gaps.
    public var spanSeconds: Double {
        guard let first = segments.first, let last = segments.last else { return 0 }
        return last.gpuEnd - first.gpuStart
    }
}
