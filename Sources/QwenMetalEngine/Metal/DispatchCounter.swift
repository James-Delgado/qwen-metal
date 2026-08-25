/// Counts compute dispatches as they are encoded (P2-5, spec D5).
///
/// The count is taken at the exact `dispatchThreads` call sites inside
/// `DecodeKernels`/`AttentionKernels` — measured, never derived from the
/// pipeline structure (METHODOLOGY rule 4) — so a future kernel addition or
/// fusion (Phase 4's dispatch-reduction work) changes the reported number
/// automatically. `GPUModel` resets it at the start of every `step` and
/// publishes the result as `lastStepDispatchCount`.
public final class DispatchCounter {
    public private(set) var count: Int = 0

    public init() {}

    func increment() {
        count += 1
    }

    public func reset() {
        count = 0
    }
}
