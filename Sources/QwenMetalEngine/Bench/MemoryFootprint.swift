import Darwin

/// P2-6: in-app phys_footprint reading for the benchmark row export.
/// CROSS-CHECK ONLY — the protocol pin (PLAN.md) makes the Xcode memory
/// gauge the one memory metric of record; this exists so a row export
/// carries a same-moment reading next to the gauge value James records.
public enum MemoryFootprint {
    /// The current process's phys_footprint in bytes (the same accounting
    /// the Xcode gauge and the jetsam limit use), or nil if the Mach call
    /// fails.
    public static func currentPhysFootprintBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return UInt64(info.phys_footprint)
    }
}
