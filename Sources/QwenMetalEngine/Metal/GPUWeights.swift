import Foundation
import Metal

/// Where the checkpoint bytes the GPU reads actually live (phase-2.md D1 and
/// the OV#9 residency comparison). Both modes expose byte-identical contents;
/// the difference is purely file-backed vs dirty memory, which is what the
/// Phase 2 on-device sustained-stability benchmark isolates.
public enum WeightsResidency: String, CaseIterable, Sendable {
    /// Mode A: the mmapped checkpoint wrapped in a no-copy buffer. File-backed
    /// clean pages — the OS may evict and re-fault them under pressure.
    case mmap
    /// Mode B: the same bytes copied into an ordinary heap-backed buffer.
    /// Dirty memory — never evicted, counts fully against phys_footprint.
    case wiredCopy
}

/// Which weight encoding the GPU pipeline consumes (P3-5): the Phase 2 bf16
/// checkpoint (register bit-shift upcast) or the Phase 3 q4g64 packed
/// triplets (register dequant — hard rule 1 either way). Both ride the same
/// whole-file `GPUWeights` buffer; the format decides which kernels bind it.
public enum WeightsFormat: String, CaseIterable, Sendable {
    case bf16
    case q4g64
}

/// GPU residency for the whole checkpoint (P2-1, spec D1): ONE `MTLBuffer`
/// over the raw safetensors bytes, plus per-tensor byte offsets from the
/// parsed header index. Kernels read weights as 16-bit words at
/// `byteOffset(for:)` and upcast in registers — no per-tensor buffers, no
/// conversion at load, nothing materialized (hard rule 1, invariant 5).
///
/// The weight bits are identical to what the CPU reference upcasts, so every
/// GPU-vs-CPU diff test sees zero weight-rounding divergence, and the two
/// residency modes must produce bit-identical results (pre-committed exact
/// gate, DECISIONS.md 2026-08-23 Phase 2 gates entry).
public final class GPUWeights {
    public let residency: WeightsResidency
    /// Covers the entire file (header included): tensor offsets returned by
    /// `byteOffset(for:)` are absolute file offsets, valid in both modes.
    public let buffer: MTLBuffer
    /// Retained for header-index lookups; on the mmap path the buffer's
    /// deallocator additionally retains the file, so the mapping stays valid
    /// even for a caller holding only `buffer`.
    private let file: SafetensorsFile

    public init(
        file: SafetensorsFile, context: MetalContext, residency: WeightsResidency
    ) throws {
        self.file = file
        self.residency = residency
        let fileSize = file.mappedFileSize

        switch residency {
        case .mmap:
            // bytesNoCopy requires a page-aligned base (mmap guarantees it)
            // and a page-multiple length. mmap maps whole pages, so rounding
            // the file size up stays inside the mapping; the bytes past EOF
            // in the last page are zero-fill and no tensor reaches them
            // (the parser bounds every tensor to the data section).
            let pageSize = Int(getpagesize())
            let roundedLength = (fileSize + pageSize - 1) / pageSize * pageSize
            guard let buffer = context.device.makeBuffer(
                bytesNoCopy: file.mappedBaseAddress,
                length: roundedLength,
                options: .storageModeShared,
                // The closure retains `file` until Metal releases the buffer:
                // the munmap (SafetensorsFile.deinit) cannot run while any
                // holder of this buffer is alive, `GPUWeights` or not.
                deallocator: { _, _ in _ = file }
            ) else {
                throw MetalHarnessError.bufferAllocationFailed(length: roundedLength)
            }
            self.buffer = buffer
        case .wiredCopy:
            guard let buffer = context.device.makeBuffer(
                length: fileSize, options: .storageModeShared
            ) else {
                throw MetalHarnessError.bufferAllocationFailed(length: fileSize)
            }
            // The copy is the point: touching every byte makes the pages
            // dirty anonymous memory instead of clean file-backed cache.
            memcpy(buffer.contents(), file.mappedBaseAddress, fileSize)
            self.buffer = buffer
        }
    }

    /// Header index passthrough (dtype/shape/element count for kernel setup).
    public func info(for name: String) throws -> TensorInfo {
        try file.info(for: name)
    }

    /// Absolute byte offset of the tensor's payload inside `buffer`. Passed
    /// to kernels as an argument — NOT as a `setBuffer` offset, whose
    /// alignment requirements the format's 2-byte-packed offsets can violate.
    public func byteOffset(for name: String) throws -> Int {
        file.dataSectionStart + (try file.info(for: name)).dataOffset
    }
}
