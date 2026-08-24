import Metal

/// Errors from the KV cache and the attention-kernel host wrappers. Every
/// validation failure is explicit and named (METHODOLOGY: clear errors at
/// system boundaries) — a bad slot index must never reach a GPU dispatch,
/// and a full cache must stop decode cleanly, never write out of bounds.
public enum KVCacheError: Error, CustomStringConvertible, Equatable {
    case nonPositiveDimension(name: String, value: Int)
    case byteCountOverflow(layers: Int, kvHeads: Int, maxContext: Int, headDim: Int)
    case allocationFailed(bytes: Int)
    case indexOutOfRange(name: String, value: Int, bound: Int)
    case contextFull(position: Int, maxContext: Int)
    case gqaMismatch(numHeads: Int, kvHeads: Int)
    case rowCountExceedsStride(count: Int, rowStride: Int)

    public var description: String {
        switch self {
        case .nonPositiveDimension(let name, let value):
            return "KV cache dimension \(name) must be positive, got \(value)"
        case .byteCountOverflow(let layers, let kvHeads, let maxContext, let headDim):
            return "KV cache byte size overflows Int for layers=\(layers) "
                + "kvHeads=\(kvHeads) maxContext=\(maxContext) headDim=\(headDim)"
        case .allocationFailed(let bytes):
            return "Failed to allocate the \(bytes)-byte KV cache buffer"
        case .indexOutOfRange(let name, let value, let bound):
            return "KV cache index \(name) = \(value) outside 0..<\(bound)"
        case .contextFull(let position, let maxContext):
            return "KV cache is full: append at position \(position) exceeds the "
                + "preallocated context limit \(maxContext) (the cache is never "
                + "grown — decode must stop at the context limit)"
        case .gqaMismatch(let numHeads, let kvHeads):
            return "GQA requires kvHeads \(kvHeads) to divide numHeads \(numHeads)"
        case .rowCountExceedsStride(let count, let rowStride):
            return "Softmax row count \(count) exceeds row stride \(rowStride)"
        }
    }
}

/// P2-3 (docs/phases/phase-2.md D3, hard rule 4): the decode KV cache — ONE
/// fp16 `MTLBuffer` of shape `[layers][K|V][kvHeads][maxContext][headDim]`,
/// head-major so each head's positions are contiguous for the attention
/// kernels' streaming reads. Allocated in full at model load and never grown;
/// at the pinned model's dims (28 layers, 8 KV heads, 4096 positions,
/// head dim 128) the buffer is exactly 448 MiB.
public final class KVCache {
    /// The K or V half of a layer's cache slab, in layout order.
    public enum Component: Int {
        case key = 0
        case value = 1
    }

    public let layers: Int
    public let kvHeads: Int
    public let maxContext: Int
    public let headDim: Int
    public let buffer: MTLBuffer

    public var byteCount: Int { buffer.length }

    /// Total cache size in bytes for the given dims: layers · 2 · kvHeads ·
    /// maxContext · headDim · 2 (fp16). Throws on non-positive dims or Int
    /// overflow — a bad config must never turn into a silently-wrong size.
    public static func byteCount(
        layers: Int, kvHeads: Int, maxContext: Int, headDim: Int
    ) throws -> Int {
        let dims = [
            ("layers", layers), ("kvHeads", kvHeads),
            ("maxContext", maxContext), ("headDim", headDim),
        ]
        for (name, value) in dims {
            guard value > 0 else {
                throw KVCacheError.nonPositiveDimension(name: name, value: value)
            }
        }
        var total = 2 * 2 // K|V pair × fp16 bytes
        for (_, value) in dims {
            let (product, overflow) = total.multipliedReportingOverflow(by: value)
            guard !overflow else {
                throw KVCacheError.byteCountOverflow(
                    layers: layers, kvHeads: kvHeads,
                    maxContext: maxContext, headDim: headDim)
            }
            total = product
        }
        return total
    }

    public init(
        device: MTLDevice, layers: Int, kvHeads: Int, maxContext: Int, headDim: Int
    ) throws {
        let bytes = try Self.byteCount(
            layers: layers, kvHeads: kvHeads, maxContext: maxContext, headDim: headDim)
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
        else {
            throw KVCacheError.allocationFailed(bytes: bytes)
        }
        self.layers = layers
        self.kvHeads = kvHeads
        self.maxContext = maxContext
        self.headDim = headDim
        self.buffer = buffer
    }

    /// fp16-element index of `(layer, component, head, position, 0)` in the
    /// buffer. Kernels receive these as element offsets (never as `setBuffer`
    /// offsets — GPUWeights convention); tests use it to address slots.
    public func elementOffset(
        layer: Int, component: Component, head: Int, position: Int
    ) throws -> Int {
        try requireIndex(layer, name: "layer", bound: layers)
        try requireIndex(head, name: "head", bound: kvHeads)
        try requireIndex(position, name: "position", bound: maxContext)
        return (((layer * 2 + component.rawValue) * kvHeads + head)
            * maxContext + position) * headDim
    }

    /// Element index of a `(layer, component)` slab's start — what the
    /// attention kernels take as their base offset.
    func baseElementOffset(layer: Int, component: Component) throws -> Int {
        try elementOffset(layer: layer, component: component, head: 0, position: 0)
    }

    private func requireIndex(_ value: Int, name: String, bound: Int) throws {
        guard value >= 0, value < bound else {
            throw KVCacheError.indexOutOfRange(name: name, value: value, bound: bound)
        }
    }
}
