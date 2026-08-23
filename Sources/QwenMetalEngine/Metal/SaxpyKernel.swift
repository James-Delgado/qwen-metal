import Metal

/// Input-validation errors shared by the toy kernels and the BLAS wrapper.
/// Separate from MetalHarnessError: these are caller mistakes, not
/// GPU/harness failures.
public enum KernelInputError: Error, CustomStringConvertible, Equatable {
    case emptyInput
    case lengthMismatch(x: Int, y: Int)
    case nonPositiveDimensions(m: Int, k: Int, n: Int)
    case elementCountMismatch(matrix: String, expected: Int, actual: Int)
    case nonPositiveElementCount(count: Int)
    case elementCountNotMultiple(of: Int, count: Int)
    case invalidIterations(warmup: Int, measured: Int)
    case sampleIndexOutOfRange(index: Int, count: Int)

    public var description: String {
        switch self {
        case .emptyInput:
            return "Kernel input arrays must be non-empty"
        case .lengthMismatch(let x, let y):
            return "Kernel input lengths differ: x has \(x) elements, y has \(y)"
        case .nonPositiveDimensions(let m, let k, let n):
            return "Matrix dimensions must all be positive: m=\(m), k=\(k), n=\(n)"
        case .elementCountMismatch(let matrix, let expected, let actual):
            return "Matrix \(matrix) has \(actual) elements, expected \(expected)"
        case .nonPositiveElementCount(let count):
            return "Element count must be positive, got \(count)"
        case .elementCountNotMultiple(let alignment, let count):
            return "Element count must be a multiple of \(alignment), got \(count)"
        case .invalidIterations(let warmup, let measured):
            return "Iteration counts invalid: warmup \(warmup) must be >= 0, measured \(measured) must be >= 1"
        case .sampleIndexOutOfRange(let index, let count):
            return "Sample index \(index) out of range for \(count) elements"
        }
    }
}

/// Result of one kernel run: the output values plus the mandatory dual
/// GPU + wall timing (hard rule 7 — timing always rides along, never optional).
public struct SaxpyResult: Sendable {
    public let values: [Float]
    public let timing: DispatchTiming
}

/// Kernel 1 (PRD 0b.3): saxpy, out = a*x + y. The first engine-shipped kernel;
/// its job is to establish the dispatch -> readback -> diff-vs-CPU loop that
/// every later kernel follows. One thread per element, compiled from source at
/// runtime per the P0B-1 convention (DECISIONS.md 2026-08-21).
public final class SaxpyKernel {
    /// Inputs are not mutated: the classic saxpy overwrites y, but the engine
    /// convention is immutable inputs with a separate output buffer.
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void saxpy(device const float *x   [[buffer(0)]],
                      device const float *y   [[buffer(1)]],
                      device float *out       [[buffer(2)]],
                      constant float &a       [[buffer(3)]],
                      constant uint &n        [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
        if (gid >= n) return;
        out[gid] = a * x[gid] + y[gid];
    }
    """

    private let context: MetalContext
    private let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        let library = try context.makeLibrary(source: Self.source)
        self.pipeline = try context.makeComputePipeline(library: library, function: "saxpy")
    }

    public func run(a: Float, x: [Float], y: [Float]) throws -> SaxpyResult {
        guard x.count == y.count else {
            throw KernelInputError.lengthMismatch(x: x.count, y: y.count)
        }
        guard !x.isEmpty else {
            throw KernelInputError.emptyInput
        }

        let count = x.count
        let byteLength = count * MemoryLayout<Float>.stride
        let device = context.device
        guard
            let xBuffer = x.withUnsafeBytes({
                device.makeBuffer(bytes: $0.baseAddress!, length: byteLength,
                                  options: .storageModeShared)
            }),
            let yBuffer = y.withUnsafeBytes({
                device.makeBuffer(bytes: $0.baseAddress!, length: byteLength,
                                  options: .storageModeShared)
            }),
            let outBuffer = device.makeBuffer(length: byteLength,
                                              options: .storageModeShared)
        else {
            throw MetalHarnessError.bufferAllocationFailed(length: byteLength)
        }

        var scalar = a
        var n = UInt32(count)
        let pipeline = self.pipeline
        let timing = try context.timedDispatch { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(xBuffer, offset: 0, index: 0)
            encoder.setBuffer(yBuffer, offset: 0, index: 1)
            encoder.setBuffer(outBuffer, offset: 0, index: 2)
            encoder.setBytes(&scalar, length: MemoryLayout<Float>.stride, index: 3)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 4)
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }

        let values = [Float](
            UnsafeBufferPointer(
                start: outBuffer.contents().bindMemory(to: Float.self, capacity: count),
                count: count
            )
        )
        return SaxpyResult(values: values, timing: timing)
    }
}
