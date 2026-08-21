import Metal
import QuartzCore

/// Errors from the Metal harness. Every failure path is explicit and named —
/// no silent nil-returns (METHODOLOGY: clear errors at system boundaries).
public enum MetalHarnessError: Error, CustomStringConvertible {
    case noDevice
    case commandQueueCreationFailed(deviceName: String)
    case libraryCompileFailed(underlying: Error)
    case functionNotFound(name: String)
    case commandBufferCreationFailed
    case encoderCreationFailed
    case gpuExecutionFailed(status: MTLCommandBufferStatus, underlying: Error?)

    public var description: String {
        switch self {
        case .noDevice:
            return "No Metal device available (MTLCreateSystemDefaultDevice returned nil)"
        case .commandQueueCreationFailed(let deviceName):
            return "Failed to create command queue on device '\(deviceName)'"
        case .libraryCompileFailed(let underlying):
            return "Metal library compilation failed: \(underlying)"
        case .functionNotFound(let name):
            return "Kernel function '\(name)' not found in library"
        case .commandBufferCreationFailed:
            return "Failed to create command buffer"
        case .encoderCreationFailed:
            return "Failed to create compute command encoder"
        case .gpuExecutionFailed(let status, let underlying):
            let detail = underlying.map { ": \($0)" } ?? ""
            return "Command buffer finished with status \(status.rawValue), not completed\(detail)"
        }
    }
}

/// Device/queue/library setup for the engine (PRD 0b.1) plus the dual-timing
/// dispatch entry point (PRD 0b.2). Shared by the CLI and, from Phase 2, the
/// iOS app — engine logic only ever lives in this package.
public final class MetalContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalHarnessError.noDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw MetalHarnessError.commandQueueCreationFailed(deviceName: device.name)
        }
        self.device = device
        self.queue = queue
    }

    /// Compiles a Metal library from source at runtime. Phase 0 toy kernels
    /// ship as source strings; a precompiled-metallib path can be added when
    /// a phase needs it.
    public func makeLibrary(source: String) throws -> MTLLibrary {
        do {
            return try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalHarnessError.libraryCompileFailed(underlying: error)
        }
    }

    public func makeComputePipeline(
        library: MTLLibrary, function name: String
    ) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw MetalHarnessError.functionNotFound(name: name)
        }
        return try device.makeComputePipelineState(function: function)
    }

    /// Encodes one command-buffer batch via `encode`, runs it to completion,
    /// and returns dual timing (hard rule 7). The wall clock brackets the
    /// entire batch — command-buffer creation, encoding, commit, scheduling,
    /// and GPU execution — so `wallDuration >= gpuDuration` by construction
    /// and the difference is the CPU dispatch overhead.
    ///
    /// `CACurrentMediaTime()` and `MTLCommandBuffer.gpuStartTime`/`gpuEndTime`
    /// share the mach host-time domain, which is what makes the two clocks
    /// directly subtractable.
    @discardableResult
    public func timedDispatch(
        _ encode: (MTLComputeCommandEncoder) throws -> Void
    ) throws -> DispatchTiming {
        let wallStart = CACurrentMediaTime()
        guard let commandBuffer = queue.makeCommandBuffer() else {
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
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let wallEnd = CACurrentMediaTime()

        guard commandBuffer.status == .completed else {
            throw MetalHarnessError.gpuExecutionFailed(
                status: commandBuffer.status, underlying: commandBuffer.error
            )
        }
        return DispatchTiming(
            wallStart: wallStart,
            wallEnd: wallEnd,
            gpuStart: commandBuffer.gpuStartTime,
            gpuEnd: commandBuffer.gpuEndTime
        )
    }
}
