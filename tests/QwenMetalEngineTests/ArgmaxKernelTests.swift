import XCTest
@testable import QwenMetalEngine
import Metal

/// P4-8 kernel correctness (hard rule 3: before any pipeline wiring): the GPU
/// argmax must equal CPU `Argmax.firstIndex` EXACTLY — same index, every
/// input, no tolerance. Ties (lowest index wins), ±0.0, NaN (incl. the
/// index-0 latch), ±inf, subnormals, threadgroup/grid-stride boundary sizes,
/// and the full pinned vocab size are all pinned here.
final class ArgmaxKernelTests: XCTestCase {

    // MARK: - Deterministic RNG (SgemmTests pattern; system RNGs are unseedable)

    private struct SplitMix64: RandomNumberGenerator {
        private var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    // MARK: - Harness

    private func makeContextOrSkip() throws -> MetalContext {
        do {
            return try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    private func gpuArgmax(
        _ values: [Float], context: MetalContext, kernel: ArgmaxKernel
    ) throws -> Int {
        let device = context.device
        let length = values.count * 4
        guard let valuesBuf = values.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: length,
                              options: .storageModeShared)
        }) else {
            throw MetalHarnessError.bufferAllocationFailed(length: length)
        }
        guard let outBuf = device.makeBuffer(length: 4, options: .storageModeShared) else {
            throw MetalHarnessError.bufferAllocationFailed(length: 4)
        }
        // Poison the output slot so a silently-skipped write cannot pass.
        outBuf.contents().storeBytes(of: UInt32.max, as: UInt32.self)
        try context.timedDispatch { encoder in
            try kernel.encodeArgmax(
                into: encoder, values: valuesBuf, count: values.count, output: outBuf)
        }
        return Int(outBuf.contents().load(as: UInt32.self))
    }

    /// The exact-equality pin, applied to one input vector.
    private func assertMatchesCPU(
        _ values: [Float], context: MetalContext, kernel: ArgmaxKernel,
        _ note: String, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let gpu = try gpuArgmax(values, context: context, kernel: kernel)
        let cpu = Argmax.firstIndex(values)
        XCTAssertEqual(
            gpu, cpu, "GPU argmax must equal CPU Argmax.firstIndex — \(note)",
            file: file, line: line)
    }

    // MARK: - Random values across sizes (incl. threadgroup boundaries)

    /// Sizes straddle 1 thread, SIMD width, the 1024-thread threadgroup, and
    /// grid-stride wrap points — every partition edge the reduction has.
    func testRandomVectorsMatchCPUAcrossSizes() throws {
        let context = try makeContextOrSkip()
        let kernel = try ArgmaxKernel(context: context)
        let sizes = [1, 2, 3, 31, 32, 33, 127, 128, 129,
                     1023, 1024, 1025, 2048, 4095, 4096, 4097]
        var rng = SplitMix64(seed: 41)
        for size in sizes {
            let values = (0..<size).map { _ in Float.random(in: -10...10, using: &rng) }
            try assertMatchesCPU(
                values, context: context, kernel: kernel, "random size \(size)")
        }
    }

    // MARK: - Crafted ties (lowest index wins)

    func testCraftedTiesLowestIndexWins() throws {
        let context = try makeContextOrSkip()
        let kernel = try ArgmaxKernel(context: context)
        let n = 3000   // > 2 × 1024: ties land in different grid-stride laps
        var rng = SplitMix64(seed: 42)
        let base = (0..<n).map { _ in Float.random(in: -1...1, using: &rng) }

        let tiePairs = [(0, n - 1), (5, 6), (100, 100 + 1024), (7, 7 + 2048)]
        for (a, b) in tiePairs {
            var values = base
            values[a] = 5.0
            values[b] = 5.0
            let gpu = try gpuArgmax(values, context: context, kernel: kernel)
            XCTAssertEqual(gpu, min(a, b), "tie at \(a)/\(b) must pick the lower index")
            XCTAssertEqual(gpu, Argmax.firstIndex(values))
        }

        let allEqual = [Float](repeating: 2.5, count: n)
        try assertMatchesCPU(
            allEqual, context: context, kernel: kernel, "all-equal vector")
        XCTAssertEqual(try gpuArgmax(allEqual, context: context, kernel: kernel), 0)
    }

    /// IEEE `>` treats -0.0 and +0.0 as equal — they must tie by index, not
    /// by bit pattern.
    func testSignedZeroTiesByIndexNotBitPattern() throws {
        let context = try makeContextOrSkip()
        let kernel = try ArgmaxKernel(context: context)
        for values in [[-0.0, 0.0], [0.0, -0.0],
                       [-1.0, -0.0, 0.0], [-1.0, 0.0, -0.0]] as [[Float]] {
            try assertMatchesCPU(
                values, context: context, kernel: kernel, "signed zeros \(values)")
        }
        XCTAssertEqual(try gpuArgmax([-0.0, 0.0], context: context, kernel: kernel), 0)
        XCTAssertEqual(
            try gpuArgmax([-1.0, -0.0, 0.0], context: context, kernel: kernel), 1)
    }

    // MARK: - NaN / infinity edge logits

    func testNaNEdgeLogits() throws {
        let context = try makeContextOrSkip()
        let kernel = try ArgmaxKernel(context: context)
        let nan = Float.nan

        // The CPU scan latches a NaN at index 0 forever — even against
        // larger finite values later.
        try assertMatchesCPU(
            [nan, 100.0, 200.0], context: context, kernel: kernel, "NaN at 0")
        XCTAssertEqual(
            try gpuArgmax([nan, 100.0, 200.0], context: context, kernel: kernel), 0)

        try assertMatchesCPU(
            [nan, nan, nan], context: context, kernel: kernel, "all NaN")
        try assertMatchesCPU(
            [1.0, nan, 2.0], context: context, kernel: kernel, "NaN mid-vector")
        try assertMatchesCPU(
            [3.0, nan], context: context, kernel: kernel, "NaN last")
        // NaN vs -inf: the scan never selects a non-leading NaN, so the real
        // -inf at index 0 must win.
        try assertMatchesCPU(
            [-.infinity, nan, nan], context: context, kernel: kernel,
            "-inf then NaNs")
        // Negative NaN bit patterns behave like any NaN.
        let negNaN = Float(bitPattern: 0xFFC0_0000)
        try assertMatchesCPU(
            [1.0, negNaN, 2.0], context: context, kernel: kernel, "negative NaN")
        try assertMatchesCPU(
            [negNaN, 7.0], context: context, kernel: kernel, "negative NaN at 0")
    }

    func testInfinityEdgeLogits() throws {
        let context = try makeContextOrSkip()
        let kernel = try ArgmaxKernel(context: context)
        let inf = Float.infinity

        try assertMatchesCPU(
            [1.0, inf, inf, 2.0], context: context, kernel: kernel,
            "+inf duplicated — first wins")
        XCTAssertEqual(
            try gpuArgmax([1.0, inf, inf, 2.0], context: context, kernel: kernel), 1)
        try assertMatchesCPU(
            [-inf, -inf, -inf], context: context, kernel: kernel, "all -inf")
        try assertMatchesCPU(
            [-inf, -inf, -1e30, -inf], context: context, kernel: kernel,
            "one finite among -inf")
        try assertMatchesCPU(
            [inf, -inf], context: context, kernel: kernel, "+inf first")
    }

    // MARK: - Arbitrary bit patterns (the strongest exactness sweep)

    /// Uniformly random u32 bit patterns reinterpreted as fp32 — NaN payloads,
    /// ±inf, subnormals, ±0.0, everything. GPU must equal the CPU scan on all
    /// of them, across multiple seeds (some draws land NaN at index 0).
    func testRandomBitPatternsMatchCPUExactly() throws {
        let context = try makeContextOrSkip()
        let kernel = try ArgmaxKernel(context: context)
        for seed: UInt64 in 1...10 {
            var rng = SplitMix64(seed: seed)
            let values = (0..<8192).map { _ in
                Float(bitPattern: UInt32(truncatingIfNeeded: rng.next()))
            }
            try assertMatchesCPU(
                values, context: context, kernel: kernel, "bit-pattern seed \(seed)")
        }
    }

    // MARK: - Full-vocab real dims

    /// The production shape: vocab 151936 fp32 logits.
    func testFullVocabRealDims() throws {
        let context = try makeContextOrSkip()
        let kernel = try ArgmaxKernel(context: context)
        let vocab = 151936
        var rng = SplitMix64(seed: 7)
        var values = (0..<vocab).map { _ in Float.random(in: -20...20, using: &rng) }
        try assertMatchesCPU(
            values, context: context, kernel: kernel, "full-vocab random")

        // Planted tie far apart, above every random draw.
        values[1234] = 25.0
        values[150000] = 25.0
        XCTAssertEqual(
            try gpuArgmax(values, context: context, kernel: kernel), 1234)
        XCTAssertEqual(Argmax.firstIndex(values), 1234)

        // Maximum at the very last index.
        values[vocab - 1] = 30.0
        try assertMatchesCPU(
            values, context: context, kernel: kernel, "max at last index")
    }

    // MARK: - Host-side validation

    func testEncodeValidationRejectsBadInputs() throws {
        let context = try makeContextOrSkip()
        let kernel = try ArgmaxKernel(context: context)
        let device = context.device
        let values = device.makeBuffer(length: 16, options: .storageModeShared)!
        let out = device.makeBuffer(length: 4, options: .storageModeShared)!

        try context.timedDispatch { encoder in
            XCTAssertThrowsError(
                try kernel.encodeArgmax(
                    into: encoder, values: values, count: 0, output: out)
            ) { error in
                guard case DecodeKernelError.nonPositiveDimension = error else {
                    return XCTFail("expected nonPositiveDimension, got \(error)")
                }
            }
            XCTAssertThrowsError(
                try kernel.encodeArgmax(
                    into: encoder, values: values, count: 5, output: out)
            ) { error in
                guard case DecodeKernelError.bufferTooSmall(let name, _, _) = error else {
                    return XCTFail("expected bufferTooSmall, got \(error)")
                }
                XCTAssertEqual(name, "values")
            }
            let tinyOut = device.makeBuffer(length: 2, options: .storageModeShared)!
            XCTAssertThrowsError(
                try kernel.encodeArgmax(
                    into: encoder, values: values, count: 4, output: tinyOut)
            ) { error in
                guard case DecodeKernelError.bufferTooSmall(let name, _, _) = error else {
                    return XCTFail("expected bufferTooSmall, got \(error)")
                }
                XCTAssertEqual(name, "output")
            }
            // Leave one valid dispatch so the command buffer completes.
            try kernel.encodeArgmax(
                into: encoder, values: values, count: 4, output: out)
        }
    }
}
