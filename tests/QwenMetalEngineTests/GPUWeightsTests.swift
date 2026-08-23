import XCTest
@testable import QwenMetalEngine
import Metal

/// P2-1 tests: GPU weight residency (docs/phases/phase-2.md D1).
///
/// Every gate here is EXACT (==, bitwise) per the pre-committed Phase 2 gates
/// (DECISIONS.md 2026-08-23): the bf16→fp32 bit-shift upcast, and residency
/// mode B (wired copy) vs mode A (mmap), carry no tolerance. Synthetic files
/// are tiny hand-built blobs (SafetensorsFileTests pattern) so the suite runs
/// on any machine; the one real-checkpoint spot check skips when the
/// local-only artifact is absent.
final class GPUWeightsTests: XCTestCase {

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("GPUWeightsTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: tempDir)
    }

    // MARK: - Byte-building helpers (SafetensorsFileTests pattern)

    private func blob(headerJSON: String, payload: [UInt8]) -> Data {
        let header = Data(headerJSON.utf8)
        var out = Data()
        var length = UInt64(header.count).littleEndian
        withUnsafeBytes(of: &length) { out.append(contentsOf: $0) }
        out.append(header)
        out.append(contentsOf: payload)
        return out
    }

    private func halfwordBytes(_ halfwords: [UInt16]) -> [UInt8] {
        halfwords.flatMap { [UInt8($0 & 0xFF), UInt8($0 >> 8)] }
    }

    @discardableResult
    private func write(_ data: Data, as name: String = "model.safetensors") throws -> String {
        let url = tempDir.appendingPathComponent(name)
        try data.write(to: url)
        return url.path
    }

    // MARK: - GPU harness

    private func makeContextOrSkip() throws -> MetalContext {
        do {
            return try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    /// Test-only readback kernels: byte-assembled 16-bit loads (so ANY tensor
    /// byte offset works, odd included — the safetensors format only promises
    /// 2-byte-sized elements, not alignment) upcast in registers exactly as
    /// spec D1 prescribes for the production kernels (bf16: integer shift +
    /// bitcast; fp16: hardware widening convert).
    private static let upcastSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void upcast_bf16(device const uchar *raw  [[buffer(0)]],
                            device float *out        [[buffer(1)]],
                            constant ulong &byteOffset [[buffer(2)]],
                            constant uint &count     [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
        if (gid >= count) return;
        ulong b = byteOffset + 2 * ulong(gid);
        ushort w = ushort(raw[b]) | (ushort(raw[b + 1]) << 8);
        out[gid] = as_type<float>(uint(w) << 16);
    }

    kernel void upcast_f16(device const uchar *raw  [[buffer(0)]],
                           device float *out        [[buffer(1)]],
                           constant ulong &byteOffset [[buffer(2)]],
                           constant uint &count     [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
        if (gid >= count) return;
        ulong b = byteOffset + 2 * ulong(gid);
        ushort w = ushort(raw[b]) | (ushort(raw[b + 1]) << 8);
        out[gid] = float(as_type<half>(w));
    }
    """

    /// Reads tensor `name` back through the residency buffer with the upcast
    /// kernel matching its dtype, returning fp32 values.
    private func gpuReadback(
        context: MetalContext, weights: GPUWeights, tensor name: String
    ) throws -> [Float] {
        let info = try weights.info(for: name)
        let function = info.dtype == .bfloat16 ? "upcast_bf16" : "upcast_f16"
        let library = try context.makeLibrary(source: Self.upcastSource)
        let pipeline = try context.makeComputePipeline(library: library, function: function)

        let count = info.elementCount
        let outLength = max(count, 1) * MemoryLayout<Float>.stride
        guard let outBuffer = context.device.makeBuffer(
            length: outLength, options: .storageModeShared) else {
            throw MetalHarnessError.bufferAllocationFailed(length: outLength)
        }
        var byteOffset = UInt64(try weights.byteOffset(for: name))
        var n = UInt32(count)
        try context.timedDispatch { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(weights.buffer, offset: 0, index: 0)
            encoder.setBuffer(outBuffer, offset: 0, index: 1)
            encoder.setBytes(&byteOffset, length: MemoryLayout<UInt64>.stride, index: 2)
            encoder.setBytes(&n, length: MemoryLayout<UInt32>.stride, index: 3)
            let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
            )
        }
        return [Float](
            UnsafeBufferPointer(
                start: outBuffer.contents().bindMemory(to: Float.self, capacity: count),
                count: count
            )
        )
    }

    /// Bitwise equality — stricter than ==: distinguishes -0 from 0 and
    /// compares NaN payloads instead of failing on them.
    private func assertBitwiseEqual(
        _ got: [Float], _ expected: [Float],
        _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(
            got.map(\.bitPattern), expected.map(\.bitPattern),
            message, file: file, line: line)
    }

    // MARK: - No-copy mmap path: exact upcast through the GPU

    func testNoCopyBF16ReadThroughGPUMatchesCPUExactly() throws {
        let context = try makeContextOrSkip()
        // Normals, ±0, subnormal, ±inf, and NaN payloads: the bf16 upcast is
        // a bit shift on BOTH sides, so every 16-bit pattern must round-trip
        // bit-for-bit — NaNs included.
        let bits: [UInt16] = [
            0x3F80, 0xBF80, 0x0000, 0x8000, 0x0001, 0x7F80, 0xFF80,
            0x7FC1, 0xFFA5, 0x4049, 0xC2F7, 0x0080,
        ]
        let json = #"{"w":{"dtype":"BF16","shape":[3,4],"data_offsets":[0,24]}}"#
        let path = try write(blob(headerJSON: json, payload: halfwordBytes(bits)))
        let file = try SafetensorsFile(path: path)
        let weights = try GPUWeights(file: file, context: context, residency: .mmap)

        assertBitwiseEqual(
            try gpuReadback(context: context, weights: weights, tensor: "w"),
            try file.fp32Values(for: "w"),
            "GPU bf16 upcast through the no-copy buffer must be exact")
    }

    func testNoCopyF16ReadThroughGPUMatchesCPUExactly() throws {
        let context = try makeContextOrSkip()
        // Finite patterns + infinities (fp16→fp32 widening is exact for all
        // of them). NaNs excluded: NaN quieting is convert-implementation
        // detail, and the checkpoint pipeline never carries NaN weights.
        let bits: [UInt16] = [
            0x3C00, 0xC000, 0x3555, 0x0001, 0x03FF, 0x0400, 0x7BFF,
            0x7C00, 0xFC00, 0x8000, 0x0000,
        ]
        let json = #"{"w":{"dtype":"F16","shape":[11],"data_offsets":[0,22]}}"#
        let path = try write(blob(headerJSON: json, payload: halfwordBytes(bits)))
        let file = try SafetensorsFile(path: path)
        let weights = try GPUWeights(file: file, context: context, residency: .mmap)

        assertBitwiseEqual(
            try gpuReadback(context: context, weights: weights, tensor: "w"),
            try file.fp32Values(for: "w"),
            "GPU fp16 upcast through the no-copy buffer must be exact")
    }

    // MARK: - Odd byte offsets (spec edge case 7)

    func testOddDataSectionOffsetUpcastsExactly() throws {
        let context = try makeContextOrSkip()
        // Odd-length header JSON ⇒ the data section starts at 8 + odd, so
        // every tensor's absolute byte offset is ODD — 16-bit typed loads
        // cannot address it, byte-assembled loads must stay exact.
        var json = #"{"w":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}"#
        if json.utf8.count % 2 == 0 { json += " " }
        XCTAssertEqual(json.utf8.count % 2, 1, "test premise: odd header length")
        let bits: [UInt16] = [0x3F80, 0x7FC1, 0x8000, 0x0001]
        let path = try write(blob(headerJSON: json, payload: halfwordBytes(bits)))
        let file = try SafetensorsFile(path: path)
        let weights = try GPUWeights(file: file, context: context, residency: .mmap)

        XCTAssertEqual(try weights.byteOffset(for: "w") % 2, 1, "test premise: odd offset")
        assertBitwiseEqual(
            try gpuReadback(context: context, weights: weights, tensor: "w"),
            try file.fp32Values(for: "w"),
            "odd tensor byte offset must not break the exact upcast")
    }

    // MARK: - Tensor offsets come from the parsed header index

    func testTensorByteOffsetsComeFromHeaderIndex() throws {
        let context = try makeContextOrSkip()
        let json = #"{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"# +
            #""b":{"dtype":"F16","shape":[3],"data_offsets":[4,10]}}"#
        let payload = halfwordBytes([0x3F80, 0x4000, 0x3C00, 0x4000, 0x4200])
        let path = try write(blob(headerJSON: json, payload: payload))
        let file = try SafetensorsFile(path: path)
        let weights = try GPUWeights(file: file, context: context, residency: .mmap)

        // Absolute offset = 8-byte length prefix + header + data_offsets begin.
        let dataSectionStart = 8 + json.utf8.count
        XCTAssertEqual(try weights.byteOffset(for: "a"), dataSectionStart + 0)
        XCTAssertEqual(try weights.byteOffset(for: "b"), dataSectionStart + 4)

        XCTAssertThrowsError(try weights.byteOffset(for: "missing")) { error in
            guard case SafetensorsError.tensorNotFound = error else {
                return XCTFail("expected tensorNotFound, got \(error)")
            }
        }
    }

    // MARK: - Wired-copy residency variant (mode B): bit-identical to mmap

    func testWiredCopyIsBitIdenticalToMmap() throws {
        let context = try makeContextOrSkip()
        let bits: [UInt16] = [0x3F80, 0x7FC1, 0xBF80, 0x0001, 0x4049, 0x8000]
        let json = #"{"w":{"dtype":"BF16","shape":[6],"data_offsets":[0,12]}}"#
        let path = try write(blob(headerJSON: json, payload: halfwordBytes(bits)))
        let file = try SafetensorsFile(path: path)

        let mmapWeights = try GPUWeights(file: file, context: context, residency: .mmap)
        let wiredWeights = try GPUWeights(file: file, context: context, residency: .wiredCopy)
        XCTAssertEqual(mmapWeights.residency, .mmap)
        XCTAssertEqual(wiredWeights.residency, .wiredCopy)

        // Same bytes: the file-sized prefix of both buffers must match, and
        // the tensor index must resolve identically.
        let fileSize = try XCTUnwrap(
            FileManager.default.attributesOfItem(atPath: path)[.size] as? Int)
        XCTAssertGreaterThanOrEqual(mmapWeights.buffer.length, fileSize)
        XCTAssertGreaterThanOrEqual(wiredWeights.buffer.length, fileSize)
        XCTAssertEqual(
            memcmp(mmapWeights.buffer.contents(), wiredWeights.buffer.contents(), fileSize), 0,
            "wired copy must hold the identical checkpoint bytes")
        XCTAssertEqual(
            try mmapWeights.byteOffset(for: "w"), try wiredWeights.byteOffset(for: "w"))

        // Same kernel, both residency modes: bitwise-equal outputs (the
        // pre-committed exact gate — residency isolates WHERE bytes live).
        assertBitwiseEqual(
            try gpuReadback(context: context, weights: wiredWeights, tensor: "w"),
            try gpuReadback(context: context, weights: mmapWeights, tensor: "w"),
            "wired-copy outputs must be bit-identical to mmap outputs")
    }

    // MARK: - Page alignment / length handling

    func testNonPageMultipleFileLengthIsHandled() throws {
        let context = try makeContextOrSkip()
        // Every synthetic file here is far below one page, so this pins the
        // rounding contract explicitly: no-copy buffers need page-multiple
        // lengths, and mmap maps whole pages, so rounding up is always safe.
        let bits: [UInt16] = [0x3F80, 0x4000, 0x4040]
        let json = #"{"w":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#
        let path = try write(blob(headerJSON: json, payload: halfwordBytes(bits)))
        let fileSize = try XCTUnwrap(
            FileManager.default.attributesOfItem(atPath: path)[.size] as? Int)
        let pageSize = Int(getpagesize())
        XCTAssertNotEqual(fileSize % pageSize, 0, "test premise: non-page-multiple file")

        let file = try SafetensorsFile(path: path)
        let weights = try GPUWeights(file: file, context: context, residency: .mmap)

        XCTAssertEqual(weights.buffer.length % pageSize, 0,
                       "no-copy buffer length must be page-rounded")
        XCTAssertGreaterThanOrEqual(weights.buffer.length, fileSize)
        XCTAssertEqual(UInt(bitPattern: weights.buffer.contents()) % UInt(pageSize), 0,
                       "no-copy buffer must sit on the page-aligned mmap base")
        // The last tensor in the file still reads back exactly.
        assertBitwiseEqual(
            try gpuReadback(context: context, weights: weights, tensor: "w"),
            try file.fp32Values(for: "w"),
            "tensor at the unpadded file tail must read back exactly")
    }

    // MARK: - Lifetime: the weights object keeps the mapping alive

    func testBufferRemainsValidAfterCallerReleasesFile() throws {
        let context = try makeContextOrSkip()
        let bits: [UInt16] = [0x3F80, 0xC2F7, 0x0080, 0x7F80]
        let json = #"{"w":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}"#
        let path = try write(blob(headerJSON: json, payload: halfwordBytes(bits)))

        let expected: [Float]
        let weights: GPUWeights
        do {
            // The caller's only SafetensorsFile reference dies with this
            // scope; GPUWeights must retain the mapping (a no-copy buffer
            // over unmapped pages would fault or read garbage).
            let file = try SafetensorsFile(path: path)
            expected = try file.fp32Values(for: "w")
            weights = try GPUWeights(file: file, context: context, residency: .mmap)
        }
        assertBitwiseEqual(
            try gpuReadback(context: context, weights: weights, tensor: "w"),
            expected,
            "no-copy buffer must stay valid after the caller drops the file")
    }

    // MARK: - Real-checkpoint spot check (skips when the artifact is absent)

    func testRealCheckpointNoCopySpotCheckMatchesCPU() throws {
        try SharedCheckpoint.skipUnlessCheckpointPresent()
        let context = try makeContextOrSkip()
        let file = try SafetensorsFile(path: SharedCheckpoint.checkpointURL.path)
        let weights = try GPUWeights(file: file, context: context, residency: .mmap)

        // Small tensor (hidden_size fp32 values) — cheap even on the 3.4 GB
        // mapping. Also records the alignment fact the P2-2 production
        // kernels rely on: real-checkpoint offsets are 16-bit aligned.
        let name = "model.norm.weight"
        XCTAssertEqual(try weights.byteOffset(for: name) % 2, 0,
                       "pinned checkpoint tensor offsets are expected 2-byte aligned")
        assertBitwiseEqual(
            try gpuReadback(context: context, weights: weights, tensor: name),
            try file.fp32Values(for: name),
            "real-checkpoint no-copy readback must match the CPU upcast exactly")
    }
}
