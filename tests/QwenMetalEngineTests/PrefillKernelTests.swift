import XCTest
@testable import QwenMetalEngine
import Metal

/// P5-3 kernel-level tests for the batched prefill kernels (phase-5.md
/// D2/D6). The batched kernels claim ARITHMETIC IDENTITY with the per-token
/// kernels they batch, so the binding assertions here are bitwise:
///
/// - batched embedding gather == the P3-4 per-token gather, row for row
///   (the pre-committed EXACT gate carries);
/// - batched qk-norm/RoPE/append cluster == the P4-3 single-position
///   cluster looped over positions (q output, k slots, v slots — v
///   bitwise-copy claim included, adversarial bit patterns);
/// - copy-row == its source row bitwise (pure move);
/// - the new SDPA host offsets: offset rows == offset-0 runs on shifted
///   data, and misaligned offsets are rejected pre-dispatch.
///
/// Hard rule 3: these tests land WITH the kernels, before any optimization.
final class PrefillKernelTests: XCTestCase {

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

    private func makeContextOrSkip() throws -> MetalContext {
        do {
            return try MetalContext()
        } catch MetalHarnessError.noDevice {
            throw XCTSkip("No Metal device available on this machine")
        }
    }

    private func makeBuffer<T>(_ device: MTLDevice, values: [T]) throws -> MTLBuffer {
        let length = values.count * MemoryLayout<T>.stride
        guard let buffer = values.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: length,
                              options: .storageModeShared)
        }) else {
            throw MetalHarnessError.bufferAllocationFailed(length: length)
        }
        return buffer
    }

    private func makeOutputBuffer(
        _ device: MTLDevice, count: Int, elementStride: Int
    ) throws -> MTLBuffer {
        let length = count * elementStride
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared)
        else {
            throw MetalHarnessError.bufferAllocationFailed(length: length)
        }
        return buffer
    }

    private func readHalfs(_ buffer: MTLBuffer, count: Int) -> [Float16] {
        [Float16](UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float16.self, capacity: count),
            count: count))
    }

    private func randomHalfs(
        _ count: Int, range: ClosedRange<Float> = -1...1, rng: inout SplitMix64
    ) -> [Float16] {
        (0..<count).map { _ in Float16(Float.random(in: range, using: &rng)) }
    }

    private func bf16Bits(_ v: Float) -> UInt16 {
        let bits = UInt16(truncatingIfNeeded: v.bitPattern >> 16)
        precondition(Float(bitPattern: UInt32(bits) << 16) == v,
                     "test value \(v) is not bf16-exact")
        return bits
    }

    // MARK: - Batched embedding gather == per-token gather, bitwise

    /// Random q4g64 embedding triplet (vocab 32, hidden 64): the batched
    /// gather's rows must be BITWISE identical to the per-token
    /// embedding_gather_q4_f16 run token by token — same dequant, same
    /// single rounding, EXACT gate (any nibble/group/row addressing bug is
    /// a hard mismatch).
    func testBatchedGatherMatchesPerTokenGatherBitwise() throws {
        let context = try makeContextOrSkip()
        let device = context.device
        var rng = SplitMix64(seed: 11)
        let vocab = 32
        let hidden = 64

        let words = (0..<(vocab * hidden / 8)).map { _ in
            UInt32(truncatingIfNeeded: rng.next())
        }
        let groups = vocab * hidden / 64
        let scales = (0..<groups).map { _ in
            Float16(Float.random(in: 0.002...0.05, using: &rng))
        }
        let biases = (0..<groups).map { _ in
            Float16(Float.random(in: -0.4...0.4, using: &rng))
        }
        let q = try makeBuffer(device, values: words)
        let s = try makeBuffer(device, values: scales)
        let b = try makeBuffer(device, values: biases)

        let tokens = [0, 31, 7, 7, 15]  // repeats + both vocab extremes
        let quant = try QuantKernels(context: context)
        var refRows: [Float16] = []
        for token in tokens {
            let row = try makeOutputBuffer(device, count: hidden, elementStride: 2)
            try context.timedDispatch { encoder in
                try quant.encodeEmbeddingGather(
                    into: encoder, q: q, qByteOffset: 0,
                    scales: s, scalesByteOffset: 0,
                    biases: b, biasesByteOffset: 0,
                    vocabSize: vocab, hiddenSize: hidden,
                    tokenId: token, output: row)
            }
            refRows += readHalfs(row, count: hidden)
        }

        let prefill = try PrefillKernels(context: context)
        let ids = try makeBuffer(device, values: tokens.map(UInt32.init))
        let out = try makeOutputBuffer(
            device, count: tokens.count * hidden, elementStride: 2)
        try context.timedDispatch { encoder in
            try prefill.encodeEmbeddingGatherBatch(
                into: encoder, q: q, qByteOffset: 0,
                scales: s, scalesByteOffset: 0,
                biases: b, biasesByteOffset: 0,
                tokenIds: ids, batch: tokens.count,
                vocabSize: vocab, hiddenSize: hidden, output: out)
        }
        let got = readHalfs(out, count: tokens.count * hidden)
        XCTAssertEqual(
            got.map(\.bitPattern), refRows.map(\.bitPattern),
            "batched gather must be bitwise identical to per-token gathers")
    }

    // MARK: - Batched cluster == single-position cluster looped, bitwise

    /// Batched qk-norm/RoPE/append over 5 positions (base 3, so absolute
    /// RoPE indexing is exercised) vs the P4-3 single-position cluster run
    /// position by position on the same values: q output rows, k cache
    /// slots, and v cache slots all bitwise equal. The v rows carry
    /// adversarial bit patterns (-0.0, NaN payload, subnormal) — the
    /// bitwise-copy claim of the gates entry.
    func testBatchedClusterMatchesSinglePositionClusterBitwise() throws {
        let context = try makeContextOrSkip()
        let device = context.device
        var rng = SplitMix64(seed: 23)
        let numHeads = 4
        let kvHeads = 2
        let headDim = 16
        let maxContext = 40
        let batch = 5
        let basePosition = 3
        let eps: Float = 1e-6

        let qNormValues = (0..<headDim).map { Float(($0 * 5 % 37) - 18) * 0.03125 }
        let kNormValues = (0..<headDim).map { Float(($0 * 11 % 41) - 20) * 0.03125 }
        let qNormBuffer = try makeBuffer(device, values: qNormValues.map(bf16Bits))
        let kNormBuffer = try makeBuffer(device, values: kNormValues.map(bf16Bits))
        let rope = try RoPE(headDim: headDim, theta: 10000, positions: maxContext)
        let cosBuffer = try makeBuffer(device, values: rope.cosValues)
        let sinBuffer = try makeBuffer(device, values: rope.sinValues)

        var qRows = randomHalfs(batch * numHeads * headDim, rng: &rng)
        let kRows = randomHalfs(batch * kvHeads * headDim, rng: &rng)
        var vRows = randomHalfs(batch * kvHeads * headDim, rng: &rng)
        // Adversarial v bit patterns (P2-3 kv-append exactness species).
        vRows[0] = Float16(bitPattern: 0x8000)              // -0.0
        vRows[1] = Float16(bitPattern: 0x7E01)              // NaN payload
        vRows[2] = Float16(bitPattern: 0x0001)              // subnormal
        qRows[0] = Float16(bitPattern: 0x8000)

        // Reference: single-position FoldedKernels cluster per position,
        // consuming the concatenated [q | k | v] per-position layout.
        let folded = try FoldedKernels(context: context)
        let refCache = try KVCache(
            device: device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        var refQ: [Float16] = []
        for p in 0..<batch {
            let qSlice = Array(qRows[(p * numHeads * headDim)..<((p + 1) * numHeads * headDim)])
            let kSlice = Array(kRows[(p * kvHeads * headDim)..<((p + 1) * kvHeads * headDim)])
            let vSlice = Array(vRows[(p * kvHeads * headDim)..<((p + 1) * kvHeads * headDim)])
            let qkv = try makeBuffer(device, values: qSlice + kSlice + vSlice)
            let qOut = try makeOutputBuffer(
                device, count: numHeads * headDim, elementStride: 2)
            try context.timedDispatch { encoder in
                try folded.encodeQKNormRoPEAppend(
                    into: encoder, qkv: qkv,
                    qNormWeight: qNormBuffer, qNormByteOffset: 0,
                    kNormWeight: kNormBuffer, kNormByteOffset: 0,
                    cosTable: cosBuffer, sinTable: sinBuffer,
                    position: basePosition + p, positions: maxContext,
                    numHeads: numHeads, eps: eps,
                    cache: refCache, layer: 0, qOut: qOut)
            }
            refQ += readHalfs(qOut, count: numHeads * headDim)
        }

        // Batched run on the same values.
        let prefill = try PrefillKernels(context: context)
        let cache = try KVCache(
            device: device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        let qIn = try makeBuffer(device, values: qRows)
        let kIn = try makeBuffer(device, values: kRows)
        let vIn = try makeBuffer(device, values: vRows)
        let qOut = try makeOutputBuffer(
            device, count: batch * numHeads * headDim, elementStride: 2)
        try context.timedDispatch { encoder in
            try prefill.encodeQKNormRoPEAppendBatch(
                into: encoder, qIn: qIn, kIn: kIn, vIn: vIn,
                qNormWeight: qNormBuffer, qNormByteOffset: 0,
                kNormWeight: kNormBuffer, kNormByteOffset: 0,
                cosTable: cosBuffer, sinTable: sinBuffer,
                basePosition: basePosition, batch: batch,
                positions: maxContext, numHeads: numHeads, eps: eps,
                cache: cache, layer: 0, qOut: qOut)
        }

        XCTAssertEqual(
            readHalfs(qOut, count: batch * numHeads * headDim).map(\.bitPattern),
            refQ.map(\.bitPattern),
            "batched cluster q output must be bitwise identical")
        let gotCache = readHalfs(cache.buffer, count: cache.buffer.length / 2)
        let refCacheHalfs = readHalfs(refCache.buffer, count: refCache.buffer.length / 2)
        for p in 0..<batch {
            for component in [KVCache.Component.key, .value] {
                for head in 0..<kvHeads {
                    let offset = try cache.elementOffset(
                        layer: 0, component: component, head: head,
                        position: basePosition + p)
                    XCTAssertEqual(
                        gotCache[offset..<(offset + headDim)].map(\.bitPattern),
                        refCacheHalfs[offset..<(offset + headDim)].map(\.bitPattern),
                        "\(component) slot at position \(basePosition + p), "
                        + "head \(head) must be bitwise identical")
                }
            }
        }
    }

    /// The batched cluster's context-full contract: a chunk whose last
    /// position would land at or past maxContext throws `.contextFull`
    /// BEFORE any dispatch, cache untouched.
    func testBatchedClusterContextFullThrowsPreDispatchCacheUntouched() throws {
        let context = try makeContextOrSkip()
        let device = context.device
        var rng = SplitMix64(seed: 31)
        let numHeads = 2
        let kvHeads = 1
        let headDim = 8
        let maxContext = 4

        let prefill = try PrefillKernels(context: context)
        let cache = try KVCache(
            device: device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        // Sentinel fill so any write is visible.
        memset(cache.buffer.contents(), 0x5A, cache.buffer.length)
        let before = readHalfs(cache.buffer, count: cache.buffer.length / 2)
            .map(\.bitPattern)

        let normValues = (0..<headDim).map { _ in Float(1) }
        let normBuffer = try makeBuffer(device, values: normValues.map(bf16Bits))
        let rope = try RoPE(headDim: headDim, theta: 10000, positions: maxContext)
        let cosBuffer = try makeBuffer(device, values: rope.cosValues)
        let sinBuffer = try makeBuffer(device, values: rope.sinValues)
        let batch = 3
        let qIn = try makeBuffer(device, values: randomHalfs(batch * numHeads * headDim, rng: &rng))
        let kIn = try makeBuffer(device, values: randomHalfs(batch * kvHeads * headDim, rng: &rng))
        let vIn = try makeBuffer(device, values: randomHalfs(batch * kvHeads * headDim, rng: &rng))
        let qOut = try makeOutputBuffer(
            device, count: batch * numHeads * headDim, elementStride: 2)

        XCTAssertThrowsError(
            try context.timedDispatch { encoder in
                try prefill.encodeQKNormRoPEAppendBatch(
                    into: encoder, qIn: qIn, kIn: kIn, vIn: vIn,
                    qNormWeight: normBuffer, qNormByteOffset: 0,
                    kNormWeight: normBuffer, kNormByteOffset: 0,
                    cosTable: cosBuffer, sinTable: sinBuffer,
                    basePosition: 2, batch: batch,  // 2+3 > 4
                    positions: maxContext, numHeads: numHeads, eps: 1e-6,
                    cache: cache, layer: 0, qOut: qOut)
            }
        ) { error in
            XCTAssertEqual(
                error as? KVCacheError,
                .contextFull(position: 4, maxContext: 4))
        }
        let after = readHalfs(cache.buffer, count: cache.buffer.length / 2)
            .map(\.bitPattern)
        XCTAssertEqual(after, before, "cache must be untouched after the throw")
    }

    // MARK: - Copy-row: pure bitwise move

    func testCopyRowIsBitwiseIncludingAdversarialPatterns() throws {
        let context = try makeContextOrSkip()
        let device = context.device
        let count = 64
        var rng = SplitMix64(seed: 47)
        var values = randomHalfs(3 * count, rng: &rng)
        // Adversarial patterns in the copied row (row 1).
        values[count + 0] = Float16(bitPattern: 0x8000)   // -0.0
        values[count + 1] = Float16(bitPattern: 0x7E01)   // NaN payload
        values[count + 2] = Float16(bitPattern: 0x0001)   // subnormal
        let src = try makeBuffer(device, values: values)
        let dst = try makeOutputBuffer(device, count: count, elementStride: 2)

        let prefill = try PrefillKernels(context: context)
        try context.timedDispatch { encoder in
            try prefill.encodeCopyRow(
                into: encoder, source: src, row: 1, count: count, output: dst)
        }
        XCTAssertEqual(
            readHalfs(dst, count: count).map(\.bitPattern),
            values[count..<(2 * count)].map(\.bitPattern),
            "copy-row must preserve every bit pattern")
    }

    // MARK: - SDPA host offsets (P5-3 additive parameters)

    /// The offset form must produce bitwise the same output as offset-0 on
    /// pre-shifted data (host-side binding only — kernel untouched), and
    /// misaligned offsets are rejected pre-dispatch.
    func testSDPAOffsetsMatchOffsetZeroBitwiseAndRejectMisalignment() throws {
        let context = try makeContextOrSkip()
        let device = context.device
        var rng = SplitMix64(seed: 59)
        let numHeads = 4
        let kvHeads = 2
        let headDim = 16
        let maxContext = 8
        let position = 5
        let qDim = numHeads * headDim

        let cache = try KVCache(
            device: device, layers: 1, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)
        let cacheValues = randomHalfs(cache.buffer.length / 2, rng: &rng)
        cacheValues.withUnsafeBytes {
            cache.buffer.contents().copyMemory(
                from: $0.baseAddress!, byteCount: $0.count)
        }

        let query = randomHalfs(qDim, rng: &rng)
        let sdpa = try FusedSDPAKernel(context: context)

        // Reference at offset 0.
        let q0 = try makeBuffer(device, values: query)
        let out0 = try makeOutputBuffer(device, count: qDim, elementStride: 2)
        try context.timedDispatch { encoder in
            try sdpa.encodeSDPA(
                into: encoder, cache: cache, layer: 0, position: position,
                query: q0, numHeads: numHeads, output: out0)
        }

        // Same query as row 2 of a padded buffer, output into row 2.
        let padRows = 4
        var padded = randomHalfs(padRows * qDim, rng: &rng)
        padded.replaceSubrange((2 * qDim)..<(3 * qDim), with: query)
        let qPadded = try makeBuffer(device, values: padded)
        let outPadded = try makeOutputBuffer(
            device, count: padRows * qDim, elementStride: 2)
        try context.timedDispatch { encoder in
            try sdpa.encodeSDPA(
                into: encoder, cache: cache, layer: 0, position: position,
                query: qPadded, queryByteOffset: 2 * qDim * 2,
                numHeads: numHeads,
                output: outPadded, outputByteOffset: 2 * qDim * 2)
        }
        let ref = readHalfs(out0, count: qDim)
        let got = Array(readHalfs(outPadded, count: padRows * qDim)[(2 * qDim)..<(3 * qDim)])
        XCTAssertEqual(
            got.map(\.bitPattern), ref.map(\.bitPattern),
            "offset SDPA must be bitwise identical to offset-0 on the same row")

        // Misaligned offsets are rejected pre-dispatch.
        XCTAssertThrowsError(
            try context.timedDispatch { encoder in
                try sdpa.encodeSDPA(
                    into: encoder, cache: cache, layer: 0, position: position,
                    query: qPadded, queryByteOffset: 2,
                    numHeads: numHeads, output: outPadded)
            }
        ) { error in
            guard case QuantKernelError.misalignedOffset = error else {
                return XCTFail("expected misalignedOffset, got \(error)")
            }
        }
    }
}
