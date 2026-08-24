import XCTest
@testable import QwenMetalEngine
import Metal

/// P2-3 cache-object tests (docs/phases/phase-2.md D3, hard rule 4): size
/// formula, one-shot preallocation at the pinned model's real dims (448 MiB),
/// layout/offset arithmetic, and loud bounds errors.
final class KVCacheTests: XCTestCase {

    private func makeDeviceOrSkip() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available on this machine")
        }
        return device
    }

    // MARK: - Size formula (no device needed)

    func testByteCountAtPinnedModelDimsIsExactly448MiB() throws {
        // 28 layers × (K|V) × 8 KV heads × 4096 positions × 128 dim × 2 B.
        let bytes = try KVCache.byteCount(
            layers: 28, kvHeads: 8, maxContext: 4096, headDim: 128)
        XCTAssertEqual(bytes, 448 * 1024 * 1024)
    }

    func testByteCountRejectsNonPositiveDims() {
        for (name, dims) in [
            ("layers", (0, 8, 4096, 128)),
            ("kvHeads", (28, -1, 4096, 128)),
            ("maxContext", (28, 8, 0, 128)),
            ("headDim", (28, 8, 4096, 0)),
        ] {
            XCTAssertThrowsError(
                try KVCache.byteCount(
                    layers: dims.0, kvHeads: dims.1,
                    maxContext: dims.2, headDim: dims.3)
            ) { error in
                guard case KVCacheError.nonPositiveDimension(let thrownName, _)? =
                    error as? KVCacheError
                else {
                    return XCTFail("expected nonPositiveDimension, got \(error)")
                }
                XCTAssertEqual(thrownName, name)
            }
        }
    }

    func testByteCountRejectsIntOverflowLoudly() {
        XCTAssertThrowsError(
            try KVCache.byteCount(
                layers: Int.max / 2, kvHeads: Int.max / 2,
                maxContext: 4096, headDim: 128)
        ) { error in
            guard case KVCacheError.byteCountOverflow? = error as? KVCacheError
            else {
                return XCTFail("expected byteCountOverflow, got \(error)")
            }
        }
    }

    // MARK: - Allocation (hard rule 4: full size at init, never grown)

    func testInitPreallocatesTheFull448MiBBufferAtRealDims() throws {
        let device = try makeDeviceOrSkip()
        let cache = try KVCache(
            device: device, layers: 28, kvHeads: 8, maxContext: 4096, headDim: 128)
        XCTAssertEqual(cache.byteCount, 448 * 1024 * 1024)
        XCTAssertEqual(cache.buffer.length, 448 * 1024 * 1024)
    }

    // MARK: - Layout arithmetic (head-major: positions contiguous per head)

    func testElementOffsetMatchesTheDeclaredLayout() throws {
        let device = try makeDeviceOrSkip()
        let (layers, kvHeads, maxContext, headDim) = (3, 2, 5, 4)
        let cache = try KVCache(
            device: device, layers: layers, kvHeads: kvHeads,
            maxContext: maxContext, headDim: headDim)

        // [layers][K|V][kvHeads][maxContext][headDim], fp16 elements.
        func expected(_ l: Int, _ c: Int, _ h: Int, _ p: Int) -> Int {
            (((l * 2 + c) * kvHeads + h) * maxContext + p) * headDim
        }
        for l in 0..<layers {
            for (c, component) in [KVCache.Component.key, .value].enumerated() {
                for h in 0..<kvHeads {
                    for p in 0..<maxContext {
                        XCTAssertEqual(
                            try cache.elementOffset(
                                layer: l, component: component, head: h, position: p),
                            expected(l, c, h, p),
                            "offset (\(l), \(c), \(h), \(p))")
                    }
                }
            }
        }

        // The properties the attention kernels rely on, stated directly:
        // consecutive positions of one head are headDim apart (streaming read)…
        let p0 = try cache.elementOffset(layer: 1, component: .key, head: 1, position: 0)
        let p1 = try cache.elementOffset(layer: 1, component: .key, head: 1, position: 1)
        XCTAssertEqual(p1 - p0, headDim)
        // …K precedes V within a layer, and layers are 2·kvHeads·ctx·dim apart.
        let k0 = try cache.elementOffset(layer: 0, component: .key, head: 0, position: 0)
        let v0 = try cache.elementOffset(layer: 0, component: .value, head: 0, position: 0)
        XCTAssertEqual(v0 - k0, kvHeads * maxContext * headDim)
        let l1 = try cache.elementOffset(layer: 1, component: .key, head: 0, position: 0)
        XCTAssertEqual(l1 - k0, 2 * kvHeads * maxContext * headDim)
        // The last slot's final element is the buffer's last element.
        let last = try cache.elementOffset(
            layer: layers - 1, component: .value,
            head: kvHeads - 1, position: maxContext - 1)
        XCTAssertEqual((last + headDim) * 2, cache.byteCount)
    }

    func testElementOffsetRejectsOutOfRangeIndices() throws {
        let device = try makeDeviceOrSkip()
        let cache = try KVCache(
            device: device, layers: 2, kvHeads: 2, maxContext: 3, headDim: 4)

        for (name, layer, head, position) in [
            ("layer", 2, 0, 0), ("layer", -1, 0, 0),
            ("head", 0, 2, 0), ("head", 0, -1, 0),
            ("position", 0, 0, 3), ("position", 0, 0, -1),
        ] {
            XCTAssertThrowsError(
                try cache.elementOffset(
                    layer: layer, component: .key, head: head, position: position)
            ) { error in
                guard case KVCacheError.indexOutOfRange(let thrownName, _, _)? =
                    error as? KVCacheError
                else {
                    return XCTFail("expected indexOutOfRange, got \(error)")
                }
                XCTAssertEqual(thrownName, name)
            }
        }
    }
}
