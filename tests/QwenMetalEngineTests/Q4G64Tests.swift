import XCTest
import QwenMetalEngine

/// Arithmetic-level tests for the pinned q4g64 schema (phase-3.md D1/D2,
/// edge cases 1/3/4/5): nibble order, the fp16-round-first recipe, the
/// rounding-rule pin, degenerate/extreme/negative-heavy groups, the
/// per-element round-trip bound, and determinism. File-level packing is
/// Q4PackerTests.
final class Q4G64Tests: XCTestCase {

    private func group(_ values: [Float], offset: Int = 0) throws -> Q4G64.Group {
        try Q4G64.packGroup(values[...], tensor: "test.weight", elementOffset: offset)
    }

    // MARK: - Edge case 1: nibble-order pin

    func testPackWordsIsLowNibbleFirst() {
        var codes = [UInt8](repeating: 0, count: 64)
        for i in 0..<16 { codes[i] = UInt8(i) }
        let words = Q4G64.packWords(codes)
        XCTAssertEqual(words.count, 8)
        // Element j occupies bits [4j, 4j+4): codes 0..7 -> 0x76543210.
        XCTAssertEqual(words[0], 0x7654_3210)
        XCTAssertEqual(words[1], 0xFEDC_BA98)
        XCTAssertEqual(words[2], 0)
    }

    func testCodeExtractionMatchesHandBuiltWord() {
        let word: UInt32 = 0x7654_3210
        for lane in 0..<8 {
            XCTAssertEqual(Q4G64.code(in: word, lane: lane), UInt32(lane))
        }
        // Asymmetric pattern: a swapped-nibble reader is a hard mismatch.
        let asymmetric: UInt32 = 0xF000_0003
        XCTAssertEqual(Q4G64.code(in: asymmetric, lane: 0), 3)
        XCTAssertEqual(Q4G64.code(in: asymmetric, lane: 7), 15)
        for lane in 1...6 {
            XCTAssertEqual(Q4G64.code(in: asymmetric, lane: lane), 0)
        }
    }

    func testDequantMatchesHandComputedValues() {
        let scale: Float16 = 2.0
        let bias: Float16 = -1.0
        XCTAssertEqual(Q4G64.dequant(code: 0, scale: scale, bias: bias), -1.0)
        XCTAssertEqual(Q4G64.dequant(code: 3, scale: scale, bias: bias), 5.0)
        XCTAssertEqual(Q4G64.dequant(code: 15, scale: scale, bias: bias), 29.0)
    }

    // MARK: - The D2 recipe: fp16-round-FIRST, codes chosen against stored values

    func testCodesAreChosenAgainstFP16RoundedScale() throws {
        // min 0, max 1 -> true scale 1/15 = 0.066666667; fp16 rounds it DOWN
        // to 0.066650390625. 0.4999/0.06665039 = 7.5018 -> code 8, while the
        // unrounded scale would give 7.4985 -> code 7. Pins the round-first
        // order of the recipe.
        var values = [Float](repeating: 0, count: 64)
        values[61] = 0.49
        values[62] = 0.4999
        values[63] = 1.0
        let g = try group(values)
        XCTAssertEqual(g.scale, Float16(0.066650390625))
        XCTAssertEqual(g.bias, 0)
        XCTAssertEqual(g.codes[0], 0)
        XCTAssertEqual(g.codes[61], 7)
        XCTAssertEqual(g.codes[62], 8)
        XCTAssertEqual(g.codes[63], 15)
    }

    func testRoundingRuleIsHalfAwayFromZero() throws {
        // scale = 30/15 = 2 exactly; (1 - 0)/2 = 0.5 sits on the boundary:
        // half-away rounds to 1 (nearest-even would give 0).
        var values = [Float](repeating: 0, count: 64)
        values[1] = 1.0
        values[2] = 30.0
        let g = try group(values)
        XCTAssertEqual(g.scale, 2.0)
        XCTAssertEqual(g.codes[1], 1)
        XCTAssertEqual(g.codes[2], 15)
    }

    // MARK: - Amended D2 (QR-1, DECISIONS.md 2026-08-30): zero-point alignment

    func testZeroStraddlingGroupIsZeroPointAligned() throws {
        // m = -1, M = 3, R = 4: z = round(15/4) = round(3.75) = 4 (half-away).
        // s = max(3/11, 1/4) = 3/11; fp16(3/11) = 2234·2⁻¹³ = 0.272705078125.
        // b = -4·s16 = -1117·2⁻¹⁰ = -1.0908203125 — EXACTLY representable in
        // fp16, so grid-zero is exact: the 62 zero elements must code to the
        // zero-point and dequant to 0.0 bit-exactly (the whole point of A1).
        var values = [Float](repeating: 0, count: 64)
        values[0] = -1
        values[63] = 3
        let g = try group(values)
        XCTAssertEqual(g.scale, Float16(0.272705078125))
        XCTAssertEqual(g.bias, Float16(-1.0908203125))
        XCTAssertEqual(g.codes[0], 0)   // m: (-1 + 1.0908)/s = 0.333 -> 0
        XCTAssertEqual(g.codes[63], 15) // M: 15.0009 -> 15
        for i in 1...62 {
            XCTAssertEqual(g.codes[i], 4, "zero element \(i) must sit on the zero-point")
            XCTAssertEqual(
                Q4G64.dequant(code: UInt32(g.codes[i]), scale: g.scale, bias: g.bias),
                0.0, "grid-zero must reconstruct exactly here (b16 = -z·s16 is fp16-exact)")
        }
    }

    func testAllPositiveGroupAnchorsGridAtZero() throws {
        // m = 0.5 > 0 -> z clamps to 0, bias = +0 exactly, s = M/15.
        // Grid runs [0, M]; the min codes interior (5), not to 0.
        var values = [Float](repeating: 1.0, count: 64)
        values[0] = 0.5
        values[63] = 1.5
        let g = try group(values)
        XCTAssertEqual(g.scale, Float16(1.5 / 15))
        XCTAssertEqual(g.bias, 0)
        XCTAssertEqual(g.bias.sign, .plus, "z = 0 pins bias to +0, not -0")
        XCTAssertEqual(g.codes[0], 5)   // 0.5/0.0999756 = 5.001 -> 5
        XCTAssertEqual(g.codes[63], 15)
    }

    func testAllNegativeGroupAnchorsGridTopNearZero() throws {
        // m = -2, M = -1 -> z clamps to 15, s = -m/15, bias = fp16(-15·s16).
        // Grid top (code 15) lands within the documented fp16-bias slack of
        // zero (QR-1 fine print: |top| <= ~2⁻¹¹·|bias|), NOT at M.
        var values = [Float](repeating: -1.0, count: 64)
        values[0] = -2
        let g = try group(values)
        XCTAssertEqual(g.scale, Float16(0.13330078125)) // fp16(2/15)
        XCTAssertEqual(g.bias, Float16(-2.0))           // fp16(-15·s16) rounds to -2
        XCTAssertEqual(g.codes[0], 0)
        for i in 1...63 {
            XCTAssertEqual(g.codes[i], 8, "-1 sits at (1.0/s16) = 7.5018 -> 8")
        }
        let top = Q4G64.dequant(code: 15, scale: g.scale, bias: g.bias)
        XCTAssertLessThanOrEqual(abs(top), 2 * Float(0x1p-11) * abs(Float(g.bias)),
                                 "grid top must be zero-anchored within fp16 bias slack")
    }

    // MARK: - Edge case 3: degenerate group

    func testDegenerateGroupHasScaleZeroAndDequantsToBias() throws {
        let values = [Float](repeating: 3.14159, count: 64)
        let g = try group(values)
        XCTAssertEqual(g.scale, 0)
        XCTAssertEqual(g.bias, Float16(3.14159))
        XCTAssertTrue(g.codes.allSatisfy { $0 == 0 })
        XCTAssertEqual(
            Q4G64.dequant(code: 0, scale: g.scale, bias: g.bias),
            Float(Float16(3.14159)))
    }

    // MARK: - Edge case 4: extreme scales dequant exactly (single rounding)

    func testDequantIsSingleRoundedAtExtremeScales() {
        // q*scale is exact in fp32, so dequant must equal the correctly
        // rounded fp32 of the exact Double computation for EVERY pattern —
        // max-normal, min-normal, subnormal, negative bias included.
        let cases: [(Float16, Float16)] = [
            (65504, -65504),                         // fp16 max normal
            (Float16.leastNormalMagnitude, 1.0),     // 2^-14
            (Float16.leastNonzeroMagnitude, -Float16.leastNonzeroMagnitude), // 2^-24 subnormal
            (Float16(0.1), Float16(-0.3)),           // inexact decimals
        ]
        for (scale, bias) in cases {
            for q in [UInt32(0), 1, 7, 15] {
                let expected = Float(
                    Double(q) * Double(Float(scale)) + Double(Float(bias)))
                XCTAssertEqual(
                    Q4G64.dequant(code: q, scale: scale, bias: bias), expected,
                    "q=\(q) scale=\(scale) bias=\(bias)")
            }
        }
    }

    // MARK: - Edge cases 4/5: negative-heavy + round-trip bound + determinism

    /// s/2 (half quantization step) + clamp-edge slack from rounding scale
    /// to fp16 (≤ 15·s·2⁻¹¹) + bias fp16 rounding (≤ ulp/2) + fp32 noise.
    private func roundTripBound(_ g: Q4G64.Group) -> Float {
        let s = Float(g.scale)
        return s / 2 + 15 * s * Float(0x1p-11) + Float(g.bias.ulp) / 2 + 2e-6
    }

    func testNegativeHeavyGroupRoundTrips() throws {
        let values = (0..<64).map { -5.0 + 0.07 * Float($0) }
        let g = try group(values)
        XCTAssertLessThan(Float(g.bias), 0)
        let bound = roundTripBound(g)
        for (v, code) in zip(values, g.codes) {
            let dq = Q4G64.dequant(code: UInt32(code), scale: g.scale, bias: g.bias)
            XCTAssertLessThanOrEqual(abs(v - dq), bound, "value \(v)")
        }
    }

    func testRoundTripBoundAndDeterminismOnRandomGroups() throws {
        // Deterministic LCG; values in [-8, 8], the magnitude regime of
        // real transformer weights.
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24) * 16 - 8
        }
        for trial in 0..<32 {
            let values = (0..<64).map { _ in next() }
            let g = try group(values)
            let bound = roundTripBound(g)
            for (v, code) in zip(values, g.codes) {
                let dq = Q4G64.dequant(code: UInt32(code), scale: g.scale, bias: g.bias)
                XCTAssertLessThanOrEqual(abs(v - dq), bound, "trial \(trial) value \(v)")
            }
            // Same input -> bit-identical group (no hidden state).
            let again = try group(values)
            XCTAssertEqual(g.scale.bitPattern, again.scale.bitPattern)
            XCTAssertEqual(g.bias.bitPattern, again.bias.bitPattern)
            XCTAssertEqual(g.codes, again.codes)
        }
    }

    // MARK: - Non-finite and overflow rejection

    func testNonFiniteWeightThrowsWithFlatIndex() {
        var values = [Float](repeating: 0.5, count: 64)
        values[10] = .nan
        XCTAssertThrowsError(
            try Q4G64.packGroup(values[...], tensor: "t", elementOffset: 128)
        ) { error in
            guard case Q4G64Error.nonFiniteWeight(let tensor, let index) = error else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertEqual(tensor, "t")
            XCTAssertEqual(index, 138)
        }
        values[10] = .infinity
        XCTAssertThrowsError(
            try Q4G64.packGroup(values[...], tensor: "t", elementOffset: 0))
    }

    func testScaleOverflowThrows() {
        // Range 2e6 -> scale ~133333 overflows fp16 (max 65504).
        var values = [Float](repeating: 0, count: 64)
        values[0] = -1e6
        values[1] = 1e6
        XCTAssertThrowsError(try group(values)) { error in
            guard case Q4G64Error.groupRangeOverflow = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testBiasOverflowThrows() {
        // Tiny range (finite fp16 scale) but |min| beyond fp16 range.
        let values = (0..<64).map { -1e6 + 0.001 * Float($0) }
        XCTAssertThrowsError(try group(values)) { error in
            guard case Q4G64Error.groupRangeOverflow = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }
}
