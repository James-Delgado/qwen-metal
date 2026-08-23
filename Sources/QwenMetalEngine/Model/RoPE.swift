import Foundation

/// Rotary position embedding in HF's Qwen3 form: half-split rotation
/// (rotate_half), applied AFTER the per-head Q/K RMSNorm (family order per
/// the PIN-1 DECISIONS.md entry). The angle tables mimic the HF fp32 path —
/// inv_freq computed in fp32, position·freq products in fp32 — so the only
/// divergence from the oracle is libm rounding, far below the phase gate.
public struct RoPE {
    public let headDim: Int
    public let positions: Int
    /// [positions, headDim/2] each; cos/sin of position·inv_freq.
    private let cosTable: [Float]
    private let sinTable: [Float]

    /// Read-only table access for the GPU decode path (Phase 2): the Metal
    /// RoPE kernel consumes THESE fp32 tables, so GPU angles are bit-identical
    /// to the oracle's — table drift can never be a divergence source.
    public var cosValues: [Float] { cosTable }
    public var sinValues: [Float] { sinTable }

    public init(headDim: Int, theta: Double, positions: Int) throws {
        guard headDim > 0, headDim % 2 == 0 else {
            throw ModelError.badInput(detail: "RoPE headDim \(headDim) must be positive and even")
        }
        guard positions > 0 else {
            throw ModelError.badInput(detail: "RoPE positions \(positions) must be positive")
        }
        self.headDim = headDim
        self.positions = positions

        let half = headDim / 2
        // HF: inv_freq = 1 / theta^(2i/dim), exponent and power in fp32.
        var invFreq = [Float](repeating: 0, count: half)
        for i in 0..<half {
            invFreq[i] = 1 / powf(Float(theta), Float(2 * i) / Float(headDim))
        }
        var cosTable = [Float](repeating: 0, count: positions * half)
        var sinTable = [Float](repeating: 0, count: positions * half)
        for position in 0..<positions {
            for i in 0..<half {
                let angle = Float(position) * invFreq[i]
                cosTable[position * half + i] = cosf(angle)
                sinTable[position * half + i] = sinf(angle)
            }
        }
        self.cosTable = cosTable
        self.sinTable = sinTable
    }

    /// Rotates every head of `x` in place. Layout [seqLen, heads·headDim];
    /// row s uses the table for absolute position s (full-sequence prefill —
    /// position offsets for incremental decode are Phase 2).
    public func apply(to x: inout [Float], seqLen: Int, heads: Int) throws {
        guard x.count == seqLen * heads * headDim else {
            throw ModelError.badInput(
                detail: "RoPE input count \(x.count) != seqLen \(seqLen) · heads \(heads) · headDim \(headDim)")
        }
        guard seqLen <= positions else {
            throw ModelError.badInput(
                detail: "RoPE sequence length \(seqLen) exceeds the \(positions)-position table")
        }
        let half = headDim / 2
        for s in 0..<seqLen {
            for h in 0..<heads {
                let base = (s * heads + h) * headDim
                for i in 0..<half {
                    let c = cosTable[s * half + i]
                    let sn = sinTable[s * half + i]
                    let x1 = x[base + i]
                    let x2 = x[base + half + i]
                    x[base + i] = x1 * c - x2 * sn
                    x[base + half + i] = x2 * c + x1 * sn
                }
            }
        }
    }
}
