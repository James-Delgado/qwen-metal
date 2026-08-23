import Foundation

/// SwiGLU MLP: down( silu(gate(x)) * up(x) ), silu(x) = x·sigmoid(x), all in
/// fp32. The three projections route through BLAS.sgemm (hard rule 8); only
/// the elementwise gate is hand-rolled.
public struct SwiGLUMLP {
    /// HF stores projections as [out, in]; sgemm's transposeB consumes that
    /// directly, so no weight is ever transposed in memory.
    public let gateWeight: [Float] // [intermediateSize, hiddenSize]
    public let upWeight: [Float]   // [intermediateSize, hiddenSize]
    public let downWeight: [Float] // [hiddenSize, intermediateSize]
    public let hiddenSize: Int
    public let intermediateSize: Int

    public init(
        gateWeight: [Float], upWeight: [Float], downWeight: [Float],
        hiddenSize: Int, intermediateSize: Int
    ) throws {
        // Validated per tensor so a mismatch names the offending projection.
        for (name, weight, shape) in [
            ("gate_proj", gateWeight, [intermediateSize, hiddenSize]),
            ("up_proj", upWeight, [intermediateSize, hiddenSize]),
            ("down_proj", downWeight, [hiddenSize, intermediateSize]),
        ] {
            guard weight.count == shape[0] * shape[1] else {
                throw ModelError.badWeightShape(
                    tensor: name, expected: shape, actual: [weight.count])
            }
        }
        self.gateWeight = gateWeight
        self.upWeight = upWeight
        self.downWeight = downWeight
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
    }

    /// x is [seqLen, hiddenSize]; returns the same shape.
    public func callAsFunction(_ x: [Float], seqLen: Int) throws -> [Float] {
        let gate = try BLAS.sgemm(
            a: x, b: gateWeight, m: seqLen, k: hiddenSize, n: intermediateSize,
            transposeB: true)
        let up = try BLAS.sgemm(
            a: x, b: upWeight, m: seqLen, k: hiddenSize, n: intermediateSize,
            transposeB: true)
        var activated = [Float](repeating: 0, count: gate.count)
        for i in 0..<gate.count {
            let g = gate[i]
            activated[i] = (g / (1 + expf(-g))) * up[i]
        }
        return try BLAS.sgemm(
            a: activated, b: downWeight, m: seqLen, k: intermediateSize, n: hiddenSize,
            transposeB: true)
    }
}
