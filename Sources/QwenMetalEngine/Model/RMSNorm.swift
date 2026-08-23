import Foundation

/// RMSNorm, matching HF's Qwen3RMSNorm: out = weight * (x * rsqrt(mean(x²) + eps)),
/// with the mean and normalization in fp32 (phase-0-1.md build step 4).
/// Also serves as the per-head Q/K norm of the pinned family, where a row is
/// one head's 128 values rather than a full hidden state.
public struct RMSNorm {
    public let weight: [Float]
    public let eps: Float
    public var dim: Int { weight.count }

    public init(weight: [Float], eps: Float) {
        self.weight = weight
        self.eps = eps
    }

    /// Normalizes each consecutive `dim`-length row of `x` independently.
    public func callAsFunction(_ x: [Float]) throws -> [Float] {
        guard dim > 0, !x.isEmpty, x.count % dim == 0 else {
            throw ModelError.badInput(
                detail: "RMSNorm input count \(x.count) is not a positive multiple of dim \(dim)")
        }
        var out = [Float](repeating: 0, count: x.count)
        for row in 0..<(x.count / dim) {
            let base = row * dim
            var sumOfSquares: Float = 0
            for j in 0..<dim {
                sumOfSquares += x[base + j] * x[base + j]
            }
            let inverseRMS = 1 / (sumOfSquares / Float(dim) + eps).squareRoot()
            for j in 0..<dim {
                out[base + j] = weight[j] * (x[base + j] * inverseRMS)
            }
        }
        return out
    }
}
