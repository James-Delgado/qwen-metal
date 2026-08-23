import Foundation

/// One decoder layer, HF Qwen3 wiring:
///   h = x + attention(inputNorm(x))
///   out = h + mlp(postAttentionNorm(h))
public struct TransformerBlock {
    public let inputNorm: RMSNorm
    public let attention: Attention
    public let postAttentionNorm: RMSNorm
    public let mlp: SwiGLUMLP

    public init(
        inputNorm: RMSNorm, attention: Attention,
        postAttentionNorm: RMSNorm, mlp: SwiGLUMLP
    ) {
        self.inputNorm = inputNorm
        self.attention = attention
        self.postAttentionNorm = postAttentionNorm
        self.mlp = mlp
    }

    /// x is [seqLen, hiddenSize]; returns the same shape (residuals included —
    /// the layer0_block_output fixture hook point).
    public func callAsFunction(_ x: [Float], seqLen: Int) throws -> [Float] {
        let attentionOut = try attention(inputNorm(x), seqLen: seqLen)
        var h = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count {
            h[i] = x[i] + attentionOut[i]
        }
        let mlpOut = try mlp(postAttentionNorm(h), seqLen: seqLen)
        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<x.count {
            out[i] = h[i] + mlpOut[i]
        }
        return out
    }
}
