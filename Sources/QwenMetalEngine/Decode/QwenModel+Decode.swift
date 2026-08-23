import Foundation

extension QwenModel: NextTokenLogitsSource {
    public var vocabSize: Int { config.vocabSize }

    /// Full forward over the whole sequence (no KV cache — Phase 2), then
    /// final norm + lm_head on the LAST position only: positions are
    /// independent after the layer stack, so projecting one row is
    /// mathematically identical to slicing the full [seq, vocab] product,
    /// and the lm_head matmul still routes through the sgemm wrapper
    /// (hard rule 8).
    public func lastPositionLogits(ids: [Int]) throws -> [Float] {
        let hidden = config.hiddenSize
        let states = try hiddenStates(ids: ids)
        let lastRow = Array(states[((ids.count - 1) * hidden)...])
        let normed = try finalNorm(lastRow)
        return try lmHead(normed, seqLen: 1)
    }
}
