import Foundation

/// Qwen3 GQA attention over the full sequence (incremental decode with a KV
/// cache is Phase 2). Family-correct per the PIN-1 entry: NO QKV biases;
/// per-head Q/K RMSNorm applied before RoPE. All matmul-shaped work — the
/// four projections AND the per-head QK^T / PV products — routes through
/// BLAS.sgemm (hard rule 8). Softmax is fp32 with max-subtraction.
public struct Attention {
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let hiddenSize: Int

    /// HF [out, in] storage, consumed via sgemm's transposeB.
    public let qProjWeight: [Float] // [numHeads·headDim, hiddenSize]
    public let kProjWeight: [Float] // [numKVHeads·headDim, hiddenSize]
    public let vProjWeight: [Float] // [numKVHeads·headDim, hiddenSize]
    public let oProjWeight: [Float] // [hiddenSize, numHeads·headDim]
    public let qNorm: RMSNorm       // over headDim
    public let kNorm: RMSNorm       // over headDim
    public let rope: RoPE

    public init(
        numHeads: Int, numKVHeads: Int, headDim: Int, hiddenSize: Int,
        qProjWeight: [Float], kProjWeight: [Float], vProjWeight: [Float],
        oProjWeight: [Float], qNorm: RMSNorm, kNorm: RMSNorm, rope: RoPE
    ) throws {
        guard numHeads > 0, numKVHeads > 0, numHeads % numKVHeads == 0 else {
            throw ModelError.badInput(
                detail: "GQA requires numKVHeads \(numKVHeads) to divide numHeads \(numHeads)")
        }
        // Validated per tensor so a mismatch names the offending projection.
        try Self.validate("q_proj", qProjWeight, shape: [numHeads * headDim, hiddenSize])
        try Self.validate("k_proj", kProjWeight, shape: [numKVHeads * headDim, hiddenSize])
        try Self.validate("v_proj", vProjWeight, shape: [numKVHeads * headDim, hiddenSize])
        try Self.validate("o_proj", oProjWeight, shape: [hiddenSize, numHeads * headDim])
        guard qNorm.dim == headDim, kNorm.dim == headDim else {
            throw ModelError.badWeightShape(
                tensor: "q_norm/k_norm", expected: [headDim],
                actual: [qNorm.dim, kNorm.dim])
        }
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.hiddenSize = hiddenSize
        self.qProjWeight = qProjWeight
        self.kProjWeight = kProjWeight
        self.vProjWeight = vProjWeight
        self.oProjWeight = oProjWeight
        self.qNorm = qNorm
        self.kNorm = kNorm
        self.rope = rope
    }

    private static func validate(
        _ name: String, _ weight: [Float], shape: [Int]
    ) throws {
        guard weight.count == shape[0] * shape[1] else {
            throw ModelError.badWeightShape(
                tensor: name, expected: shape, actual: [weight.count])
        }
    }

    /// x is [seqLen, hiddenSize]; returns the same shape (o_proj output,
    /// before the residual add — the layer0_attn_output fixture hook point).
    public func callAsFunction(_ x: [Float], seqLen: Int) throws -> [Float] {
        guard seqLen > 0, x.count == seqLen * hiddenSize else {
            throw ModelError.badInput(
                detail: "attention input count \(x.count) != seqLen \(seqLen) · hidden \(hiddenSize)")
        }
        // Projections: [seqLen, heads·headDim], head-major within each row.
        var q = try BLAS.sgemm(
            a: x, b: qProjWeight, m: seqLen, k: hiddenSize, n: numHeads * headDim,
            transposeB: true)
        var k = try BLAS.sgemm(
            a: x, b: kProjWeight, m: seqLen, k: hiddenSize, n: numKVHeads * headDim,
            transposeB: true)
        let v = try BLAS.sgemm(
            a: x, b: vProjWeight, m: seqLen, k: hiddenSize, n: numKVHeads * headDim,
            transposeB: true)

        // Family order: per-head RMSNorm on Q and K, THEN RoPE.
        q = try qNorm(q) // each consecutive headDim run is one head's row
        k = try kNorm(k)
        try rope.apply(to: &q, seqLen: seqLen, heads: numHeads)
        try rope.apply(to: &k, seqLen: seqLen, heads: numKVHeads)

        let scale = 1 / Float(headDim).squareRoot()
        let groupSize = numHeads / numKVHeads
        var context = [Float](repeating: 0, count: seqLen * numHeads * headDim)
        var qHead = [Float](repeating: 0, count: seqLen * headDim)
        var kHead = [Float](repeating: 0, count: seqLen * headDim)
        var vHead = [Float](repeating: 0, count: seqLen * headDim)

        for head in 0..<numHeads {
            let kvHead = head / groupSize // GQA: consecutive Q heads share a KV head
            for s in 0..<seqLen {
                for d in 0..<headDim {
                    qHead[s * headDim + d] = q[(s * numHeads + head) * headDim + d]
                    kHead[s * headDim + d] = k[(s * numKVHeads + kvHead) * headDim + d]
                    vHead[s * headDim + d] = v[(s * numKVHeads + kvHead) * headDim + d]
                }
            }

            // scores = Q·K^T / sqrt(d), causal, softmax rows in fp32.
            var scores = try BLAS.sgemm(
                a: qHead, b: kHead, m: seqLen, k: headDim, n: seqLen, transposeB: true)
            for i in 0..<seqLen {
                let row = i * seqLen
                var rowMax = -Float.infinity
                for j in 0...i {
                    scores[row + j] *= scale
                    rowMax = max(rowMax, scores[row + j])
                }
                var sum: Float = 0
                for j in 0...i {
                    let e = expf(scores[row + j] - rowMax)
                    scores[row + j] = e
                    sum += e
                }
                for j in 0...i {
                    scores[row + j] /= sum
                }
                for j in (i + 1)..<seqLen { // causal: no attention to the future
                    scores[row + j] = 0
                }
            }

            let headContext = try BLAS.sgemm(
                a: scores, b: vHead, m: seqLen, k: seqLen, n: headDim)
            for s in 0..<seqLen {
                for d in 0..<headDim {
                    context[(s * numHeads + head) * headDim + d] =
                        headContext[s * headDim + d]
                }
            }
        }

        return try BLAS.sgemm(
            a: context, b: oProjWeight, m: seqLen, k: numHeads * headDim, n: hiddenSize,
            transposeB: true)
    }
}
