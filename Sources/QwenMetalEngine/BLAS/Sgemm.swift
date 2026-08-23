import Accelerate

/// The single validated matmul path for the CPU reference (hard rule 8):
/// QKV/MLP/lm_head projections AND per-head QK^T / PV products all route
/// through this one wrapper. Validated against a test-only naive triple loop
/// (SgemmTests, gate pre-committed in DECISIONS.md 2026-08-23) and
/// independently policed by the external HF fp32 oracle — a botched call
/// here cannot self-validate.
public enum BLAS {

    /// Computes C = op(A)·op(B) in fp32, where op(A) is m×k and op(B) is k×n.
    ///
    /// All matrices are row-major and contiguous. With `transposeA` set, `a`
    /// holds the k×m storage of op(A)^T; with `transposeB` set, `b` holds the
    /// n×k storage of op(B)^T — so per-head QK^T and the tied-embeddings
    /// lm_head never materialize a transposed copy. Element counts are m·k
    /// and k·n regardless of the flags.
    ///
    /// - Throws: `KernelInputError.nonPositiveDimensions` if any of m/k/n
    ///   is < 1; `KernelInputError.elementCountMismatch` if `a` or `b`
    ///   holds the wrong number of elements for the given dimensions.
    public static func sgemm(
        a: [Float],
        b: [Float],
        m: Int,
        k: Int,
        n: Int,
        transposeA: Bool = false,
        transposeB: Bool = false
    ) throws -> [Float] {
        guard m > 0, k > 0, n > 0 else {
            throw KernelInputError.nonPositiveDimensions(m: m, k: k, n: n)
        }
        guard a.count == m * k else {
            throw KernelInputError.elementCountMismatch(matrix: "A", expected: m * k, actual: a.count)
        }
        guard b.count == k * n else {
            throw KernelInputError.elementCountMismatch(matrix: "B", expected: k * n, actual: b.count)
        }

        // Row-major leading dimension = column count of the STORED matrix:
        // stored A is m×k, or k×m when transposed (likewise B is k×n / n×k).
        let lda = transposeA ? m : k
        let ldb = transposeB ? k : n

        var c = [Float](repeating: 0, count: m * n)
        cblas_sgemm(
            CblasRowMajor,
            transposeA ? CblasTrans : CblasNoTrans,
            transposeB ? CblasTrans : CblasNoTrans,
            Int32(m), Int32(n), Int32(k),
            1.0, a, Int32(lda),
            b, Int32(ldb),
            0.0, &c, Int32(n)
        )
        return c
    }
}
