import Foundation

/// The CPU-quant reference front end (phase-3.md D3, PLAN invariant 4's
/// explicit carve-out): feeds the FROZEN CPU model fp32 weights materialized
/// by dequantizing the packed file. This materialization is a macOS test
/// oracle only — hard rule 1 (register-only dequant) continues to bind the
/// engine GPU path and the iOS app unqualified.
///
/// Both this path and the P3-4 GPU kernels evaluate the identical
/// single-rounded `q·scale + bias` (`Q4G64.dequant`), so their dequant
/// values are bit-identical by construction (the Phase 3 gates premise).
extension PackedCheckpoint: WeightSource {
    public func fp32Tensor(_ name: String, shape: [Int]) throws -> [Float] {
        if let dims = try? dims(for: name) {
            guard [dims.outDim, dims.inDim] == shape else {
                throw ModelError.badWeightShape(
                    tensor: name, expected: shape,
                    actual: [dims.outDim, dims.inDim])
            }
            return try dequantMatrix(name)
        }
        // Pass-through path: 1-D norm vectors stored as raw bf16 (schema D1)
        // ride the existing exact-upcast parser. An unknown name (including
        // lm_head.weight against an untied config — the packed artifact
        // stores the tied matrix once) fails with the parser's loud
        // tensorNotFound rather than silently substituting anything.
        let info = try file.info(for: name)
        guard info.shape == shape else {
            throw ModelError.badWeightShape(
                tensor: name, expected: shape, actual: info.shape)
        }
        return try file.fp32Values(for: name)
    }
}
