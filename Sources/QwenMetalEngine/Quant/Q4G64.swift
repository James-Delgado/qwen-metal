import Foundation

/// Errors for the q4g64 packed-weight format, pack side and load side. Loud
/// and specific per METHODOLOGY: a bad packed file names what is wrong,
/// never crashes or silently mis-loads.
public enum Q4G64Error: Error, CustomStringConvertible {
    case nonFiniteWeight(tensor: String, index: Int)
    case groupRangeOverflow(tensor: String, groupIndex: Int)
    case inDimNotMultipleOfGroup(tensor: String, shape: [Int])
    case unsupportedRank(tensor: String, shape: [Int])
    case unsafeName(name: String)
    case inputAlreadyPacked(path: String)
    case missingSourceRevision(path: String)
    case notAPackedCheckpoint(path: String, detail: String)
    case revisionMismatch(path: String, found: String, expected: String)
    case badTriplet(name: String, detail: String)
    case misalignedQTensor(name: String, absoluteByteOffset: Int)

    public var description: String {
        switch self {
        case .nonFiniteWeight(let tensor, let index):
            return "q4g64: tensor '\(tensor)' has a non-finite weight at flat "
                + "index \(index) — the pinned checkpoint is finite; this is corruption"
        case .groupRangeOverflow(let tensor, let groupIndex):
            return "q4g64: tensor '\(tensor)' group \(groupIndex) has a value "
                + "range whose fp16 scale or bias is non-finite (|w| beyond fp16 range)"
        case .inDimNotMultipleOfGroup(let tensor, let shape):
            return "q4g64: tensor '\(tensor)' shape \(shape) has an input "
                + "(reduction) dimension that is not a positive multiple of "
                + "\(Q4G64.groupSize) — the q4g64 schema cannot represent it"
        case .unsupportedRank(let tensor, let shape):
            return "q4g64: tensor '\(tensor)' shape \(shape) is neither a 2-D "
                + "weight matrix nor a 1-D pass-through vector"
        case .unsafeName(let name):
            return "q4g64: name or metadata value '\(name)' contains characters "
                + "outside the safe ASCII subset for the hand-built JSON header"
        case .inputAlreadyPacked(let path):
            return "q4g64: \(path) is already a q4g64 packed checkpoint — the "
                + "packer input must be the bf16 consolidated checkpoint"
        case .missingSourceRevision(let path):
            return "q4g64: \(path) has no source_revision metadata — provenance "
                + "is required (consolidate with tools/consolidate_shards.py)"
        case .notAPackedCheckpoint(let path, let detail):
            return "q4g64: \(path) is not a q4g64 packed checkpoint: \(detail)"
        case .revisionMismatch(let path, let found, let expected):
            return "q4g64: \(path) has source_revision '\(found)', expected the "
                + "pinned \(expected)"
        case .badTriplet(let name, let detail):
            return "q4g64: packed tensor triplet '\(name)' is invalid: \(detail)"
        case .misalignedQTensor(let name, let absoluteByteOffset):
            return "q4g64: tensor '\(name)' sits at file offset "
                + "\(absoluteByteOffset), which is not 4-byte aligned — typed "
                + "u32 loads would be illegal (schema D1 requires offset ≡ 0 mod 4)"
        }
    }
}

/// The pinned q4g64 packed-weight schema and its arithmetic
/// (docs/phases/phase-3.md D1/D2; approved and PINNED by the DECISIONS.md
/// 2026-08-26 veto-close entry — changing anything here requires a new
/// DECISIONS.md entry).
///
/// Every 2-D weight matrix `W [out, in]` becomes three tensors:
///
///     {name}.q      u32 [out, in/8]   8 consecutive 4-bit codes per word;
///                                     element j of each group of 8 occupies
///                                     bits [4j, 4j+4) — low nibble first
///     {name}.scales f16 [out, in/64]  per-group scale
///     {name}.biases f16 [out, in/64]  per-group bias
///
/// with w ≈ scale·q + bias per group of 64 along the input (reduction)
/// dimension, q unsigned 0..15. 1-D norm vectors are not matrices — they
/// pass through unquantized (raw bf16), matching the MLX recipe's
/// linears+embeddings coverage (PLAN invariant 2 covers matrices).
public enum Q4G64 {
    public static let groupSize = 64
    public static let codesPerWord = 8
    /// u32 words per group of 64 codes.
    public static let wordsPerGroup = groupSize / codesPerWord

    public static let qSuffix = ".q"
    public static let scalesSuffix = ".scales"
    public static let biasesSuffix = ".biases"

    /// `__metadata__` keys in the packed file. `source_revision` matches the
    /// consolidated checkpoint's key so provenance chains end-to-end.
    public static let formatTag = "q4g64"
    public static let packerVersion = "1"
    public static let formatMetadataKey = "format"
    public static let groupSizeMetadataKey = "group_size"
    public static let packerVersionMetadataKey = "packer_version"
    public static let sourceRepoMetadataKey = "source_repo"
    public static let sourceRevisionMetadataKey = "source_revision"

    /// One packed group of 64 values.
    public struct Group {
        public let scale: Float16
        public let bias: Float16
        /// 64 codes, each 0...15.
        public let codes: [UInt8]
    }

    /// Packs one group of exactly 64 fp32 values per the AMENDED D2 recipe —
    /// zero-point-aligned selection (option A1, decided by James: DECISIONS.md
    /// 2026-08-30 QR-1 entry, superseding the original min/max selection
    /// after the P3-3 quality gate root-caused it):
    ///
    /// With group min m, max M, range R = M − m:
    ///   z = clamp(round(−m·15/R), 0, 15)          (integer zero-point)
    ///   s = max(M/(15−z) if z < 15, −m/z if z > 0) (both endpoints covered)
    ///   scale = fp16(s) FIRST, then bias = fp16(−z·scale)
    ///
    /// so the value 0 sits on the stored grid (exactly when z·scale is
    /// fp16-exact, else within ~2⁻¹¹·|bias| — the QR-1 fine print). Codes are
    /// then chosen against the fp16 values AS STORED, so dequant is exactly
    /// reproducible from the file with no hidden packer state.
    /// Degenerate group (max == min): scale = 0, all codes 0, w = bias.
    ///
    /// Rounding pin: every `round` here (z selection and
    /// `q = clamp(round((w − bias)/scale), 0, 15)`) is
    /// round-half-away-from-zero — the C/Metal `round()` semantics the
    /// P3-4 GPU kernels share.
    ///
    /// `elementOffset` is the group's flat start index within the source
    /// tensor, used only for error reporting.
    public static func packGroup(
        _ values: ArraySlice<Float>, tensor: String, elementOffset: Int
    ) throws -> Group {
        precondition(values.count == groupSize,
                     "packGroup needs exactly \(groupSize) values")
        var minValue = Float.infinity
        var maxValue = -Float.infinity
        for (i, v) in values.enumerated() {
            guard v.isFinite else {
                throw Q4G64Error.nonFiniteWeight(
                    tensor: tensor, index: elementOffset + i)
            }
            minValue = min(minValue, v)
            maxValue = max(maxValue, v)
        }
        let range = maxValue - minValue
        let scale: Float16
        let bias: Float16
        if range == 0 {
            scale = 0
            bias = Float16(minValue)
        } else {
            let zeroPoint = min(max(
                (-minValue * 15 / range).rounded(.toNearestOrAwayFromZero), 0), 15)
            var stretch: Float = 0
            if zeroPoint < 15 { stretch = max(stretch, maxValue / (15 - zeroPoint)) }
            if zeroPoint > 0 { stretch = max(stretch, -minValue / zeroPoint) }
            scale = Float16(stretch)
            bias = zeroPoint == 0 ? Float16(0) : Float16(-zeroPoint * Float(scale))
        }
        guard scale.isFinite, bias.isFinite else {
            throw Q4G64Error.groupRangeOverflow(
                tensor: tensor, groupIndex: elementOffset / groupSize)
        }

        var codes = [UInt8](repeating: 0, count: groupSize)
        let s = Float(scale)
        let b = Float(bias)
        if s != 0 {
            for (i, v) in values.enumerated() {
                let q = ((v - b) / s).rounded(.toNearestOrAwayFromZero)
                codes[i] = UInt8(min(max(q, 0), 15))
            }
        }
        return Group(scale: scale, bias: bias, codes: codes)
    }

    /// The dequant arithmetic shared verbatim by the CPU packer/oracle and
    /// (as Metal source, P3-4) the GPU kernels. q·scale is exact in fp32
    /// (a ≤4-bit integer times an 11-bit fp16 significand needs ≤15
    /// significand bits), so this is ONE correctly rounded operation and
    /// equals fma(q, scale, bias) bit-for-bit — the premise of the Phase 3
    /// exact gates (DECISIONS.md 2026-08-25 gates entry).
    public static func dequant(code: UInt32, scale: Float16, bias: Float16) -> Float {
        return Float(code) * Float(scale) + Float(bias)
    }

    /// Packs 64 codes into 8 u32 words, low nibble first (schema pin).
    public static func packWords(_ codes: [UInt8]) -> [UInt32] {
        precondition(codes.count == groupSize)
        var words = [UInt32](repeating: 0, count: wordsPerGroup)
        for (i, code) in codes.enumerated() {
            precondition(code <= 15, "4-bit code out of range: \(code)")
            words[i / codesPerWord] |= UInt32(code) << (4 * (i % codesPerWord))
        }
        return words
    }

    /// Extracts code `lane` (0..7) from a packed word — the read side of the
    /// nibble-order pin.
    public static func code(in word: UInt32, lane: Int) -> UInt32 {
        precondition((0..<codesPerWord).contains(lane))
        return (word >> (4 * lane)) & 0xF
    }
}
