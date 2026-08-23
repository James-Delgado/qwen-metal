import Foundation

/// Errors thrown by `ModelConfig`. A missing required key names the key
/// (phase-0-1.md enumerated edge case 6).
public enum ModelConfigError: Error, Equatable, CustomStringConvertible {
    case cannotOpen(path: String, reason: String)
    case malformedJSON(detail: String)
    case missingKey(String)
    case invalidValue(key: String, detail: String)

    public var description: String {
        switch self {
        case .cannotOpen(let path, let reason):
            return "config: cannot open \(path): \(reason)"
        case .malformedJSON(let detail):
            return "config: malformed JSON: \(detail)"
        case .missingKey(let key):
            return "config: required key '\(key)' is missing"
        case .invalidValue(let key, let detail):
            return "config: invalid value for '\(key)': \(detail)"
        }
    }
}

/// Typed view of the checkpoint's config.json — exactly the fields the
/// Phase 1 CPU reference consumes (phase-0-1.md build step 2). Unknown keys
/// are ignored; missing required keys fail loudly by name.
public struct ModelConfig: Equatable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    /// Explicit in Qwen 3 configs; derived as hiddenSize / numAttentionHeads
    /// when absent (Qwen 2.5 configs omit it).
    public let headDim: Int
    public let intermediateSize: Int
    public let rmsNormEps: Double
    public let ropeTheta: Double
    public let vocabSize: Int
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int
    /// Family flag: Qwen 2.x projects QKV with biases; Qwen 3 does not.
    /// Honors an explicit `attention_bias` key, else defaults by family.
    public let attentionBias: Bool
    /// Family flag: Qwen 3 applies per-head RMSNorm to Q and K. HF configs
    /// carry no key for this — it follows from model_type (the architecture
    /// fork recorded in the PIN-1 DECISIONS.md entry).
    public let usesQKNorm: Bool
    /// Decode stop tokens from the optional `eos_token_id` key (an id or a
    /// list of ids in HF configs; the pinned checkpoint carries 151645).
    /// Empty when absent — the decode loop then only stops on max tokens.
    public let eosTokenIds: [Int]

    public static func load(path: String) throws -> ModelConfig {
        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: path))
        } catch {
            throw ModelConfigError.cannotOpen(path: path, reason: "\(error)")
        }
        return try ModelConfig(jsonData: data)
    }

    public init(jsonData: Data) throws {
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: jsonData)
        } catch {
            throw ModelConfigError.malformedJSON(detail: "\(error)")
        }
        guard let dict = parsed as? [String: Any] else {
            throw ModelConfigError.malformedJSON(detail: "top-level JSON value is not an object")
        }

        modelType = try Self.string(dict, "model_type")
        hiddenSize = try Self.positiveInt(dict, "hidden_size")
        numHiddenLayers = try Self.positiveInt(dict, "num_hidden_layers")
        numAttentionHeads = try Self.positiveInt(dict, "num_attention_heads")
        numKeyValueHeads = try Self.positiveInt(dict, "num_key_value_heads")
        intermediateSize = try Self.positiveInt(dict, "intermediate_size")
        rmsNormEps = try Self.positiveFiniteDouble(dict, "rms_norm_eps")
        ropeTheta = try Self.positiveFiniteDouble(dict, "rope_theta")
        vocabSize = try Self.positiveInt(dict, "vocab_size")
        tieWordEmbeddings = try Self.bool(dict, "tie_word_embeddings")
        maxPositionEmbeddings = try Self.positiveInt(dict, "max_position_embeddings")

        guard numAttentionHeads % numKeyValueHeads == 0 else {
            throw ModelConfigError.invalidValue(
                key: "num_key_value_heads",
                detail: "\(numKeyValueHeads) does not divide num_attention_heads "
                    + "\(numAttentionHeads) (GQA requires it)")
        }

        if let raw = dict["head_dim"], !(raw is NSNull) {
            headDim = try Self.positiveInt(dict, "head_dim")
        } else {
            guard hiddenSize % numAttentionHeads == 0 else {
                throw ModelConfigError.invalidValue(
                    key: "head_dim",
                    detail: "absent, and hidden_size \(hiddenSize) is not divisible "
                        + "by num_attention_heads \(numAttentionHeads)")
            }
            headDim = hiddenSize / numAttentionHeads
        }

        if let raw = dict["attention_bias"], !(raw is NSNull) {
            attentionBias = try Self.bool(dict, "attention_bias")
        } else if modelType == "qwen3" {
            attentionBias = false
        } else if modelType == "qwen2" {
            attentionBias = true // Qwen 2.x family ships QKV biases
        } else {
            // Unknown family: refuse to guess a load-bearing flag.
            throw ModelConfigError.missingKey("attention_bias")
        }
        usesQKNorm = (modelType == "qwen3")

        if let raw = dict["eos_token_id"], !(raw is NSNull) {
            eosTokenIds = try Self.intList(raw, key: "eos_token_id")
        } else {
            eosTokenIds = []
        }
    }

    // MARK: - Typed key access (missing keys name themselves, edge case 6)

    private static func present(_ dict: [String: Any], _ key: String) throws -> Any {
        guard let value = dict[key], !(value is NSNull) else {
            throw ModelConfigError.missingKey(key)
        }
        return value
    }

    private static func string(_ dict: [String: Any], _ key: String) throws -> String {
        guard let value = try present(dict, key) as? String else {
            throw ModelConfigError.invalidValue(key: key, detail: "expected a string")
        }
        return value
    }

    private static func bool(_ dict: [String: Any], _ key: String) throws -> Bool {
        guard let value = try present(dict, key) as? Bool else {
            throw ModelConfigError.invalidValue(key: key, detail: "expected a boolean")
        }
        return value
    }

    /// Both callers (`rms_norm_eps`, `rope_theta`) feed sqrt/pow radicands;
    /// zero, negative, or non-finite values reach the logits as silent NaN.
    private static func positiveFiniteDouble(_ dict: [String: Any], _ key: String) throws -> Double {
        guard let value = try present(dict, key) as? NSNumber else {
            throw ModelConfigError.invalidValue(key: key, detail: "expected a number")
        }
        let double = value.doubleValue
        guard double.isFinite, double > 0 else {
            throw ModelConfigError.invalidValue(
                key: key, detail: "expected a positive finite number, got \(value)")
        }
        return double
    }

    /// `eos_token_id` arrives as a single id or a list of ids in HF configs;
    /// normalize both to a list of non-negative integers.
    private static func intList(_ raw: Any, key: String) throws -> [Int] {
        let values: [Any] = (raw as? [Any]) ?? [raw]
        return try values.map { value in
            guard let number = value as? NSNumber,
                  let id = Int(exactly: number),
                  id >= 0 else {
                throw ModelConfigError.invalidValue(
                    key: key, detail: "expected a token id or list of token ids")
            }
            return id
        }
    }

    private static func positiveInt(_ dict: [String: Any], _ key: String) throws -> Int {
        guard let value = try present(dict, key) as? NSNumber else {
            throw ModelConfigError.invalidValue(key: key, detail: "expected a number")
        }
        // Int(exactly:) checks integrality and Int range without the Double
        // round-trip whose inclusive Int.max bound admitted exactly 2^63
        // (Double(Int.max) rounds up to it, and intValue wraps to Int.min).
        guard let int = Int(exactly: value), int >= 1 else {
            throw ModelConfigError.invalidValue(
                key: key, detail: "expected a positive integer, got \(value)")
        }
        return int
    }
}
