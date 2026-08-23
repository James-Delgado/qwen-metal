import Foundation

/// Typed view of the checkpoint's optional generation_config.json — exactly
/// the one field decode consumes. HF `generate()` consults this file for
/// stopping, so its `eos_token_id` list belongs in the decode stop set: the
/// pinned checkpoint lists [151645, 151643] here while config.json carries
/// only 151645 (docs/AUDIT.md F2). Sampling keys (temperature, top_p, ...)
/// are ignored — greedy decode is a benchmark protocol pin.
///
/// The file is optional (`ModelDirectory` reports it as such); a present but
/// unreadable/malformed file fails loudly via `ModelConfigError`.
public struct GenerationConfig: Equatable {
    /// Stop ids from the optional `eos_token_id` key (an id or a list of
    /// ids, the same shape config.json uses). Empty when the key is absent.
    public let eosTokenIds: [Int]

    public static func load(path: String) throws -> GenerationConfig {
        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: path))
        } catch {
            throw ModelConfigError.cannotOpen(path: path, reason: "\(error)")
        }
        return try GenerationConfig(jsonData: data)
    }

    public init(jsonData: Data) throws {
        let parsed: Any
        do {
            parsed = try JSONSerialization.jsonObject(with: jsonData)
        } catch {
            throw ModelConfigError.malformedJSON(detail: "\(error)")
        }
        guard let dict = parsed as? [String: Any] else {
            throw ModelConfigError.malformedJSON(
                detail: "top-level JSON value is not an object")
        }

        if let raw = dict["eos_token_id"], !(raw is NSNull) {
            eosTokenIds = try ModelConfig.intList(raw, key: "eos_token_id")
        } else {
            eosTokenIds = []
        }
    }
}
