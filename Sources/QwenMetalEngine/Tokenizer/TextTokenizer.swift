import Foundation
import Tokenizers

/// Thin adapter over swift-transformers' tokenizer (PLAN.md: borrow, don't
/// build). Loads ONLY from a local model folder (tokenizer.json +
/// tokenizer_config.json) — no Hub network access at runtime; the files are
/// downloaded once at the pinned revision, like the checkpoint.
///
/// Equivalence with the Python tokenizer is asserted against the committed
/// tokenizer_ids.json fixture (phase-0-1.md edge case 11). On disagreement
/// the rule is: log it, match Python.
public struct TextTokenizer {
    private let tokenizer: any Tokenizers.Tokenizer

    /// From tokenizer_config.json (`<|im_end|>` for the pinned family).
    public var eosTokenId: Int? { tokenizer.eosTokenId }

    public init(modelFolder: URL) async throws {
        tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
    }

    /// Token ids for `text`, with special tokens added — matching the
    /// fixture dump's Python default (the pinned family's template adds
    /// none for plain text, so this is the raw BPE encoding).
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    public func decode(_ ids: [Int], skipSpecialTokens: Bool = false) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: skipSpecialTokens)
    }
}
