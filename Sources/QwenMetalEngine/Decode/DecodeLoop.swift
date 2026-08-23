import Foundation

/// Anything that can produce full-vocab logits for the last position of a
/// token sequence. `QwenModel` is the real implementation; tests script this
/// to exercise stop conditions without a checkpoint.
public protocol NextTokenLogitsSource {
    var vocabSize: Int { get }
    /// Full-vocab fp32 logits for the LAST position of `ids`.
    func lastPositionLogits(ids: [Int]) throws -> [Float]
}

/// Errors from the decode loop's input validation (phase-0-1.md CLI edge
/// cases: empty prompt, prompt exceeding the context limit).
public enum DecodeError: Error, Equatable, CustomStringConvertible {
    case emptyPrompt
    case promptTooLong(promptTokens: Int, maxContext: Int)
    case invalidMaxNewTokens(Int)

    public var description: String {
        switch self {
        case .emptyPrompt:
            return "decode: prompt produced no tokens — provide a non-empty prompt"
        case .promptTooLong(let promptTokens, let maxContext):
            return "decode: prompt is \(promptTokens) tokens but the context "
                + "limit is \(maxContext) — it leaves no room to generate"
        case .invalidMaxNewTokens(let value):
            return "decode: max new tokens must be >= 1, got \(value)"
        }
    }
}

public enum Argmax {
    /// Index of the maximum value, first index winning ties — the fixture
    /// protocol's tie-break (torch.argmax, DECISIONS.md P1-1 entry). A plain
    /// strict-> scan gives exactly that.
    public static func firstIndex(_ values: [Float]) -> Int {
        precondition(!values.isEmpty, "argmax of an empty vector")
        var best = 0
        for i in 1..<values.count where values[i] > values[best] {
            best = i
        }
        return best
    }
}

/// Greedy decode over a full re-forward each step — incremental (KV-cached)
/// decode is Phase 2 (phase-0-1.md scope). Temperature sampling is not built
/// (spec marks it optional; PLAN.md scope discipline says don't).
public struct DecodeLoop {
    public let model: any NextTokenLogitsSource
    /// Hard bound on prompt + generated tokens (the Phase 1 CLI passes its
    /// 4K cap; it also bounds the RoPE table the model was built with).
    public let maxContext: Int

    public init(model: any NextTokenLogitsSource, maxContext: Int) {
        precondition(maxContext >= 1, "maxContext must be >= 1")
        self.model = model
        self.maxContext = maxContext
    }

    /// Greedy generation. Returns the generated ids only (EOS included when
    /// it fired). Stops on: a token in `eosTokenIds`, `maxNewTokens` reached,
    /// or the context filling up.
    /// - Parameter onStep: called per step with (stepIndex, fullVocabLogits,
    ///   chosenToken) — the logit-match suite and the CLI's verbose path hook
    ///   in here.
    public func generate(
        promptIds: [Int],
        maxNewTokens: Int,
        eosTokenIds: Set<Int> = [],
        onStep: ((Int, [Float], Int) -> Void)? = nil
    ) throws -> [Int] {
        guard !promptIds.isEmpty else { throw DecodeError.emptyPrompt }
        guard promptIds.count < maxContext else {
            throw DecodeError.promptTooLong(
                promptTokens: promptIds.count, maxContext: maxContext)
        }
        guard maxNewTokens >= 1 else {
            throw DecodeError.invalidMaxNewTokens(maxNewTokens)
        }

        var ids = promptIds
        var generated: [Int] = []
        for step in 0..<maxNewTokens {
            let logits = try model.lastPositionLogits(ids: ids)
            let token = Argmax.firstIndex(logits)
            generated.append(token)
            ids.append(token)
            onStep?(step, logits, token)
            if eosTokenIds.contains(token) { break }
            if ids.count >= maxContext { break }
        }
        return generated
    }
}
