import XCTest
import QwenMetalEngine

/// P1-5 decode-loop tests: enumerated edge cases 8 (EOS stop) and 9
/// (max-tokens stop) plus the CLI validations (empty prompt, prompt over the
/// context limit) — all against a scripted logits source, no checkpoint.
/// Temperature (case 10) is not built, so per the spec it gets no test.
final class DecodeLoopTests: XCTestCase {

    private struct ScriptedSource: NextTokenLogitsSource {
        let vocabSize: Int
        let script: ([Int]) -> [Float]
        func lastPositionLogits(ids: [Int]) throws -> [Float] { script(ids) }
    }

    /// Logits whose argmax is `index`.
    private func peaked(_ index: Int, vocab: Int = 4) -> [Float] {
        var values = [Float](repeating: 0, count: vocab)
        values[index] = 10
        return values
    }

    // MARK: - (8) EOS stop

    func testStopsOnEOSAndIncludesItInOutput() throws {
        let eos = 3
        // First step emits 1, every later step would emit eos forever.
        let source = ScriptedSource(vocabSize: 4) { ids in
            self.peaked(ids.count == 1 ? 1 : eos)
        }
        let generated = try DecodeLoop(model: source, maxContext: 100).generate(
            promptIds: [0], maxNewTokens: 10, eosTokenIds: [eos])
        XCTAssertEqual(generated, [1, eos], "must stop right after emitting EOS")
    }

    func testEOSIgnoredWhenNotRequested() throws {
        let source = ScriptedSource(vocabSize: 4) { _ in self.peaked(3) }
        let generated = try DecodeLoop(model: source, maxContext: 100).generate(
            promptIds: [0], maxNewTokens: 4, eosTokenIds: [])
        XCTAssertEqual(generated, [3, 3, 3, 3])
    }

    // MARK: - (9) max-tokens stop

    func testStopsAtMaxNewTokens() throws {
        let source = ScriptedSource(vocabSize: 4) { _ in self.peaked(1) }
        let generated = try DecodeLoop(model: source, maxContext: 100).generate(
            promptIds: [0], maxNewTokens: 3, eosTokenIds: [2])
        XCTAssertEqual(generated, [1, 1, 1])
    }

    func testStopsWhenContextFills() throws {
        let source = ScriptedSource(vocabSize: 4) { _ in self.peaked(1) }
        // Prompt of 3 in a context of 5 leaves room for exactly 2 tokens.
        let generated = try DecodeLoop(model: source, maxContext: 5).generate(
            promptIds: [0, 0, 0], maxNewTokens: 10, eosTokenIds: [])
        XCTAssertEqual(generated, [1, 1])
    }

    // MARK: - Input validation (CLI edge cases)

    func testEmptyPromptThrows() {
        let source = ScriptedSource(vocabSize: 4) { _ in self.peaked(1) }
        XCTAssertThrowsError(
            try DecodeLoop(model: source, maxContext: 100).generate(
                promptIds: [], maxNewTokens: 5)
        ) { error in
            XCTAssertEqual(error as? DecodeError, .emptyPrompt)
        }
    }

    func testPromptFillingTheContextThrows() {
        let source = ScriptedSource(vocabSize: 4) { _ in self.peaked(1) }
        let loop = DecodeLoop(model: source, maxContext: 4)
        // Exactly at the limit: no room to generate even one token.
        XCTAssertThrowsError(
            try loop.generate(promptIds: [0, 0, 0, 0], maxNewTokens: 5)
        ) { error in
            XCTAssertEqual(
                error as? DecodeError,
                .promptTooLong(promptTokens: 4, maxContext: 4))
        }
        // Over the limit.
        XCTAssertThrowsError(
            try loop.generate(promptIds: [0, 0, 0, 0, 0], maxNewTokens: 5)
        ) { error in
            XCTAssertEqual(
                error as? DecodeError,
                .promptTooLong(promptTokens: 5, maxContext: 4))
        }
    }

    func testNonpositiveMaxNewTokensThrows() {
        let source = ScriptedSource(vocabSize: 4) { _ in self.peaked(1) }
        XCTAssertThrowsError(
            try DecodeLoop(model: source, maxContext: 100).generate(
                promptIds: [0], maxNewTokens: 0)
        ) { error in
            XCTAssertEqual(error as? DecodeError, .invalidMaxNewTokens(0))
        }
    }

    // MARK: - Per-step callback (the logit suite's hook point)

    func testOnStepDeliversLogitsAndChosenTokens() throws {
        let source = ScriptedSource(vocabSize: 4) { ids in
            self.peaked(ids.count % 4)
        }
        var seen: [(step: Int, token: Int, logitCount: Int)] = []
        let generated = try DecodeLoop(model: source, maxContext: 100).generate(
            promptIds: [0], maxNewTokens: 3,
            onStep: { step, logits, token in
                seen.append((step, token, logits.count))
            })
        XCTAssertEqual(generated, [1, 2, 3])
        XCTAssertEqual(seen.map(\.step), [0, 1, 2])
        XCTAssertEqual(seen.map(\.token), generated)
        XCTAssertEqual(seen.map(\.logitCount), [4, 4, 4])
    }

    // MARK: - Argmax tie-break (fixture protocol: first index wins)

    func testArgmaxFirstIndexWinsTies() {
        XCTAssertEqual(Argmax.firstIndex([1, 5, 5, 2]), 1)
        XCTAssertEqual(Argmax.firstIndex([7, 7, 7]), 0)
        XCTAssertEqual(Argmax.firstIndex([-3, -1, -2]), 1)
        XCTAssertEqual(Argmax.firstIndex([2]), 0)
    }
}
