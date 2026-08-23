import CryptoKit
import XCTest
import QwenMetalEngine

/// P1-5 tokenizer equivalence (phase-0-1.md edge case 11): swift-transformers
/// must reproduce the committed Python tokenizer ids for all 5 fixture
/// prompts. On disagreement the rule is: log it, match Python.
///
/// Needs the local-only tokenizer files in models/ (downloaded once at the
/// pinned revision — see the DECISIONS.md P1-5 entry) and skips cleanly when
/// they are absent, like the checkpoint-backed suites.
final class TokenizerEquivalenceTests: XCTestCase {

    private static let promptNames = [
        "short_english", "multi_sentence", "code_snippet",
        "non_ascii", "chat_template",
    ]

    /// `<|im_end|>` in the pinned tokenizer_config.json.
    private static let pinnedEOSTokenId = 151645

    /// TOK-1 (docs/AUDIT.md F5): sha256 pins for the local-only tokenizer
    /// artifacts, mirroring SharedCheckpoint's source_revision pin. Recorded
    /// in the DECISIONS.md P1-5/TOK-1 entries; a drifted models/ file must
    /// fail loudly here, not pass because the 5 short prompts happen to
    /// tokenize identically.
    private static let pinnedArtifactSHA256 = [
        "tokenizer.json":
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "tokenizer_config.json":
            "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ]

    private static var shared: Result<TextTokenizer, Error>?

    override func setUpWithError() throws {
        for name in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
            let path = SharedCheckpoint.modelsDir.appendingPathComponent(name).path
            guard FileManager.default.fileExists(atPath: path) else {
                throw XCTSkip(
                    "models/\(name) not present (local-only — download it at the "
                    + "pinned revision \(SharedCheckpoint.pinnedRevision); see the "
                    + "DECISIONS.md P1-5 entry)")
            }
        }
        // Present but drifted is an error, never a skip: a wrong tokenizer
        // silently weakens the equivalence oracle.
        for (name, pinnedHash) in Self.pinnedArtifactSHA256 {
            let url = SharedCheckpoint.modelsDir.appendingPathComponent(name)
            let hash = SHA256.hash(data: try Data(contentsOf: url))
                .map { String(format: "%02x", $0) }.joined()
            guard hash == pinnedHash else {
                throw SharedCheckpoint.HelperError(description:
                    "models/\(name) sha256 \(hash) does not match the pinned "
                    + "\(pinnedHash) — the local artifact drifted from revision "
                    + "\(SharedCheckpoint.pinnedRevision); re-download it "
                    + "(see the DECISIONS.md P1-5 entry)")
            }
        }
    }

    private func loadedTokenizer() async throws -> TextTokenizer {
        if Self.shared == nil {
            do {
                Self.shared = .success(
                    try await TextTokenizer(modelFolder: SharedCheckpoint.modelsDir))
            } catch {
                Self.shared = .failure(error)
            }
        }
        return try Self.shared!.get()
    }

    /// (11) identical ids for every fixture prompt. chat_template tokenizes
    /// the RECORDED fully-rendered string (P1-1 DECISIONS entry) — Swift
    /// does no chat templating of its own in Phase 1.
    func testProducesPythonTokenizerIdsForAllFixturePrompts() async throws {
        let tokenizer = try await loadedTokenizer()
        for name in Self.promptNames {
            let fixture = try SharedCheckpoint.promptFixture(name)
            XCTAssertEqual(
                tokenizer.encode(fixture.inputText), fixture.inputIds,
                "prompt '\(name)': swift-transformers ids diverge from the Python dump")
        }
    }

    func testEOSTokenIdMatchesPinnedFamily() async throws {
        let tokenizer = try await loadedTokenizer()
        XCTAssertEqual(tokenizer.eosTokenId, Self.pinnedEOSTokenId)
    }

    /// Byte-level BPE decode must reproduce the raw prompt exactly — the CLI
    /// prints decoded text, so this guards its output path.
    func testDecodeRoundTripsRawPrompt() async throws {
        let tokenizer = try await loadedTokenizer()
        let fixture = try SharedCheckpoint.promptFixture("short_english")
        XCTAssertEqual(tokenizer.decode(fixture.inputIds), fixture.inputText)
    }
}
