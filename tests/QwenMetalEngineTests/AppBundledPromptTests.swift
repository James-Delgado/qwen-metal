import XCTest

/// P2-6: the app bundles copies of the two pinned rendered prompts (D8
/// quick-load buttons). Copies can drift; the prompt strings are a parity
/// pin (benchmarks/prompts/README — changing one invalidates comparability).
/// This pins the bundled bytes BYTE-identical to the canonical files,
/// trailing newlines included (the CLI-1 `$(cat)` lesson: the decode-essay
/// rendered form ends in "\n\n" and losing it changes the token count).
final class AppBundledPromptTests: XCTestCase {

    private let promptNames = [
        "decode-essay.rendered.txt",
        "prefill-summarize.rendered.txt",
    ]

    func testBundledPromptsAreByteIdenticalToCanonicalRenderedPrompts() throws {
        let canonicalDir = SharedCheckpoint.repoRoot
            .appendingPathComponent("benchmarks/prompts/rendered")
        let bundledDir = SharedCheckpoint.repoRoot
            .appendingPathComponent("QwenMetalApp/Sources/Resources")

        for name in promptNames {
            let canonical = try Data(
                contentsOf: canonicalDir.appendingPathComponent(name))
            let bundled = try Data(
                contentsOf: bundledDir.appendingPathComponent(name))
            XCTAssertEqual(
                bundled, canonical,
                "\(name): app-bundled copy has drifted from "
                    + "benchmarks/prompts/rendered/ — re-copy it; never "
                    + "edit either side independently (parity pin)")
        }
    }

    func testDecodeEssayRenderedFormKeepsItsTrailingNewlines() throws {
        let bundled = try Data(contentsOf: SharedCheckpoint.repoRoot
            .appendingPathComponent(
                "QwenMetalApp/Sources/Resources/decode-essay.rendered.txt"))
        let text = try XCTUnwrap(String(data: bundled, encoding: .utf8))
        XCTAssertTrue(
            text.hasSuffix("\n\n"),
            "the pinned 84-token form ends with \\n\\n (CLI-1); a stripped "
                + "copy silently changes the prompt token count")
    }
}
