import XCTest
import QwenMetalEngine

/// P1-5 CLI plumbing tests: `--model-dir` resolution errors are clear and
/// name what is missing (phase-0-1.md: CLI edge inputs error, never crash).
final class ModelDirectoryTests: XCTestCase {

    private static let supportFiles = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
    ]

    private var tempDir: URL!

    override func setUpWithError() throws {
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelDirectoryTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: tempDir)
    }

    private func touch(_ names: [String]) throws {
        for name in names {
            try Data().write(to: tempDir.appendingPathComponent(name))
        }
    }

    func testValidLayoutResolvesCheckpointAndConfig() throws {
        try touch(Self.supportFiles + ["model.safetensors"])
        let directory = try ModelDirectory(validating: tempDir)
        XCTAssertEqual(directory.checkpointURL.lastPathComponent, "model.safetensors")
        XCTAssertEqual(directory.configURL.lastPathComponent, "config.json")
        XCTAssertEqual(directory.directoryURL.path, tempDir.path)
    }

    func testEachMissingSupportFileIsNamed() throws {
        for missing in Self.supportFiles {
            let caseDir = tempDir.appendingPathComponent(missing)
            try FileManager.default.createDirectory(
                at: caseDir, withIntermediateDirectories: true)
            for name in Self.supportFiles where name != missing {
                try Data().write(to: caseDir.appendingPathComponent(name))
            }
            try Data().write(to: caseDir.appendingPathComponent("model.safetensors"))
            XCTAssertThrowsError(try ModelDirectory(validating: caseDir)) { error in
                XCTAssertEqual(
                    error as? ModelDirectoryError,
                    .missingFile(name: missing, directory: caseDir.path))
            }
        }
    }

    // MARK: - (EOS-1) optional generation_config.json

    func testGenerationConfigURLNilWhenAbsent() throws {
        try touch(Self.supportFiles + ["model.safetensors"])
        let directory = try ModelDirectory(validating: tempDir)
        XCTAssertNil(
            directory.generationConfigURL,
            "generation_config.json is optional — absence must not be an error")
    }

    func testGenerationConfigURLResolvedWhenPresent() throws {
        try touch(Self.supportFiles + ["model.safetensors", "generation_config.json"])
        let directory = try ModelDirectory(validating: tempDir)
        XCTAssertEqual(
            directory.generationConfigURL?.lastPathComponent,
            "generation_config.json")
    }

    func testNoCheckpointThrows() throws {
        try touch(Self.supportFiles)
        XCTAssertThrowsError(try ModelDirectory(validating: tempDir)) { error in
            XCTAssertEqual(
                error as? ModelDirectoryError,
                .noCheckpoint(directory: tempDir.path))
        }
    }

    func testMultipleCheckpointsThrows() throws {
        try touch(Self.supportFiles + ["a.safetensors", "b.safetensors"])
        XCTAssertThrowsError(try ModelDirectory(validating: tempDir)) { error in
            XCTAssertEqual(
                error as? ModelDirectoryError,
                .multipleCheckpoints(
                    directory: tempDir.path,
                    found: ["a.safetensors", "b.safetensors"]))
        }
    }

    func testNonexistentPathThrows() {
        let missing = tempDir.appendingPathComponent("does-not-exist")
        XCTAssertThrowsError(try ModelDirectory(validating: missing)) { error in
            XCTAssertEqual(
                error as? ModelDirectoryError,
                .notADirectory(path: missing.path))
        }
    }

    func testPlainFileInsteadOfDirectoryThrows() throws {
        let file = tempDir.appendingPathComponent("just-a-file")
        try Data().write(to: file)
        XCTAssertThrowsError(try ModelDirectory(validating: file)) { error in
            XCTAssertEqual(
                error as? ModelDirectoryError,
                .notADirectory(path: file.path))
        }
    }
}
