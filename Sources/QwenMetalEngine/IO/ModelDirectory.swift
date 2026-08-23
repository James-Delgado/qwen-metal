import Foundation

/// Errors from resolving a `--model-dir` layout. Every case names what is
/// missing and where it looked (CLI edge inputs get clear errors, not
/// crashes — phase-0-1.md enumerated case list).
public enum ModelDirectoryError: Error, Equatable, CustomStringConvertible {
    case notADirectory(path: String)
    case missingFile(name: String, directory: String)
    case noCheckpoint(directory: String)
    case multipleCheckpoints(directory: String, found: [String])

    public var description: String {
        switch self {
        case .notADirectory(let path):
            return "model dir: '\(path)' is not a directory"
        case .missingFile(let name, let directory):
            return "model dir: '\(directory)' has no \(name)"
        case .noCheckpoint(let directory):
            return "model dir: '\(directory)' contains no .safetensors checkpoint"
        case .multipleCheckpoints(let directory, let found):
            return "model dir: '\(directory)' contains \(found.count) .safetensors "
                + "files (\(found.joined(separator: ", "))) — expected exactly one"
        }
    }
}

/// A local model folder: exactly one single-file .safetensors checkpoint
/// (the consolidated artifact — sharded checkpoints are rejected downstream
/// by the parser), config.json, and the tokenizer pair. All local, never
/// fetched (the files are downloaded once at the pinned revision).
public struct ModelDirectory {
    public let directoryURL: URL
    public let checkpointURL: URL
    public let configURL: URL

    public init(validating url: URL) throws {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw ModelDirectoryError.notADirectory(path: url.path)
        }

        for required in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            let path = url.appendingPathComponent(required).path
            guard FileManager.default.fileExists(atPath: path) else {
                throw ModelDirectoryError.missingFile(name: required, directory: url.path)
            }
        }

        let entries = try FileManager.default.contentsOfDirectory(atPath: url.path)
        let checkpoints = entries.filter { $0.hasSuffix(".safetensors") }.sorted()
        guard !checkpoints.isEmpty else {
            throw ModelDirectoryError.noCheckpoint(directory: url.path)
        }
        guard checkpoints.count == 1 else {
            throw ModelDirectoryError.multipleCheckpoints(
                directory: url.path, found: checkpoints)
        }

        directoryURL = url
        checkpointURL = url.appendingPathComponent(checkpoints[0])
        configURL = url.appendingPathComponent("config.json")
    }
}
