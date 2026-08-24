import Foundation

/// Errors thrown by the Phase 1 CPU reference model. Loud and specific:
/// a wrong checkpoint or a family mismatch must name itself, never produce
/// silently-wrong numbers the oracle then has to catch.
public enum ModelError: Error, CustomStringConvertible {
    case unsupportedFamily(detail: String)
    case badWeightShape(tensor: String, expected: [Int], actual: [Int])
    case badWeightDtype(tensor: String, expected: String, actual: String)
    case tokenIdOutOfRange(id: Int, vocabSize: Int)
    case badInput(detail: String)

    public var description: String {
        switch self {
        case .unsupportedFamily(let detail):
            return "model: unsupported family: \(detail)"
        case .badWeightShape(let tensor, let expected, let actual):
            return "model: tensor '\(tensor)' has shape \(actual), expected \(expected)"
        case .badWeightDtype(let tensor, let expected, let actual):
            return "model: tensor '\(tensor)' has dtype \(actual), expected \(expected)"
        case .tokenIdOutOfRange(let id, let vocabSize):
            return "model: token id \(id) is outside the vocabulary (0..<\(vocabSize))"
        case .badInput(let detail):
            return "model: bad input: \(detail)"
        }
    }
}
