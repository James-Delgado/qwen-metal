import Foundation

/// Token-embedding lookup: each id copies one row of the [vocab, hidden]
/// table. A pure row copy of exactly-upcast weights, so the P1-4 activation
/// test asserts it with ==, no tolerance (DECISIONS.md 2026-08-23 gate).
public struct Embedding {
    /// [vocabSize, hiddenSize], row-major fp32 (exact upcast from checkpoint).
    public let weight: [Float]
    public let vocabSize: Int
    public let hiddenSize: Int

    public init(weight: [Float], vocabSize: Int, hiddenSize: Int) throws {
        guard weight.count == vocabSize * hiddenSize else {
            throw ModelError.badWeightShape(
                tensor: "embedding", expected: [vocabSize, hiddenSize],
                actual: [weight.count])
        }
        self.weight = weight
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
    }

    /// Returns [ids.count, hiddenSize] row-major.
    public func callAsFunction(_ ids: [Int]) throws -> [Float] {
        var out = [Float]()
        out.reserveCapacity(ids.count * hiddenSize)
        for id in ids {
            guard id >= 0, id < vocabSize else {
                throw ModelError.tokenIdOutOfRange(id: id, vocabSize: vocabSize)
            }
            out.append(contentsOf: weight[(id * hiddenSize)..<((id + 1) * hiddenSize)])
        }
        return out
    }
}
