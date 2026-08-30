import CryptoKit
import XCTest
import Foundation
import QwenMetalEngine

// MARK: - Metric computation (P3-3, phase-3.md D6)
//
// All statistics in float64 over fp32 logits — the identical definitions the
// band-setter dump (tools/dump_quality_gate.py) applies to mlx-lm 4-bit, so
// the band comparison is measured identically on both sides. Free functions
// so they are unit-tested synthetically even when the artifact-backed suite
// skips (tieAwareArgmaxHolds precedent).

/// Stable float64 logsumexp of an fp32 logit vector.
func quantLogSumExp(_ logits: [Float]) -> Double {
    var maxValue = -Double.infinity
    for value in logits where Double(value) > maxValue { maxValue = Double(value) }
    var sumExp = 0.0
    for value in logits { sumExp += exp(Double(value) - maxValue) }
    return maxValue + log(sumExp)
}

/// -log softmax(logits)[target] in float64 (per-position NLL).
func quantNegativeLogLikelihood(logits: [Float], target: Int) -> Double {
    precondition((0..<logits.count).contains(target), "target out of range")
    return quantLogSumExp(logits) - Double(logits[target])
}

/// KL(P_reference ‖ P_engine) in nats: float64 softmax over the full vector.
func quantKLDivergence(reference: [Float], engine: [Float]) -> Double {
    precondition(reference.count == engine.count, "vocab size mismatch")
    let refLSE = quantLogSumExp(reference)
    let engLSE = quantLogSumExp(engine)
    var kl = 0.0
    for i in 0..<reference.count {
        let pLog = Double(reference[i]) - refLSE
        let qLog = Double(engine[i]) - engLSE
        kl += exp(pLog) * (pLog - qLog)
    }
    return kl
}

/// Synthetic pins for the metric helpers — hand-derived expected values, no
/// artifacts needed. These always run.
final class QuantQualityMetricUnitTests: XCTestCase {

    func testKLOfIdenticalDistributionsIsZero() {
        let logits: [Float] = [0.5, -1.0, 2.0, 0.0, -3.25]
        XCTAssertEqual(quantKLDivergence(reference: logits, engine: logits),
                       0.0, accuracy: 1e-12)
    }

    func testKLMatchesHandComputedTwoElementCase() {
        // ref logits [0, ln 3] -> p = [1/4, 3/4]; eng [0, 0] -> q = [1/2, 1/2].
        // KL = 0.25·ln(0.5) + 0.75·ln(1.5) = 0.13081203594113694 nats.
        let kl = quantKLDivergence(reference: [0, 1.0986123], engine: [0, 0])
        // fp32 rounding of ln3 in the input shifts the value ~1e-8 — compare
        // against the exact recomputation from the fp32 inputs instead.
        let p1 = 1.0 / (1.0 + exp(-Double(Float(1.0986123))))
        let expected = (1 - p1) * log((1 - p1) / 0.5) + p1 * log(p1 / 0.5)
        XCTAssertEqual(kl, expected, accuracy: 1e-12)
        XCTAssertEqual(kl, 0.130812035941, accuracy: 1e-7)
    }

    func testKLIsInvariantUnderLogitShift() {
        let ref: [Float] = [1.5, -0.5, 3.0, 0.25]
        let eng: [Float] = [0.5, 0.5, 2.0, -1.0]
        let klBase = quantKLDivergence(reference: ref, engine: eng)
        let klShifted = quantKLDivergence(
            reference: ref.map { $0 + 7 }, engine: eng.map { $0 - 3 })
        XCTAssertEqual(klBase, klShifted, accuracy: 1e-10)
        XCTAssertGreaterThan(klBase, 0)
    }

    func testNLLUniformTwoElementIsLogTwo() {
        XCTAssertEqual(quantNegativeLogLikelihood(logits: [0, 0], target: 0),
                       0.6931471805599453, accuracy: 1e-12)
    }

    func testNLLHandComputedAsymmetricCase() {
        // logits [1, 3]: NLL(target 1) = ln(1 + e^-2) = 0.12692801104297263;
        // NLL(target 0) adds the 2.0 logit gap.
        XCTAssertEqual(quantNegativeLogLikelihood(logits: [1, 3], target: 1),
                       0.12692801104297263, accuracy: 1e-12)
        XCTAssertEqual(quantNegativeLogLikelihood(logits: [1, 3], target: 0),
                       2.1269280110429727, accuracy: 1e-12)
    }

    func testNLLIsInvariantUnderLogitShift() {
        let logits: [Float] = [0.25, -1.5, 2.0]
        XCTAssertEqual(
            quantNegativeLogLikelihood(logits: logits, target: 2),
            quantNegativeLogLikelihood(logits: logits.map { $0 + 11 }, target: 2),
            accuracy: 1e-10)
    }
}

// MARK: - Shared CPU-quant model + quality artifacts

/// Shared plumbing for the P3-3+ quality/oracle suites: ONE materialized
/// CPU-quant model (fp32 dequant of the packed artifact through the frozen
/// CPU modules — P3-2 front end; materialization is ~2.6 s release but the
/// resident model is ~7 GB, so suites share it; the P3-5 note asks for
/// exactly this pattern). RoPE table sized 4096 to cover the ppl window.
enum SharedQuantModel {
    static let packedURL = SharedCheckpoint.modelsDir
        .appendingPathComponent("qwen3-1.7b-70d244cc-q4g64.safetensors")
    static let refLogitsURL = SharedCheckpoint.modelsDir
        .appendingPathComponent("qwen3-1.7b-70d244cc-ref-logits-250.bin")
    static let qualityDir = SharedCheckpoint.repoRoot
        .appendingPathComponent("tests/fixtures/qwen3-1.7b-quality")

    struct Band: Decodable {
        struct Blob: Decodable {
            let dtype: String
            let shape: [Int]
            let byte_len: Int
            let sha256: String
        }
        struct RefArtifact: Decodable {
            let path: String
            let dtype: String
            let shape: [Int]
            let byte_len: Int
            let sha256: String
            let prompt_order: [String]
        }
        struct Reference: Decodable {
            let ppl_fp32: Double
            let ref_logits_artifact: RefArtifact
        }
        struct Setters: Decodable {
            let A_mlx: Double
            let A_mlx_count: Int
            let steps_total: Int
            let KL_mlx_nats: Double
            let ppl_mlx: Double
            let dppl_mlx: Double
        }
        let reference: Reference
        let band_setters: Setters
        let files: [String: Blob]
    }

    private static var sharedModel: Result<QwenModel, Error>?
    private static var sharedRefLogits: Result<Data, Error>?

    /// Call from setUpWithError: skips (never fails) when a local-only
    /// artifact is absent; committed fixtures missing is a hard error only
    /// once the local artifacts exist (then the set is inconsistent).
    static func skipUnlessArtifactsPresent() throws {
        for (url, hint) in [
            (packedURL, "produce it with `swift run qwen-metal-cli pack ...`"),
            (refLogitsURL, "produce it with tools/dump_quality_gate.py"),
        ] where !FileManager.default.fileExists(atPath: url.path) {
            throw XCTSkip("local-only artifact missing at \(url.path) — \(hint)")
        }
        guard FileManager.default.fileExists(
            atPath: qualityDir.appendingPathComponent("band.json").path) else {
            throw XCTSkip("quality fixtures missing at \(qualityDir.path) — "
                + "run tools/dump_quality_gate.py")
        }
    }

    static func band() throws -> Band {
        let data = try Data(contentsOf: qualityDir.appendingPathComponent("band.json"))
        return try JSONDecoder().decode(Band.self, from: data)
    }

    static func model() throws -> QwenModel {
        if sharedModel == nil {
            sharedModel = Result {
                let packed = try PackedCheckpoint(
                    path: packedURL.path,
                    expectedRevision: SharedCheckpoint.pinnedRevision)
                let config = try ModelConfig(
                    jsonData: Data(SharedCheckpoint.pinnedConfigJSON.utf8))
                return try QwenModel(
                    weights: packed, config: config, maxSequenceLength: 4096)
            }
        }
        return try sharedModel!.get()
    }

    /// The reference-logits artifact, sha256-verified ONCE against band.json
    /// (TOK-1 pattern: a drifted local artifact fails loudly, never silently
    /// grades against the wrong reference).
    static func refLogits() throws -> Data {
        if sharedRefLogits == nil {
            sharedRefLogits = Result {
                let band = try band()
                let artifact = band.reference.ref_logits_artifact
                let data = try Data(contentsOf: refLogitsURL, options: .mappedIfSafe)
                guard data.count == artifact.byte_len else {
                    throw SharedCheckpoint.HelperError(description:
                        "ref-logits artifact byte length \(data.count) != "
                        + "band.json record \(artifact.byte_len)")
                }
                let hash = SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }.joined()
                guard hash == artifact.sha256 else {
                    throw SharedCheckpoint.HelperError(description:
                        "ref-logits artifact sha256 \(hash) != band.json record "
                        + "\(artifact.sha256) — regenerate with "
                        + "tools/dump_quality_gate.py")
                }
                return data
            }
        }
        return try sharedRefLogits!.get()
    }

    /// One [vocab] fp32 row of the [5, 50, vocab] artifact.
    static func refRow(_ data: Data, promptIndex: Int, step: Int, vocab: Int) -> [Float] {
        let start = (promptIndex * 50 + step) * vocab * 4
        return data.subdata(in: start..<(start + vocab * 4)).withUnsafeBytes { raw in
            raw.bindMemory(to: UInt32.self).map {
                Float(bitPattern: UInt32(littleEndian: $0))
            }
        }
    }

    /// Committed quality blob as int32s, verified against the band.json index.
    static func qualityInt32s(_ name: String, expectedCount: Int) throws -> [Int] {
        let band = try band()
        guard let record = band.files[name] else {
            throw SharedCheckpoint.HelperError(description:
                "band.json has no file record for \(name)")
        }
        let data = try Data(contentsOf: qualityDir.appendingPathComponent(name))
        guard data.count == expectedCount * 4, data.count == record.byte_len else {
            throw SharedCheckpoint.HelperError(description:
                "\(name): byte length \(data.count) disagrees with expected "
                + "\(expectedCount * 4) / band.json \(record.byte_len)")
        }
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard hash == record.sha256 else {
            throw SharedCheckpoint.HelperError(description:
                "\(name): sha256 \(hash) != band.json record \(record.sha256)")
        }
        return data.withUnsafeBytes { raw in
            raw.bindMemory(to: Int32.self).map { Int(Int32(littleEndian: $0)) }
        }
    }
}

// MARK: - The quality gate (pre-committed band vs mlx-lm 4-bit)

/// P3-3 in-band verification: the CPU-quant reference's quality metrics vs
/// the measured mlx-lm 4-bit band-setters (band.json; also recorded in
/// DECISIONS.md BEFORE this suite first ran, per the gates-entry ordering
/// rule). Gate formulas and constants are the binding DECISIONS.md
/// 2026-08-25 record, veto-approved 2026-08-26 — never loosen (hard rule 6).
/// Out-of-band ⇒ suspect the PACKER; oracle layer 1 arbitrates (Issue 2).
///
/// Heavy suite (250 CPU-quant full re-forwards + one 4096-token window):
/// run release-mode, e.g.
///   swift test -c release --filter QuantQualityGateTests
final class QuantQualityGateTests: XCTestCase {

    // Pre-committed gate constants (DECISIONS.md "Phase 3 gates pre-committed").
    private static let agreementMargin = 0.04       // A_ours >= A_mlx - 4pp
    private static let klMultiplier = 1.5           // KL_ours <= 1.5 x KL_mlx
    private static let dpplMultiplier = 1.5         // dppl_ours <= 1.5 x dppl_mlx
    private static let dpplFloor = 0.01             //             + 0.01

    private static let generationSteps = 50
    private static let sliceTokens = 4096
    private static let lmHeadChunkRows = 256

    override func setUpWithError() throws {
        try SharedQuantModel.skipUnlessArtifactsPresent()
    }

    private struct StepsFile: Decodable {
        let prompt_id: String
        let argmax_token_ids: [Int]
    }

    private func referenceArgmax(_ prompt: String) throws -> [Int] {
        let url = SharedCheckpoint.fixturesDir
            .appendingPathComponent("prompts/\(prompt)/steps.json")
        let file = try JSONDecoder().decode(StepsFile.self, from: Data(contentsOf: url))
        XCTAssertEqual(file.prompt_id, prompt)
        return file.argmax_token_ids
    }

    func testTeacherForcedAgreementAndKLWithinBand() throws {
        let band = try SharedQuantModel.band()
        let model = try SharedQuantModel.model()
        let vocab = model.config.vocabSize
        let refData = try SharedQuantModel.refLogits()
        let promptOrder = band.reference.ref_logits_artifact.prompt_order
        XCTAssertEqual(band.band_setters.steps_total,
                       promptOrder.count * Self.generationSteps)

        var agree = 0
        var klSum = 0.0
        for (promptIndex, prompt) in promptOrder.enumerated() {
            let argmax = try referenceArgmax(prompt)
            XCTAssertEqual(argmax.count, Self.generationSteps)
            var ids = try SharedCheckpoint.promptFixture(prompt).inputIds
            for step in 0..<Self.generationSteps {
                let refRow = SharedQuantModel.refRow(
                    refData, promptIndex: promptIndex, step: step, vocab: vocab)
                // Artifact/fixture correspondence: the reference row must
                // reproduce the committed argmax (defense in depth on top of
                // the sha256 pin and the dump-time byte-identity check).
                XCTAssertEqual(Argmax.firstIndex(refRow), argmax[step],
                               "\(prompt) step \(step): ref artifact argmax "
                               + "disagrees with committed steps.json")

                let ours = try model.lastPositionLogits(ids: ids)
                XCTAssertEqual(ours.count, vocab)
                if Argmax.firstIndex(ours) == argmax[step] { agree += 1 }
                klSum += quantKLDivergence(reference: refRow, engine: ours)
                ids.append(argmax[step])
            }
        }

        let stepsTotal = promptOrder.count * Self.generationSteps
        let aOurs = Double(agree) / Double(stepsTotal)
        let klOurs = klSum / Double(stepsTotal)
        print("QuantQualityGate: A_ours = \(agree)/\(stepsTotal) = \(aOurs), "
            + "A_mlx = \(band.band_setters.A_mlx_count)/\(stepsTotal) = "
            + "\(band.band_setters.A_mlx)")
        print("QuantQualityGate: KL_ours = \(klOurs) nats, "
            + "KL_mlx = \(band.band_setters.KL_mlx_nats) nats")

        XCTAssertGreaterThanOrEqual(
            aOurs, band.band_setters.A_mlx - Self.agreementMargin,
            "top-1 agreement \(aOurs) below the pre-committed band "
            + "(A_mlx \(band.band_setters.A_mlx) - 0.04) — suspect the packer "
            + "(oracle layer 1 arbitrates)")
        XCTAssertLessThanOrEqual(
            klOurs, Self.klMultiplier * band.band_setters.KL_mlx_nats,
            "mean KL \(klOurs) above the pre-committed band "
            + "(1.5 x KL_mlx \(band.band_setters.KL_mlx_nats)) — suspect the "
            + "packer (oracle layer 1 arbitrates)")
    }

    func testPplSliceDeltaWithinBand() throws {
        let band = try SharedQuantModel.band()
        let model = try SharedQuantModel.model()
        let vocab = model.config.vocabSize
        let hidden = model.config.hiddenSize
        let tokens = try SharedQuantModel.qualityInt32s(
            "ppl_slice_tokens.bin", expectedCount: Self.sliceTokens)

        // One causal window; per-position logits from the frozen CPU stack
        // (final norm + lm_head over all rows, chunked to bound the logits
        // buffer at ~150 MB; lm_head stays on the sgemm wrapper, hard rule 8).
        let states = try model.hiddenStates(ids: tokens)
        let normed = try model.finalNorm(states)

        var nllSum = 0.0
        var position = 0
        while position < tokens.count - 1 {
            let rows = min(Self.lmHeadChunkRows, tokens.count - 1 - position)
            let slice = Array(normed[(position * hidden)..<((position + rows) * hidden)])
            let logits = try model.lmHead(slice, seqLen: rows)
            for r in 0..<rows {
                let row = Array(logits[(r * vocab)..<((r + 1) * vocab)])
                nllSum += quantNegativeLogLikelihood(
                    logits: row, target: tokens[position + r + 1])
            }
            position += rows
        }

        let pplOurs = exp(nllSum / Double(tokens.count - 1))
        let dpplOurs = pplOurs - band.reference.ppl_fp32
        print("QuantQualityGate: ppl_ours = \(pplOurs), "
            + "ppl_fp32 = \(band.reference.ppl_fp32), dppl_ours = \(dpplOurs), "
            + "dppl_mlx = \(band.band_setters.dppl_mlx)")

        XCTAssertTrue(pplOurs.isFinite && pplOurs > 1,
                      "degenerate perplexity \(pplOurs)")
        XCTAssertLessThanOrEqual(
            dpplOurs,
            Self.dpplMultiplier * band.band_setters.dppl_mlx + Self.dpplFloor,
            "ppl delta \(dpplOurs) above the pre-committed band "
            + "(1.5 x dppl_mlx \(band.band_setters.dppl_mlx) + 0.01) — suspect "
            + "the packer (oracle layer 1 arbitrates)")
    }
}
