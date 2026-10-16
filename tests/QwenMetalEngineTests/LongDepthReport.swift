import Foundation
import Metal
@testable import QwenMetalEngine

/// Pure helpers for the LD-1 long-depth free-run REPORT harness
/// (LongDepthFreeRunReportTests). Everything here is artifact-free and
/// unit-tested on synthetic inputs; nothing here is a gate — the species
/// constants are the Phase 2 Tier-E values reused as DESCRIPTIVE
/// classifications of a checkpoint (within / outside), never asserted.
enum LongDepthReport {

    /// The pinned context (PLAN.md non-goals: fixed 4K, preallocated KV).
    static let pinnedContext = 4096

    static let allPrompts = [
        "short_english", "multi_sentence", "code_snippet",
        "non_ascii", "chat_template",
    ]

    /// Tier-E species (DECISIONS.md 2026-08-23 Phase 2 gates; reused verbatim
    /// by the GPU-quant logit suite): descriptive here.
    static let relE = exp2(Float(-5))     // full-vocab / top-64 / lse / mean
    static let relStd = exp2(Float(-4))   // std is 2-Lipschitz
    static let tieRel = exp2(-4.0)        // ε_tie = 2⁻⁴·M64
    static let topK = 64

    struct KnobError: Error, CustomStringConvertible {
        let description: String
    }

    // MARK: - Knobs

    /// A positive-integer environment knob: absent → default; anything that
    /// is not a positive integer fails loudly, naming the knob.
    static func positiveInt(_ raw: String?, name: String, default value: Int) throws -> Int {
        guard let raw else { return value }
        let trimmed = raw.trimmingCharacters(in: .whitespaces)
        guard let parsed = Int(trimmed), parsed >= 1 else {
            throw KnobError(description:
                "\(name)='\(raw)' is not a positive integer")
        }
        return parsed
    }

    /// Comma-separated fixture prompt names (absent → all five); an unknown
    /// name or an empty list fails loudly.
    static func promptList(_ raw: String?) throws -> [String] {
        guard let raw else { return allPrompts }
        let names = raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !names.isEmpty else {
            throw KnobError(description: "QWEN_LONG_DEPTH_PROMPTS is empty")
        }
        for name in names where !allPrompts.contains(name) {
            throw KnobError(description:
                "QWEN_LONG_DEPTH_PROMPTS: unknown prompt '\(name)' "
                + "(known: \(allPrompts.joined(separator: ", ")))")
        }
        return names
    }

    /// Checkpoint steps (1-based: "after k generated tokens"): every `every`
    /// steps plus the final step, ascending, no duplicates. The checkpoint
    /// at step k compares the logits that produced token k — a prefix of
    /// prompt + k−1 generated tokens.
    static func checkpointSchedule(steps: Int, every: Int) throws -> [Int] {
        guard steps >= 1 else {
            throw KnobError(description: "steps must be >= 1, got \(steps)")
        }
        guard every >= 1 else {
            throw KnobError(description: "checkpoint interval must be >= 1, got \(every)")
        }
        var schedule = Array(stride(from: every, through: steps, by: every))
        if schedule.last != steps { schedule.append(steps) }
        return schedule
    }

    // MARK: - Per-step statistics

    /// Indices of the `count` largest FINITE values, ties broken by lower
    /// index (the suite's `topIndices` ordering) — a single pass keeping a
    /// sorted window of `count`, so the full-vocab vector is never sorted.
    static func topIndices(_ values: [Float], count: Int) -> [Int] {
        guard count >= 1 else { return [] }
        var best: [Int] = []
        best.reserveCapacity(count + 1)
        for i in values.indices {
            let v = values[i]
            guard v.isFinite else { continue }
            // Full window and not strictly above its floor: a later index
            // never displaces an equal value.
            if best.count == count, !(v > values[best[count - 1]]) { continue }
            var j = best.count
            while j > 0, v > values[best[j - 1]] { j -= 1 }
            best.insert(i, at: j)
            if best.count > count { best.removeLast() }
        }
        return best
    }

    /// What the harness keeps per step from the GPU's own logits.
    struct StepStats {
        let nonFiniteCount: Int
        /// -1 when fewer than one / two finite logits exist.
        let top1: Int
        let top2: Int
        /// logits[top1] − logits[top2] (float64); NaN when top2 is absent.
        let margin: Double
        /// max |value| over the top-64 — the suite's M64 on this vector.
        let m64: Float
        let fingerprint: (logsumexp: Double, mean: Double, std: Double)

        /// margin < ε_tie = 2⁻⁴·M64: a step the committed per-element bound
        /// cannot distinguish from a tie — the population that can fork.
        var isForkEligible: Bool {
            margin.isFinite && margin < tieRel * Double(m64)
        }

        init(_ logits: [Float]) {
            var nonFinite = 0
            for value in logits where !value.isFinite { nonFinite += 1 }
            nonFiniteCount = nonFinite
            let top = topIndices(logits, count: topK)
            top1 = top.count > 0 ? top[0] : -1
            top2 = top.count > 1 ? top[1] : -1
            margin = top.count > 1
                ? Double(logits[top[0]]) - Double(logits[top[1]]) : .nan
            var m: Float = 0
            for index in top { m = max(m, abs(logits[index])) }
            m64 = m
            fingerprint = logitFingerprints(logits)
        }
    }

    // MARK: - Checkpoint comparison (GPU logits vs the CPU-quant oracle)

    struct CheckpointComparison {
        let mStep: Float
        let worstFull: Float
        let worstFullIndex: Int
        let m64: Float
        let worstTop64: Float
        let worstTop64Rank: Int
        let dLogsumexp: Double
        let dMean: Double
        let dStd: Double
        let cpuTop1: Int
        let cpuTop2: Int
        let cpuMargin: Double
        let gpuArgmax: Int

        var fullSpecies: Float { relE * mStep }
        var top64Species: Float { relE * m64 }
        var scalarSpecies: Double { Double(relE * m64) }
        var stdSpecies: Double { Double(relStd * m64) }
        var tieEpsilon: Double { tieRel * Double(m64) }

        var fullWithinSpecies: Bool { worstFull <= fullSpecies }
        var top64WithinSpecies: Bool { worstTop64 <= top64Species }
        var fingerprintsWithinSpecies: Bool {
            dLogsumexp <= scalarSpecies && dMean <= scalarSpecies && dStd <= stdSpecies
        }
        var argmaxAgrees: Bool { gpuArgmax == cpuTop1 }
        var tieAwareTop1Holds: Bool {
            tieAwareArgmaxHolds(
                ours: gpuArgmax, top1: cpuTop1, top2: cpuTop2,
                margin: cpuMargin, epsilon: tieEpsilon)
        }

        /// The GPU-quant logit suite's per-step arithmetic, verbatim, with
        /// the CPU-quant vector as the reference side.
        init(gpu: [Float], cpu: [Float]) {
            precondition(gpu.count == cpu.count && cpu.count >= 2,
                         "checkpoint vectors must match and hold >= 2 logits")
            var mStep: Float = 0
            for value in cpu { mStep = max(mStep, abs(value)) }
            self.mStep = mStep
            var worst: Float = 0
            var worstIndex = -1
            for i in cpu.indices {
                let diff = abs(gpu[i] - cpu[i])
                if diff > worst { worst = diff; worstIndex = i }
            }
            worstFull = worst
            worstFullIndex = worstIndex

            let top = topIndices(cpu, count: topK)
            var m64: Float = 0
            for index in top { m64 = max(m64, abs(cpu[index])) }
            self.m64 = m64
            var worstTop: Float = 0
            var worstRank = -1
            for (rank, index) in top.enumerated() {
                let diff = abs(gpu[index] - cpu[index])
                if diff > worstTop { worstTop = diff; worstRank = rank }
            }
            worstTop64 = worstTop
            worstTop64Rank = worstRank

            let gpuFingerprint = logitFingerprints(gpu)
            let cpuFingerprint = logitFingerprints(cpu)
            dLogsumexp = abs(gpuFingerprint.logsumexp - cpuFingerprint.logsumexp)
            dMean = abs(gpuFingerprint.mean - cpuFingerprint.mean)
            dStd = abs(gpuFingerprint.std - cpuFingerprint.std)

            // A reference vector with fewer than two finite logits is
            // pathological (fp32 CPU-quant never produces one); report it
            // rather than crash: top-2 absent, margin NaN (tie band).
            cpuTop1 = top.count > 0 ? top[0] : -1
            cpuTop2 = top.count > 1 ? top[1] : -1
            cpuMargin = top.count > 1
                ? Double(cpu[top[0]]) - Double(cpu[top[1]]) : .nan
            gpuArgmax = Argmax.firstIndex(gpu)
        }
    }

    // MARK: - KV cache scan (fp16-overflow tripwire 2)

    struct KVScan {
        let nonFiniteCount: Int
        let maxAbsFinite: Float
        let elementsScanned: Int
    }

    /// Reads the fp16 K and V slots of positions `0..<positions` on every
    /// layer and head (shared storage mode — the CPU sees the GPU's
    /// stores) and counts non-finite entries; positions beyond the scanned
    /// range are never read. Throws when `positions` exceeds the cache's
    /// maxContext.
    static func kvCacheScan(_ cache: KVCache, positions: Int) throws -> KVScan {
        guard positions >= 0, positions <= cache.maxContext else {
            throw KVCacheError.indexOutOfRange(
                name: "positions", value: positions, bound: cache.maxContext + 1)
        }
        guard positions > 0 else {
            return KVScan(nonFiniteCount: 0, maxAbsFinite: 0, elementsScanned: 0)
        }
        let halves = cache.buffer.contents().bindMemory(
            to: Float16.self, capacity: cache.byteCount / 2)
        var nonFinite = 0
        var maxAbs: Float = 0
        var scanned = 0
        for layer in 0..<cache.layers {
            for component in [KVCache.Component.key, .value] {
                for head in 0..<cache.kvHeads {
                    let base = try cache.elementOffset(
                        layer: layer, component: component, head: head, position: 0)
                    let count = positions * cache.headDim
                    for i in base..<(base + count) {
                        let value = Float(halves[i])
                        if value.isFinite {
                            maxAbs = max(maxAbs, abs(value))
                        } else {
                            nonFinite += 1
                        }
                    }
                    scanned += count
                }
            }
        }
        return KVScan(
            nonFiniteCount: nonFinite, maxAbsFinite: maxAbs, elementsScanned: scanned)
    }

    // MARK: - Descriptive summaries

    static func distinctTokens(in tokens: [Int], tail: Int) -> Int {
        Set(tokens.suffix(max(tail, 0))).count
    }

    struct Quantiles {
        let min: Double
        let p1: Double
        let p10: Double
        let median: Double
    }

    /// Nearest-rank quantiles over the finite values; NaN everywhere when
    /// nothing finite exists.
    static func quantiles(_ values: [Double]) -> Quantiles {
        let sorted = values.filter(\.isFinite).sorted()
        guard !sorted.isEmpty else {
            return Quantiles(min: .nan, p1: .nan, p10: .nan, median: .nan)
        }
        func at(_ fraction: Double) -> Double {
            let rank = Int((fraction * Double(sorted.count)).rounded(.up))
            return sorted[Swift.min(Swift.max(rank - 1, 0), sorted.count - 1)]
        }
        return Quantiles(min: sorted[0], p1: at(0.01), p10: at(0.10), median: at(0.5))
    }

    /// A one-line excerpt (newlines shown as ⏎) — the head or the tail of a
    /// long generation, for the report's coherence note.
    static func excerpt(_ text: String, head: Int) -> String {
        let flat = text.replacingOccurrences(of: "\n", with: "⏎")
        return flat.count > head ? String(flat.prefix(head)) + "…" : flat
    }

    static func excerpt(_ text: String, tail: Int) -> String {
        let flat = text.replacingOccurrences(of: "\n", with: "⏎")
        return flat.count > tail ? "…" + String(flat.suffix(tail)) : flat
    }

    static func hostDescription() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var brand = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &brand, &size, nil, 0)
        let cpu = String(cString: brand)
        let memoryGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        let os = ProcessInfo.processInfo.operatingSystemVersionString
        return "\(cpu), \(String(format: "%.0f", memoryGB)) GB, macOS \(os)"
    }
}
