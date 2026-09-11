import SwiftUI
import UIKit
import QwenMetalEngine

/// D8 benchmark screen: runs the pinned protocol (burst decode-essay,
/// sustained 5-min regenerate loop), weights bf16/q4g64 toggle (P3-5),
/// residency mmap/wired toggle, and displays + exports the row fields. The prompt picker on burst also
/// serves the prefill row (prefill-summarize — prompts/README roles).
struct BenchmarkView: View {
    /// Screen-local run modes: the two BenchmarkReport generation modes, the
    /// P3-6 dequant-matvec microbench (weights-only, no generation), and the
    /// P4-1 diagnostic attribution run (never a benchmark row).
    private enum RunMode: String, CaseIterable {
        case burst
        case sustained
        case microbench
        case attribution
    }

    @EnvironmentObject private var model: AppModel
    @State private var mode: RunMode = .burst
    @State private var burstPrompt: BundledPrompt = .decodeEssay
    @State private var batteryNote = ""
    @State private var coldWarmNote = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Engine") {
                    Picker("Weights", selection: $model.weightsFormat) {
                        ForEach(WeightsFormat.allCases, id: \.self) {
                            Text($0.rawValue)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(model.isRunning || model.isLoading)
                    .onChange(of: model.weightsFormat) {
                        model.weightsFormatChanged()
                    }
                    Picker("Residency", selection: $model.residency) {
                        ForEach(WeightsResidency.allCases, id: \.self) {
                            Text($0.rawValue)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(model.isRunning || model.isLoading)
                    .onChange(of: model.residency) {
                        model.residencyChanged()
                    }
                    // P4-4 (spec D4): fused default; naive exists for the
                    // P4-5 interleaved before/after row. q4g64 only — the
                    // bf16 backend is permanently naive.
                    if model.weightsFormat == .q4g64 {
                        Picker("Kernels", selection: $model.kernelPath) {
                            ForEach(GPUModel.KernelPath.allCases, id: \.self) {
                                Text($0.rawValue)
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(model.isRunning || model.isLoading)
                        .onChange(of: model.kernelPath) {
                            model.kernelPathChanged()
                        }
                    }
                    Button(model.loadSummary == nil
                        ? "Load model" : "Reload model") {
                        Task { await model.loadModel() }
                    }
                    .disabled(model.isRunning || model.isLoading)
                    if let summary = model.loadSummary {
                        Text(summary).font(.caption)
                    }
                }

                Section("Protocol") {
                    Picker("Mode", selection: $mode) {
                        Text("burst").tag(RunMode.burst)
                        Text("sustained (≥5 min)").tag(RunMode.sustained)
                        Text("microbench").tag(RunMode.microbench)
                        Text("attribution").tag(RunMode.attribution)
                    }
                    .pickerStyle(.segmented)
                    .disabled(model.isRunning)
                    switch mode {
                    case .burst:
                        Picker("Prompt", selection: $burstPrompt) {
                            ForEach(BundledPrompt.allCases) {
                                Text($0.rawValue).tag($0)
                            }
                        }
                        .disabled(model.isRunning)
                    case .sustained:
                        Text("sustained is pinned to decode-essay "
                            + "(prompt role separation)")
                            .font(.caption)
                    case .microbench:
                        Text("D7 dequant-matvec sweep (197 packed matvecs, "
                            + "weights-only; q4g64 artifact required). Gate "
                            + "30.7 GB/s = best of ≥3 same-session runs, "
                            + "detached (D8).")
                            .font(.caption)
                    case .attribution:
                        Text("P4 D1 per-kernel-class GPU attribution "
                            + "(DIAGNOSTIC — never a benchmark row). "
                            + "decode-essay, 64 interleaved forwards; feeds "
                            + "the P4-EXEC roofline decomposition.")
                            .font(.caption)
                    }
                    TextField(
                        "battery health % (Settings → Battery)",
                        text: $batteryNote)
                    TextField("cold / warm annotation", text: $coldWarmNote)
                }

                Section("Run") {
                    HStack {
                        Button("Run") {
                            let prompt = burstPrompt
                            let battery = batteryNote
                            let coldWarm = coldWarmNote
                            let mode = mode
                            Task {
                                switch mode {
                                case .burst:
                                    await model.runBurst(
                                        prompt: prompt,
                                        batteryNote: battery,
                                        coldWarmNote: coldWarm)
                                case .sustained:
                                    await model.runSustained(
                                        batteryNote: battery,
                                        coldWarmNote: coldWarm)
                                case .microbench:
                                    await model.runMicrobench(
                                        batteryNote: battery,
                                        coldWarmNote: coldWarm)
                                case .attribution:
                                    await model.runAttribution()
                                }
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isRunning || model.isLoading)

                        Button("Stop", role: .destructive) {
                            model.requestStop()
                        }
                        .buttonStyle(.bordered)
                        .disabled(!model.isRunning)
                    }
                    if !model.statusLine.isEmpty {
                        Text(model.statusLine).font(.caption)
                    }
                    if let error = model.errorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.red)
                            .textSelection(.enabled)
                    }
                }

                if let report = model.lastReport {
                    Section("Row export") {
                        Text(report)
                            .font(.system(.caption2, design: .monospaced))
                            .textSelection(.enabled)
                        ShareLink(item: report) {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                        Button {
                            UIPasteboard.general.string = report
                        } label: {
                            Label("Copy", systemImage: "doc.on.doc")
                        }
                    }
                }
            }
            .navigationTitle("Benchmark")
        }
    }
}
