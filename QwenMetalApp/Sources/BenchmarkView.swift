import SwiftUI
import UIKit
import QwenMetalEngine

/// D8 benchmark screen: runs the pinned protocol (burst decode-essay,
/// sustained 5-min regenerate loop), weights bf16/q4g64 toggle (P3-5),
/// residency mmap/wired toggle, and displays + exports the row fields. The prompt picker on burst also
/// serves the prefill row (prefill-summarize — prompts/README roles).
struct BenchmarkView: View {
    @EnvironmentObject private var model: AppModel
    @State private var mode: BenchmarkReport.Mode = .burst
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
                        Text("burst").tag(BenchmarkReport.Mode.burst)
                        Text("sustained (≥5 min)")
                            .tag(BenchmarkReport.Mode.sustained)
                    }
                    .pickerStyle(.segmented)
                    .disabled(model.isRunning)
                    if mode == .burst {
                        Picker("Prompt", selection: $burstPrompt) {
                            ForEach(BundledPrompt.allCases) {
                                Text($0.rawValue).tag($0)
                            }
                        }
                        .disabled(model.isRunning)
                    } else {
                        Text("sustained is pinned to decode-essay "
                            + "(prompt role separation)")
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
