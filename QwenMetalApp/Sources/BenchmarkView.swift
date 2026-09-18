import SwiftUI
import UIKit
import QwenMetalEngine

/// D8 benchmark screen: runs the pinned protocol (burst decode-essay,
/// sustained 5-min regenerate loop), weights bf16/q4g64 toggle (P3-5),
/// residency mmap/wired toggle, kernels toggle (P4-4), prefill toggle
/// (P5-4), and displays + exports the row fields. The prompt picker on burst also
/// serves the prefill row (prefill-summarize — prompts/README roles).
struct BenchmarkView: View {
    /// Screen-local run modes: the two BenchmarkReport generation modes, the
    /// P3-6 dequant-matvec microbench (weights-only, no generation), and the
    /// two diagnostic runs (never benchmark rows): P4-1 attribution and
    /// OA-1 overhead anatomy.
    private enum RunMode: String, CaseIterable {
        case burst
        case sustained
        case microbench
        case attribution
        case overheadAnatomy
    }

    @EnvironmentObject private var model: AppModel
    @State private var mode: RunMode = .burst
    @State private var burstPrompt: BundledPrompt = .decodeEssay
    /// P5-5 session default: the GEMM M-sweep (the Phase 5 gate); matvec
    /// stays selectable for the Phase 3 protocol.
    @State private var microbenchKernel: AppModel.MicrobenchKernel = .gemm
    /// PF-1: prefill breakdown by default for the iterate round; decode
    /// stays selectable (the P4-1 breakdown).
    @State private var attributionMode: AppModel.AttributionMode = .prefill
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
                            // The naive arm is sequential-only: a kernels
                            // switch can change the effective prefill path.
                            model.prefillPathChanged()
                        }
                    }
                    // P5-4 (phase-5.md D5): tiled default; sequential exists
                    // for the P5-5 interleaved before/after row. q4g64 +
                    // fused only — bf16 keeps sequential permanently and
                    // the naive kernel arm supports sequential only.
                    if model.weightsFormat == .q4g64, model.kernelPath == .fused {
                        Picker("Prefill", selection: $model.prefillPath) {
                            ForEach(GPUModel.PrefillPath.allCases, id: \.self) {
                                Text($0.rawValue)
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(model.isRunning || model.isLoading)
                        .onChange(of: model.prefillPath) {
                            model.prefillPathChanged()
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
                        Text("overhead").tag(RunMode.overheadAnatomy)
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
                        Picker("Kernel", selection: $microbenchKernel) {
                            ForEach(AppModel.MicrobenchKernel.allCases) {
                                Text($0.rawValue).tag($0)
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(model.isRunning)
                        switch microbenchKernel {
                        case .matvec:
                            Text("P3-6 D7 dequant-matvec sweep (197 packed "
                                + "matvecs, weights-only; q4g64 artifact "
                                + "required). Gate 30.7 GB/s = best of ≥3 "
                                + "same-session runs, detached (D8).")
                                .font(.caption)
                        case .gemm:
                            Text("P5-2 D7 tiled dequant-GEMM M-sweep (M = 8, "
                                + "64, 512 over the same 197 matrices; "
                                + "weights-only). Gate: M=8 effective ≥ 30.69 "
                                + "GB/s = best of ≥3 same-session runs, "
                                + "detached (D8); GB/s + GFLOPS at every M "
                                + "are reported, never gated.")
                                .font(.caption)
                        }
                    case .attribution:
                        Picker("Breakdown", selection: $attributionMode) {
                            ForEach(AppModel.AttributionMode.allCases) {
                                Text($0.rawValue).tag($0)
                            }
                        }
                        .pickerStyle(.segmented)
                        .disabled(model.isRunning)
                        switch attributionMode {
                        case .decode:
                            Text("P4 D1 per-kernel-class GPU attribution "
                                + "(DIAGNOSTIC — never a benchmark row). "
                                + "decode-essay, 64 interleaved forwards; feeds "
                                + "the P4-EXEC roofline decomposition.")
                                .font(.caption)
                        case .prefill:
                            Text("PF-1 per-kernel-class GPU attribution inside "
                                + "the tiled prefill chunks (DIAGNOSTIC — never "
                                + "a benchmark row). prefill-summarize, "
                                + "\(BenchDefaults.prefillAttributionRuns) "
                                + "interleaved prefills (class-split vs "
                                + "production); needs Prefill = tiled.")
                                .font(.caption)
                        }
                    case .overheadAnatomy:
                        Text("OA-1 wall-GPU overhead anatomy (DIAGNOSTIC — "
                            + "never a benchmark row). decode-essay, 96 "
                            + "round-robin forwards; the device span split "
                            + "for the P4-9 verdict (P4-11 session).")
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
                            let kernel = microbenchKernel
                            let attribution = attributionMode
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
                                        kernel: kernel,
                                        batteryNote: battery,
                                        coldWarmNote: coldWarm)
                                case .attribution:
                                    await model.runAttribution(mode: attribution)
                                case .overheadAnatomy:
                                    await model.runOverheadAnatomy()
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
