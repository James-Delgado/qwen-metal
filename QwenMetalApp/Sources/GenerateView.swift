import SwiftUI
import QwenMetalEngine

/// D8 generate screen: prompt → text, backend fixed to GPU. Quick-load
/// buttons for the two pinned prompts and a Regenerate button (the manual
/// form of the sustained regenerate-loop protocol) are the tester-friction
/// controls James asked for.
struct GenerateView: View {
    @EnvironmentObject private var model: AppModel
    @State private var prompt = ""
    @State private var maxTokens = 256

    var body: some View {
        NavigationStack {
            Form {
                Section("Prompt") {
                    TextEditor(text: $prompt)
                        .frame(minHeight: 120)
                        .font(.system(.footnote, design: .monospaced))
                        .disabled(model.isRunning)
                    HStack {
                        ForEach(BundledPrompt.allCases) { pinned in
                            Button(pinned.rawValue) {
                                prompt = (try? pinned.text()) ?? prompt
                            }
                            .buttonStyle(.bordered)
                            .disabled(model.isRunning)
                        }
                    }
                    Stepper(
                        "max new tokens: \(maxTokens)",
                        value: $maxTokens, in: 16...4096, step: 16)
                }

                Section("Run") {
                    HStack {
                        Button("Generate") {
                            let text = prompt
                            let cap = maxTokens
                            Task {
                                await model.generate(
                                    prompt: text, maxNewTokens: cap)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(
                            model.isRunning || model.isLoading
                                || prompt.isEmpty)

                        Button("Regenerate") {
                            guard let last = model.lastPrompt else { return }
                            let cap = maxTokens
                            Task {
                                await model.generate(
                                    prompt: last, maxNewTokens: cap)
                            }
                        }
                        .buttonStyle(.bordered)
                        .disabled(
                            model.isRunning || model.isLoading
                                || model.lastPrompt == nil)

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

                if !model.outputText.isEmpty {
                    Section("Output") {
                        Text(model.outputText)
                            .font(.system(.footnote, design: .monospaced))
                            .textSelection(.enabled)
                    }
                }
            }
            .navigationTitle("Generate")
        }
    }
}
