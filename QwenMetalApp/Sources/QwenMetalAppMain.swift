import SwiftUI

// P2-6 (phase-2.md D8): thin iOS shell around QwenMetalEngine. Two screens:
// generate (prompt → text, GPU backend) and benchmark (pinned protocol).
// James signs, deploys, and runs on device — agents never do (standing rule).

@main
struct QwenMetalAppMain: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            TabView {
                GenerateView()
                    .tabItem { Label("Generate", systemImage: "text.cursor") }
                BenchmarkView()
                    .tabItem { Label("Benchmark", systemImage: "gauge") }
            }
            .environmentObject(model)
        }
    }
}
