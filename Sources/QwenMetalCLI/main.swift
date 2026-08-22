import Foundation
import QwenMetalEngine

// Thin CLI entry point — all engine logic lives in QwenMetalEngine.
// Subcommands arrive with their phases; P0B-4 added `bandwidth`.
let arguments = CommandLine.arguments.dropFirst()
switch arguments.first {
case nil:
    print("\(EngineInfo.name) v\(EngineInfo.version) — subcommands: bandwidth")
case "bandwidth":
    exit(runBandwidthCommand())
case let unknown?:
    FileHandle.standardError.write(
        Data("unknown subcommand '\(unknown)' — available: bandwidth\n".utf8)
    )
    exit(2)
}
