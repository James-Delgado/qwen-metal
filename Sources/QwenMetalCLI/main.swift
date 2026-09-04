import Foundation
import QwenMetalEngine

// Thin CLI entry point — all engine logic lives in QwenMetalEngine.
// Subcommands arrive with their phases; P0B-4 added `bandwidth`,
// P1-5 added `generate`, P3-1 added `pack`, P3-6 added `microbench`.
let arguments = CommandLine.arguments.dropFirst()
switch arguments.first {
case nil:
    print("\(EngineInfo.name) v\(EngineInfo.version) — subcommands: bandwidth, generate, pack, microbench")
case "bandwidth":
    exit(runBandwidthCommand())
case "generate":
    exit(await runGenerateCommand(Array(arguments.dropFirst())))
case "pack":
    exit(runPackCommand(Array(arguments.dropFirst())))
case "microbench":
    exit(runMicrobenchCommand(Array(arguments.dropFirst())))
case let unknown?:
    FileHandle.standardError.write(
        Data("unknown subcommand '\(unknown)' — available: bandwidth, generate, pack, microbench\n".utf8)
    )
    exit(2)
}
