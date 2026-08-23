// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "qwen-metal",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "QwenMetalEngine", targets: ["QwenMetalEngine"]),
        .executable(name: "qwen-metal-cli", targets: ["QwenMetalCLI"]),
    ],
    dependencies: [
        // Tokenizer only (PLAN.md dependencies list; non-goals forbid growing
        // our own). Pinned exact — tokenization drifting under an unpinned
        // version is the same bug as unpinned Python fixture deps.
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            exact: "1.3.3"),
    ],
    targets: [
        // Engine core — shared by the CLI and (Phase 2) the iOS app.
        // No engine logic ever lives in app/CLI targets (CLAUDE.md project shape).
        .target(
            name: "QwenMetalEngine",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers")
            ]),
        .executableTarget(
            name: "QwenMetalCLI",
            dependencies: ["QwenMetalEngine"]
        ),
        // Explicit path: the repo's shared test root is lowercase `tests/`
        // (it also holds the Python backlog drift test), and macOS's
        // case-insensitive filesystem would silently alias a default `Tests/`
        // onto it while case-sensitive checkouts would not find it.
        .testTarget(
            name: "QwenMetalEngineTests",
            dependencies: ["QwenMetalEngine"],
            path: "tests/QwenMetalEngineTests"
        ),
    ]
)
