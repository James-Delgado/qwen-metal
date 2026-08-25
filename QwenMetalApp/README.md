# QwenMetalApp — thin iOS shell (Phase 2, P2-6)

The committed iOS target planned in CLAUDE.md (spec: docs/phases/phase-2.md D8).
Thin by rule: all engine logic, the benchmark protocol runner, timing, and row
export live in the `QwenMetalEngine` package (local dependency on the repo
root); this target is SwiftUI glue only. Agents build it; **James signs,
deploys, and runs on device — never agents.**

## Layout

```
QwenMetalApp.xcodeproj      hand-authored project (synchronized Sources folder,
                            shared QwenMetalApp scheme; Run defaults to Release)
Sources/                    app code (AppModel + 2 screens) + Resources/
Sources/Resources/          bundled copies of the two pinned rendered prompts —
                            drift-tested byte-identical to benchmarks/prompts/
                            rendered/ by AppBundledPromptTests
Info.plist                  file sharing ON (model transfer path, see below)
QwenMetalApp.entitlements   Increased Memory Limit (Phase 2 needs ~4.0 GB)
```

## One-time setup (Xcode 26.6 — the release Xcode, never the beta)

1. Open `QwenMetalApp.xcodeproj`. Signing & Capabilities → select your personal
   team (bundle id `dev.qwenmetal.QwenMetalApp`; change it if your team needs a
   different prefix). The Increased Memory Limit entitlement is already wired.
2. Build & run on the iPhone 15 Pro (pinned device). The shared scheme's Run
   action is already **Release** (benchmark protocol pin).
3. **Model transfer:** connect the phone → Finder → Files tab → QwenMetalApp →
   drag in the pinned model folder (any name, e.g. `qwen3-1.7b/`) containing:
   the consolidated `qwen3-1.7b-70d244cc.safetensors`, `config.json`,
   `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`
   (all from `models/`). The app searches its Documents folder and validates
   via the engine's `ModelDirectory` — a clear error lists what's missing.
4. Metal API validation is OFF in normal (non-Xcode-scheme-diagnostics) runs;
   the row export reminds you to confirm and record it (P0A-1 addendum).

## Screens

- **Generate** — prompt → text on the GPU backend. Quick-load buttons for the
  two pinned prompts (bundled rendered forms — never paste from clipboard, the
  P0A-1 lesson; trailing `\n\n` preserved, the CLI-1 lesson), Regenerate
  (re-runs the last prompt — the manual form of the sustained loop), Stop
  (cooperative, token-boundary).
- **Benchmark** — the pinned protocol via the engine harness:
  - *burst*: 640-token cap (P2-5 Mac row precedent), prompt picker
    (decode-essay for decode rows; prefill-summarize for the prefill row).
  - *sustained*: ≥5-min regenerate loop, pinned to decode-essay; per-generation
    tok/s sequence is kept (the OV#9 bimodality signal).
  - Residency toggle mmap / wiredCopy — switching drops the loaded model; the
    next run reloads in the new mode (residency is baked in at load, spec D1).
  - Row export: all fields (dual timing medians, wall−GPU overhead,
    dispatches/token, canonical-window rate, prefill note, phys_footprint
    cross-check, PROVISIONAL marker) as shareable/copyable text. The Xcode
    memory gauge remains the phys_footprint metric of record; battery health
    and cold/warm are operator-entered fields.

## P2-7 run protocol

The on-device "before" row + mmap-vs-wired sustained comparison follow
benchmarks/phase0-runbook.md conventions (battery > 50%, rest to ambient,
cold vs warm annotated, 3 repeats for headline numbers). Rows append to
benchmarks/results.md; the residency decision lands in DECISIONS.md.
