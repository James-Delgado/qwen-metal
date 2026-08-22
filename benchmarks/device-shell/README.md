# Disposable device shell — iPhone triad run (P0A-1, decision 1A)

Gets `TriadRunnerView.swift` onto the phone without creating a committed iOS
target (the real app scaffold arrives with Phase 2, seeded by SPEC-P2). The
scratch Xcode project lives OUTSIDE this repo and is thrown away afterwards;
only the numbers land in the repo.

## One-time setup (Xcode 26.6 — release Xcode, NOT the beta; toolchain noted per row)

1. Xcode → File → New → Project → iOS → App. Name: `TriadScratch` (anywhere
   outside this repo, e.g. `~/Desktop`). Interface: SwiftUI. Team: your
   personal/dev team for signing.
2. Add the engine package: Project → Package Dependencies → Add Local… →
   select this repo's root directory (`qwen-metal`, the folder containing
   `Package.swift`). Add product `QwenMetalEngine` to the `TriadScratch`
   target.
3. Replace the generated `ContentView.swift` contents with
   `benchmarks/device-shell/TriadRunnerView.swift`, and change the `App`
   struct's body to `TriadRunnerView()`.
4. Signing & Capabilities → add capability **Increased Memory Limit**
   (`com.apple.developer.kernel.increased-memory-limit`). The three 384 MiB
   buffers fit without it on an 8 GB phone, but run with it anyway — it is the
   entitlement every later on-device phase uses.
5. Build & run on the iPhone 15 Pro (pinned device) in **Release**
   configuration (Edit Scheme → Run → Build Configuration → Release —
   matches the Mac row's release-build protocol).

## Run protocol (mirrors the pinned P0B-4 protocol)

- Battery > 50%, battery health % noted, rest to ambient temperature first,
  unplugged, airplane mode not required for bandwidth (no comparative energy).
- Tap run. The view enforces the pre-committed 1e-6 correctness gate and
  reports per-iteration GB/s (GPU time; wall + overhead alongside), sustained
  (median), and spread.
- Run it 3 times (rest between runs); record the run whose sustained figure is
  the median of the three, with the spread across all three noted.

## Recording (append; never overwrite rows)

`benchmarks/results.md` row (PROVISIONAL), same columns as the Mac row:
date, device (iPhone 15 Pro), iOS version, battery health %, starting temp,
toolchain (Xcode 26.6 build 17F113), release build, sustained (median) GB/s,
spread, dispatch overhead. Then update DECISIONS.md's OPEN item: measured
iPhone DRAM bandwidth = roofline denominator (PLAN.md invariant 1), and
update PLAN.md's provisional figures to derive from it.
