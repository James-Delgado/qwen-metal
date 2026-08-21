# DECISIONS.md — dated record of decisions and measurements

Append-only. Every session that decides or measures something adds a dated entry:
what was decided/measured, and why. See CLAUDE.md session discipline.

---

## 2026-08-20 — Plan solidified via full engineering review (20 findings folded)

Source: /plan-eng-review over the Downloads drafts; 8 first-pass findings + 12
outside-voice (cross-model) findings, all accepted. Load-bearing decisions:

1. **safetensors parser: hand-written, not swift-safetensors.** ~150 lines, mmap-based,
   single-file only. Raw pointer/mmap control matters for iOS memory accounting;
   the library's zero-copy path returns Core ML types the engine doesn't use.
2. **KV cache split.** Minimal preallocated cache (append per step, naive unfused
   attention over the cache) lands in Phase 2 so decode is incremental from the first
   on-device build. Phase 4 keeps the hard parts: fused GQA SDPA, norm/RoPE folding,
   dispatch reduction. Phase 3 gains a standalone fused dequant-matvec microbenchmark
   (achieved GB/s vs measured bandwidth, fixed short context).
3. **Layered quant oracle (Phase 3).** Exact match required at the dequant-tile layer
   (deterministic; where nibble-order/group-boundary bugs live). Matvec layer: ~1e-3
   relative tolerance vs CPU-quant reference, fp32 accumulation both sides (reduction
   order is the only difference). Adversarial packing fixtures: uneven group
   boundaries, extreme scales, all-zero groups, negative-heavy groups. Quality gate:
   top-1 agreement + mean KL vs mlx-lm 4-bit on fixture prompts + fixed-slice
   perplexity — graded "same band as MLX-4bit-vs-fp16," not exact match.
4. **Energy metric: battery-delta joules, sustained runs only.** ΔSoC ×
   health-scaled capacity; ≥8–10% SoC burn/run; idle-baseline subtraction; 80→70%
   band; ≥3 repeats, mean ± spread; 3–9 W plausibility anchor. Instruments Energy
   Log demoted to labeled unitless indicator. Burst energy undefined, not reported.
5. **Logit fixtures.** fp32 storage (fp16 spacing ~0.016 exceeds tolerance);
   full-vocab checkpoints at steps {0,1,mid,last}; per-step logsumexp/mean/std
   fingerprints; per-step top-64; tie-aware argmax via recorded top-1-vs-top-2
   margins. ~13 MB plain git. Regeneration script checked in, versions pinned.
6. **Reference oracle: HF transformers CPU fp32, tolerance ≤ 1e-3, no loosening
   escape hatch.** mlx-lm demoted to secondary ecosystem check; returns as Phase 3
   quality-gate comparator.
7. **Per-module activation fixtures are primary** — each CPU module unit-tested in
   isolation before pipeline wiring; same fixtures serve Phase 2–5 kernel diffs.
8. **CPU reference uses Accelerate cblas_sgemm** for ALL matmul-shaped work via one
   wrapper (validated against a test-only triple-loop on odd shapes). Scalar suite
   would take 5–15 h and stop being run. Elementwise logic stays hand-rolled.
9. **Outside-voice batch (all 12 accepted):** absolute success metric
   (≥ 0.75 × MLX measured, same session, canonical window 128–512); on-device
   bandwidth microbench is the roofline denominator; regenerate-loop sustained
   protocol; quantize embeddings + lm_head (fp16 head alone blows the roofline);
   Phase 2 fp16 gate pre-committed; model pinned BEFORE Phase 0a (2.5-vs-3 is an
   architecture fork: QKV bias vs QK-norm); Phase 0 baselines PROVISIONAL, Phase 6
   re-runs all engines same-session/same-OS; energy calendar cost budgeted; mmap vs
   wired-copy sustained-stability bench in Phase 2; dual GPU+wall timing from day
   one; numeric gates pre-committed per phase; cross-engine parity pins (greedy,
   pinned prompts, per-engine token counts, phys_footprint).

## OPEN — to be pinned before Phase 0a begins

- [ ] Exact model repo + revision (Qwen 2.5 vs Qwen 3 family fork decides module list).
- [ ] mlx-community 4-bit checkpoint provenance verified against the pinned base
      (or convert from pinned fp16 via mlx_lm.convert).
- [ ] Measured iPhone DRAM bandwidth (GB/s) — from the Phase 0 triad kernel.
- [ ] Absolute decode target = 0.75 × MLX measured decode tok/s (canonical window).
- [ ] Energy method validation result from the Phase 0 dry-run.

## 2026-08-20 — Backlog expanded to full-project DAG; review record archived

- Full deliberation record of the solidification review (all 20 findings verbatim,
  resolutions, coverage/failure-mode audits, per-phase spec obligations) archived
  at docs/reviews/2026-08-20-eng-review.md. DECISIONS.md stays the operative
  summary; the archive is the context source for future SPEC-P2..P6 sessions.
- docs/PRIORITIES.yaml now carries the whole project: concrete tasks for Phase 0
  (PIN-1, SETUP-1, P0B-1..4, P0A-1) and Phase 1 (P1-1..5, whose spec exists), and
  for Phases 2-6 only SPEC-Pn (write the spec, pre-commit gates) + Pn-EXEC
  (milestone placeholder, replaced by real tasks when the spec lands). This makes
  the just-in-time spec rule an enforceable DAG edge instead of convention.
- AUDIT-1 retargeted to depend on P1-5 (audit needs code to exist).

## 2026-08-21 — Architecture PDF generator incorporated (docs/generator/)

- Adopted the browser-agent's system-design PDF generator: make_diagrams.py
  (Figures 1-7) + build_pdf.py -> docs/architecture.pdf. Paths made repo-relative.
- Content drift fixed before first commit (v1.1): absolute success metric wording
  (>=0.75x MLX measured, canonical window), phys_footprint as the sole memory
  metric, "MLX ~61 tok/s" relabeled published/provisional, finding count corrected
  to 20, parity pins + regenerate-loop added to the methodology section.
- Its CLAUDE.md was NOT adopted (it predated the v2 hard rules); only its
  "Architecture document upkeep" section (regeneration triggers + anti-drift
  clause) was merged into ours. Root README.md added.

## 2026-08-21 — SETUP-1 scaffold landed; BLOCKED on missing Xcode (james)

- Swift package scaffold committed: root Package.swift (swift-tools-version 5.9),
  library target QwenMetalEngine (engine core, shared), executable target
  QwenMetalCLI exposed as product `qwen-metal-cli` (hyphens aren't valid module
  names, so the target is CamelCase and the product keeps the CLI-facing name),
  test target QwenMetalEngineTests with the placeholder XCTest. Platform floors:
  macOS 14 / iOS 17 (reversible; chosen for Metal feature parity headroom).
- ENVIRONMENT CONFLICT (SOP "spec conflicts with reality"): this machine has NO
  Xcode — only CommandLineTools (`xcode-select -p` = /Library/Developer/
  CommandLineTools; no Xcode bundle found). CLT lacks XCTest, so `swift test`
  fails with "error: XCTest not available". Verified working under CLT:
  `swift build` (Build complete!) and `swift run qwen-metal-cli` (banner prints).
  SETUP-1's exit condition ("placeholder XCTest green via `swift test`") cannot
  be verified until James installs Xcode and runs
  `sudo xcode-select -s /Applications/Xcode.app`. SETUP-1 stays in_progress with
  the blocker noted; NOT worked around (e.g. no swap to a non-XCTest framework).
  Downstream P0B tasks all require XCTest + the Metal compiler (also Xcode-only),
  so the install unblocks the whole Phase 0b chain.

## 2026-08-21 — Xcode blocker resolved; SETUP-1 closed (Xcode 27.0 beta via DEVELOPER_DIR)

- Xcode IS installed — as /Applications/Xcode-beta.app (Xcode 27.0, build
  27A5237l), which is why `xcode-select -s /Applications/Xcode.app` failed.
  James's sudo also can't run inside the agent session (no TTY for password).
- Resolution: no xcode-select switch needed. `DEVELOPER_DIR=/Applications/
  Xcode-beta.app swift test` runs the full toolchain. Agents use this env prefix
  for all swift test/build/run until xcode-select is switched system-wide.
- `swift test` result: "Executed 1 test, with 0 failures (0 unexpected)" —
  SETUP-1's exit condition verified; marked done; P0B-1, P1-2, P1-3 flipped to
  ready.
- CAUTION recorded: the only full toolchain on this machine is a BETA (Xcode 27
  beta, macOS 26 SDK line). Fine for scaffold/unit tests; before any benchmark
  row or numeric-gate commitment lands, note the toolchain build in the row per
  the benchmark protocol, and prefer a release Xcode once available.

## 2026-08-21 — Toolchain pinned to release Xcode 26.6 (beta caution retired)

- James installed release Xcode 26.6 (build 17F113) at /Applications/Xcode.app
  and switched xcode-select to it. `swift test` re-verified without any
  DEVELOPER_DIR prefix: "Executed 1 test, with 0 failures (0 unexpected)";
  `swift run qwen-metal-cli` prints the banner.
- The previous entry's beta caution is resolved: dev + benchmarks run on the
  release toolchain. Xcode-beta.app remains installed side-by-side; it is NOT
  to be used for benchmark rows. CLAUDE.md environment section updated
  (DEVELOPER_DIR prefix removed).

## 2026-08-21 — P0B-1 landed: Metal harness + dual-timing utility

- MetalContext (device/queue/library/pipeline setup, explicit error enum) and
  DispatchTiming land in Sources/QwenMetalEngine/Metal/. timedDispatch brackets
  the whole batch (create -> encode -> commit -> waitUntilCompleted) with
  CACurrentMediaTime and reads MTLCommandBuffer.gpuStartTime/gpuEndTime — both
  clocks share the mach host-time domain, so wall >= GPU holds by construction
  and dispatchOverhead = wall - gpu is the Phase 4 overhead metric (hard rule 7).
- Convention (reversible): Phase 0 toy kernels compile from source strings at
  runtime via device.makeLibrary(source:), not SPM-compiled .metal resources.
  Chosen so the test kernel can live in the test target and the engine ships no
  kernels before P0B-2; a precompiled-metallib path can be added when a phase
  needs it (e.g. iOS deployment of the triad bench may prefer it — revisit at
  P0B-4).
- Verified: `swift test` — "Executed 7 tests, with 0 failures (0 unexpected)"
  (6 new harness tests + placeholder). No numeric gates involved; timing sanity
  assertions (nonzero, start <= end, wall >= GPU) are structural, not tolerances.

## 2026-08-21 — P0B-2 gate pre-committed: saxpy GPU-vs-CPU tolerance

- Gate, set BEFORE the test was written or run (METHODOLOGY rule 2): saxpy GPU
  output vs hand-rolled CPU reference, max absolute element difference <= 1e-6,
  inputs drawn from [-1, 1] (seeded deterministic generator). Rationale: fp32 ulp
  at these magnitudes is ~1.2e-7; the Metal compiler may legally contract
  a*x + y into fma while the Swift reference rounds twice, so a few-ulp headroom
  is required — 1e-6 (~4-8 ulp) covers that while any real indexing/dispatch bug
  produces errors orders of magnitude larger. Per the standing rule this
  tolerance never loosens.
- Scope note: saxpy is elementwise, so the CPU reference is a hand-rolled loop —
  hard rule 8 (Accelerate sgemm wrapper) applies only to matmul-shaped work and
  is not implicated here. The P0B-3 matmul tolerance is NOT set by this entry;
  it gets its own pre-committed gate when P0B-3 starts.

## 2026-08-21 — P0B-3 gate pre-committed: naive fp16 matmul GPU-vs-CPU tolerance

- Kernel numeric design (convention-following): operands are fp16, the kernel
  accumulates in fp32 and rounds once to half on store — matching the project's
  recorded "fp32 accumulation both sides" convention (2026-08-20 entry, item 3)
  that Phase 2-5 kernels will use. "Naive fp16 matmul" in PRD 0b.4 names the
  operand type, not the accumulator.
- Gate, set BEFORE the test was written or run (METHODOLOGY rule 2): per output
  element, |gpu − ref| <= max(2^-9, 2^-9 · |ref|), where ref is the UNROUNDED
  fp32 CPU value accumulated from the same fp16 inputs; inputs drawn fp16 from
  [-1, 1] via the seeded deterministic generator. Rationale: with fp32
  accumulation on both sides, the GPU-vs-CPU difference is one half
  round-to-nearest on store (<= 2^-11 relative) plus fp32 reduction-order noise
  (orders of magnitude below half spacing at these K); 2^-9 gives ~4x headroom
  on the rounding term, and the matching absolute floor covers cancellation
  near zero, where relative error is unbounded but absolute error stays at
  accumulation-noise scale. Any real indexing/transpose/stride bug produces
  O(1) errors. Per the standing rule this tolerance never loosens.
- Oracle scoping: hard rule 8 (single Accelerate sgemm wrapper) binds the
  Phase 1 model CPU reference. The P1-3 wrapper does not exist yet and P0B-3
  does not depend on it in the DAG, so this toy-kernel test uses a TEST-ONLY
  naive fp32 triple loop — the same species of oracle P1-3 itself will be
  validated against. It lives in the test target and is not engine code.
