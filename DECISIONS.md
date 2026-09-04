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

- [x] Exact model repo + revision (Qwen 2.5 vs Qwen 3 family fork decides module
      list). → Pinned 2026-08-21: Qwen/Qwen3-1.7B (see PIN-1 entry below).
- [x] mlx-community 4-bit checkpoint provenance verified against the pinned base
      (or convert from pinned fp16 via mlx_lm.convert). → Verified via HF file
      history 2026-08-21 (see PIN-1 entry below); residual runtime check at P0A-1.
- [x] Measured iPhone DRAM bandwidth (GB/s) — from the Phase 0 triad kernel.
      → Measured 2026-08-22: 43.84 GB/s sustained (see entry below).
- [x] Absolute decode target = 0.75 × MLX measured decode tok/s (canonical window).
      → Committed 2026-08-22: 29.4 tok/s (0.75 × 39.2 warm-burst median; entry below).
- [x] Energy method validation result from the Phase 0 dry-run.
      → VALIDATED 2026-08-22: both engines 3.2–3.7 W, MLX 0.104 /
      llama.cpp 0.131 net J/token (see P0A-1 close-out entry).

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

## 2026-08-21 — P0B-4 gate + measurement protocol pre-committed: triad bandwidth microbench

- Correctness gate, set BEFORE the test was written or run (METHODOLOGY rule 2):
  triad GPU output a[i] = b[i] + s·c[i] vs hand-rolled fp32 CPU reference,
  max absolute element difference <= 1e-6 on sampled elements, inputs in
  [-1, 1] from a deterministic init pattern. Same species and rationale as the
  P0B-2 saxpy gate (elementwise, one multiply + one add; legal fma contraction
  on the GPU vs double rounding on the CPU costs a few ulp; a real
  indexing/stride bug costs orders of magnitude more). Never loosens.
- Measurement protocol, pinned BEFORE any number was produced (operationalizes
  the PRD's "report best sustained, not peak"):
  - Kernel: STREAM triad over float4 elements; bytes moved per iteration =
    3 × N × 4 (read b, read c, write a).
  - Working set: N = 96 × 2^20 fp32 elements per buffer (384 MiB each,
    1.125 GiB streamed per iteration) — satisfies the >= 1 GiB floor and
    dwarfs any Apple SLC so the number is DRAM, not cache.
  - Iterations: 2 warmup (discarded) + 10 measured. Per-iteration GB/s uses
    GPU timestamps (wall recorded alongside per hard rule 7; GB is 10^9 bytes).
  - Reported "sustained GB/s" = MEDIAN of the 10 measured iterations, with
    min/max spread alongside. Median over max because the roofline denominator
    must be a rate the decode loop can actually sustain, not a lucky burst.
- Scope: the Mac row this lands is dev-loop sanity only, marked PROVISIONAL.
  The roofline denominator for the project is the iPhone run of this same
  kernel (P0A-1, James), recorded here when it happens.

## 2026-08-21 — P0B-4 landed: triad microbench; Mac sustained 178.19 GB/s

- Measured (Mac, dev-loop sanity, PROVISIONAL): Apple M2 Pro, macOS 26.5.1
  (25F80), Xcode 26.6 (17F113), release build, pinned protocol: sustained
  (median) 178.19 GB/s, spread 172.47–179.85 GB/s, dispatch overhead
  ~0.2 ms/iteration. ~89% of M2 Pro's rated 200 GB/s — consistent with a
  DRAM-bound triad, i.e. the 1.125 GiB working set defeats the SLC and the
  protocol behaves. First row in benchmarks/results.md. This number is NOT
  the roofline denominator (that is the iPhone run, P0A-1).
- Correctness observation: the deterministic input pattern (values on the
  2^-11 grid) with s = 0.75 makes b + s·c exactly representable in fp32, so
  GPU and CPU agree bit-for-bit; the 1e-6 gate held with observed Δ = 0 on
  the full-array small run and on the benchmark's sampled elements. The gate
  stays as committed (headroom is for the general fma-contraction case).
- P0B-1 "revisit at P0B-4" resolved: runtime source compilation RETAINED for
  the triad kernel. `device.makeLibrary(source:)` is available on iOS at
  runtime, and a microbench has no startup-latency requirement, so a
  precompiled metallib is still not needed; the convention stands until a
  phase has a concrete reason (recorded then).
- Surfaced for James (P0A-1 prerequisite, noted in its backlog entry): the
  iPhone triad run needs a thin device shell — no iOS target exists until
  Phase 2, but P0A-1 precedes SPEC-P2 in the DAG. Options: a disposable
  scratch Xcode app importing QwenMetalEngine and calling
  TriadBandwidthKernel at the pinned protocol (agent can prepare the call
  site on request), or pulling the Phase 2 app scaffold earlier (a
  convention-setting decision that is James's to make, per AGENT_OPERATION).

## 2026-08-21 — PIN-1: model pinned — Qwen/Qwen3-1.7B (decided by James)

- **Pinned model: Qwen/Qwen3-1.7B, revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`** (main as of 2026-08-21).
  HF file history confirms the load-bearing files — model-*.safetensors,
  config.json, tokenizer.json, vocab/merges — are unchanged since the initial
  2025-04-28 upload; later commits touched only README (05-21), LICENSE
  (07-26), and tokenizer_config.json (05-19). Decision made by James from the
  comparison brief (Qwen2.5-1.5B vs Qwen3-1.7B); deciding factors: current
  generation (stronger writeup), QK-norm architecture, and the relative
  success metric making the extra bytes/token engine-neutral.
- **Family fork resolved → Phase 1 module list is Qwen3:** NO attention
  biases (config `attention_bias: false`); per-head Q/K RMSNorm (head_dim
  128) applied before RoPE. Verified config: hidden 2048, 28 layers, GQA
  16 Q : 8 KV heads, intermediate 6144 (SwiGLU), rope_theta 1e6, rms_norm_eps
  1e-6, vocab 151,936, tied embeddings (lm_head = embedding^T — invariant 2's
  quantized-lm_head requirement applies to the shared matrix).
- **Derived planning numbers** (from config, not measured): ~1.72B params;
  ~0.97 GB weights at 4-bit group-64; fp16 KV at 4K = 448 MiB (112 KiB/token);
  total resident ≈ 1.5 GB — inside the iPhone 15 Pro Increased-Memory-Limit
  envelope with headroom.
- **Benchmark device pinned: James's iPhone 15 Pro** (A17 Pro, 8 GB RAM).
  All official rows run there per PLAN.md protocol.
- **MLX baseline checkpoint: mlx-community/Qwen3-1.7B-4bit, revision
  `3b1b1768f8f8cf8351c712464f906e86c2b8269e`.** Card declares
  `base_model: Qwen/Qwen3-1.7B`; converted 2025-04-28 — the same weight
  revision as the pin (weights never changed after). PRD 0a.2 provenance is
  satisfied on file-history evidence; P0A-1 keeps a residual sanity check
  (or `mlx_lm.convert` from the pinned fp16 if anything looks off).
- **Thinking-mode parity pin (new cross-engine pin):** Qwen3-1.7B is a hybrid
  thinking model. ALL comparative rows run NON-thinking (`enable_thinking:
  false` or engine equivalent) with greedy sampling; the chat template of
  record is the one in the PINNED base revision (note: tokenizer_config.json
  was updated 2025-05-19, i.e. the pinned template postdates the mlx
  conversion — per-engine template application must be documented per the
  existing parity pins, and any `<think>` tokens in output invalidate a row.
- **Spec-vs-reality conflict, surfaced not worked around (session-discipline
  rule):** the pinned repo ships TWO safetensors shards
  (model-0000{1,2}-of-00002), while PLAN.md pins a single-file-only parser
  that rejects shards loudly. Resolution: the parser scope is UNCHANGED; a
  one-time offline consolidation step (Python, tools/, pinned deps — lands
  with P1-1) merges the pinned shards into the single-file artifact the
  engine consumes. The consolidated file's provenance (source revision +
  script) gets recorded when it's produced.

## 2026-08-22 — P1-1 landed: reference/fixture tooling + committed oracle set

- **tools/ shipped** (pins.py as single source of truth, dump_reference.py,
  dump_mlx.py, consolidate_shards.py, requirements.txt, README.md). Exact
  version pins: Python 3.14.6, torch 2.13.0, transformers 5.15.1, tokenizers
  0.22.2, safetensors 0.8.0, numpy 2.5.2, huggingface_hub 1.28.0, mlx 0.32.1,
  mlx-lm 0.31.3. Regeneration commands documented in tools/README.md;
  regenerating with unpinned versions remains a bug.
- **Fixture protocol pins** (operationalizing phase-0-1.md; convention-
  following, recorded for exactness): checkpoint steps = {0, 1, 24, 49}
  0-based ("mid" = 24); greedy = torch.argmax (first-index tie-break), NO EOS
  stop (fixtures always cover all 50 steps); primary dump uses the HF KV-cache
  decode path and sdpa attention (both recorded in manifest.json); scalar
  fingerprints computed in float64 over the fp32 logit vector; blobs are raw
  little-endian with dtype/shape/sha256 per manifest entry — deliberately NOT
  safetensors, so the oracle set has zero dependency on the engine parser
  under test (oracle-independence, METHODOLOGY rule 1). The 5 pinned prompt
  strings live in tools/fixture_prompts.json; the 6 activation hook points are
  enumerated in the manifest.
- **Committed set: 36 blobs, 12.53 MB** (budget ~13 MB, plain git) under
  tests/fixtures/qwen3-1.7b/ + manifest.json with per-blob sha256.
  tests/test_fixtures.py (stdlib-only, root .venv pytest) validates integrity,
  spec-required contents, and pins ↔ requirements.txt consistency: 8 tests.
- **Reproducibility VERIFIED:** full second run of dump_reference.py produced
  byte-identical output — 36/36 blobs same sha256.
- **Near-tie observation:** min top1-vs-top2 margin in the set is 0.0048
  (short_english); exactly 1 of 250 steps has margin < 1e-2 — the tie-aware
  argmax design (P1-5) has a real exercising case in the fixtures.
- **Chat-template observation (for P1-5 Swift work):** enable_thinking=false
  renders an EMPTY think block (`<think>\n\n</think>`) in the assistant
  preamble — that is the correct non-thinking form, not thinking-mode leakage.
  tokenizer_ids.json records the fully rendered input_text, so Swift tests can
  tokenize the recorded string and stay independent of template re-rendering.
- **mlx-lm secondary dump (loose, argmax-level, NOT an oracle — as designed):**
  4/5 prompts agree with the fp32 oracle at step 0; divergence onset at steps
  {1, 4, 1, 0, 4}. The step-0 flip (non_ascii) sits on a 0.139 step-0 margin —
  unremarkable for a 4-bit comparator. Text is coherent on all 5. Recorded in
  mlx_secondary.json; returns as the Phase 3 quality-gate comparator.
- **Consolidated single-file artifact PRODUCED (provenance per PIN-1 entry):**
  models/qwen3-1.7b-70d244cc.safetensors — 4.064 GB, 311 tensors, sorted
  names, bf16 preserved, provenance in __metadata__ (source repo + revision),
  self-checked byte-for-byte against the source shards.
  sha256 = 8538a19cec4c28dce3b784010dfba63842546963feec21b20ef5abdd3944f5f5.
  models/ was already gitignored; artifact is local-only, never committed.

## 2026-08-22 — P0A-1 prep: device shell (1A), GGUF pin (local convert), prompt set drafted

Decisions by James this session; agent prepared the harnesses (P0A-1 stays
`ready`, owner james — the on-device runs are his).

- **Device shell = disposable scratch app (option 1A).** The iPhone triad run
  uses a throwaway Xcode project OUTSIDE the repo importing QwenMetalEngine;
  the Phase 2 app scaffold is NOT pulled early (just-in-time rule preserved).
  Agent-prepared call site + setup/run/record instructions:
  benchmarks/device-shell/{TriadRunnerView.swift, README.md}.
- **llama.cpp GGUF pin = LOCAL CONVERSION (decided after investigation).**
  Finding: the official Qwen/Qwen3-1.7B-GGUF repo uploaded Q4_K_M/Q5/Q6
  quants 2025-05-08 and deleted them the SAME day (bare commit messages, no
  stated reason; README quant list edited from "q4_K_M, q5_0, q5_K_M, q6_K,
  q8_0" to "q8_0" — deliberate catalog change; official README separately
  recommends presence_penalty 1.5 for quantized models "to suppress
  repetitive outputs"). James chose conversion from our pinned base over the
  official-but-withdrawn revision (7fb011e9) and community quants.
  - Toolchain pin: llama.cpp release tag b9999 = commit 47c78692, cloned at
    ~/Projects/llama.cpp (also the source for the Phase 0a iOS runner build).
  - Recipe: convert_hf_to_gguf.py (llama.cpp's own pinned convert env,
    torch 2.11.0 / transformers 4.57.6) from the pinned HF snapshot
    @ 70d244cc, --outtype bf16 → llama-quantize Q4_K_M; intermediate bf16
    GGUF deleted.
  - Artifact: models/qwen3-1.7b-70d244cc-Q4_K_M.gguf, 1217.35 MiB (5.03 BPW),
    sha256 72b1b7b9ad563f21862ae60cd884c8911105ca8a214d5149ee33115575c52db4.
    Local-only (models/ gitignored); Phase 6 reuses the same recipe.
  - Smoke-tested on Mac via llama-completion (raw rendered prompt, greedy):
    coherent output, no <think> tags, ~1153 tok/s prompt / ~117 tok/s
    generation (M2 Pro, dev sanity only — not a benchmark row).
  - CAUTION for the device session: in this llama.cpp build, `llama-cli` is a
    chat TUI that ignores -no-cnv and re-templates input (double templating);
    use `llama-completion` for any completion-mode run. Cross-engine variance
    observed and expected: llama.cpp tokenizes the rendered decode-essay
    prompt to 92 tokens vs HF's 84.
- **Benchmark prompt set DRAFTED (parity pin; becomes load-bearing at first
  baseline row):** benchmarks/prompts/{decode-essay,prefill-summarize}.txt +
  non-thinking rendered forms (tools/render_bench_prompts.py) + protocol
  README. Verified empirically (mlx-lm 4-bit, greedy): decode-essay = 84
  HF-rendered tokens, runs ≥600 tokens with NO EOS → safely covers the
  canonical 128–512 window; prefill-summarize = 852 HF-rendered tokens,
  EOSes at ~425 → pinned as PREFILL-ONLY (role separation recorded in the
  prompts README). Sustained regenerate-loop operationalized: restart on EOS
  or context-fill, whichever first, identical for all engines.
- **Runbook:** benchmarks/phase0-runbook.md sequences the whole device
  session (triad → MLX → llama.cpp → energy dry-run) with row templates and
  the close-out list that resolves the three remaining OPEN items (measured
  GB/s, 0.75×MLX target, energy method validation).

## 2026-08-22 — MEASURED: iPhone DRAM bandwidth 43.84 GB/s — the roofline denominator

- **Sustained 43.84 GB/s** (median of 10 measured iterations), spread
  42.19–44.45 GB/s, dispatch overhead ~0.6–1.1 ms/iteration. Run by James on
  the pinned device per the pinned P0B-4 protocol (1.125 GiB streamed/iter,
  2+10 iters, GPU-timestamp basis, wall alongside): iPhone 15 Pro (A17 Pro),
  iOS 26.5.2, battery health 85%, >50% charge, rested-to-ambient (procedural
  check — no instrumented temp readout exists on iOS; recorded as procedure,
  which is how all future rows record it), Release build, Xcode 26.6 (17F113),
  scratch device shell (decision 1A). Row appended to benchmarks/results.md.
- **Repeatability:** three prior same-session runs in Debug config gave
  medians 43.28 / 43.67 / 43.89 (spread 42.43–44.36, overhead ~0.5–0.7 ms) —
  within ~1% of the Release figure, as expected since GB/s derives from GPU
  timestamps and the kernel is runtime-compiled (host build config only
  perturbs dispatch overhead). Recorded figure = the Release run (the
  protocol-conforming config). No downward drift across four runs ⇒ no
  thermal throttling at this workload.
- **Plausibility:** 43.84 = ~85.6% of A17 Pro's rated 51.2 GB/s (LPDDR5);
  the Mac dev row achieved ~89% of rated — same achievable-fraction ballpark
  on two chips says the working set defeats the SLC and the number is DRAM.
- **Derived planning number (NOT a gate; Phase 3/4 gates get their own
  pre-committed fractions):** with 4-bit weights ≈ 0.97 GB and canonical-window
  KV/activation traffic, bytes/token ≈ 1.0 GB ⇒ decode ceiling on the order
  of ~42–43 tok/s. Every roofline fraction from here on derives from 43.84.
- **PLAN.md check:** v2 contains no stale assumed bandwidth/tok-s figures to
  replace — invariant 1 already delegates to the DECISIONS.md measured figure,
  so the PRD's "PLAN.md numbers derive from measured" criterion is satisfied
  with no PLAN.md edit (verified by grep).
- Remaining P0A-1 device work: MLX + llama.cpp baseline rows, absolute
  target commit (0.75 × MLX), energy method dry-run.

## 2026-08-22 — MLX baseline measured; ABSOLUTE DECODE TARGET COMMITTED: 29.4 tok/s

- **MLX warm-burst decode (median of 3): 39.2 tok/s** (39.2 / 39.4 / 38.4,
  identical 1488-token greedy outputs — determinism confirmed). Full rows +
  session metadata in benchmarks/results.md (PROVISIONAL). Run by James:
  LLMEval @ mlx-swift-examples `378f2449` + 2 pinned parity edits, deps
  mlx-swift-lm 3.31.3 / mlx-swift 0.31.4, checkpoint 3b1b1768, Release,
  iPhone 15 Pro, iOS 26.5.2, battery health 85%. Zero <think> content.
- **TARGET (James's basis decision: warm-burst median, canonical-window
  proxy): 0.75 × 39.2 = 29.4 tok/s.** Caveat recorded: the app reports
  overall generation rate, not the strict 128–512 window; Phase 6 re-measures
  MLX same-session with windowed instrumentation. Per hard rule 6 this target
  does not loosen; a Phase 6 MLX re-measure recomputes the comparison, not
  this planning gate.
- **Roofline context:** 39.2 ≈ 89% of the ~44 tok/s naive ceiling
  (43.84 GB/s ÷ ~1.0 GB/token) — MLX is near-roofline on this device, so
  0.75× is a demanding target, and beating MLX outright would require
  near-perfect bandwidth utilization.
- **Thermal finding (sustained, 5 min regenerate loop, 6 generations):**
  38.3 → 26.7 tok/s (−30%), smooth decline, no stutter. Sustained rows and
  the energy dry-run operate in this throttled regime; burst vs sustained
  must never be compared across engines without matching regime.
- **Prefill: ~370 tok/s** (862 app-side prompt tokens ÷ 2.328 s TTFT).
- **Template-delta observation:** LLMEval's own template renders ~10 more
  tokens than the pinned rendered form (862 vs 852) — likely a default
  system message. Acceptable under the documented "engine applies its own
  template" feeding mode for PROVISIONAL rows; Phase 6 should pin the system
  prompt (or feed rendered forms everywhere) for the publishable head-to-head.
- Remaining P0A-1: llama.cpp baseline rows, energy method dry-run.

## 2026-08-22 — llama.cpp baseline measured; example-app harness defects found + fixed

- **llama.cpp warm-burst decode (median): 32.44 tok/s ≈ 83% of MLX's 39.2.**
  Prefill ≈ 452 tok/s (vs MLX ~370 — llama.cpp's stronger leg). Sustained
  5-min: 31.42 → 20.88 (−34%; MLX showed −30% — same thermal envelope).
  Full rows + session metadata in benchmarks/results.md (PROVISIONAL).
  Checkpoint: our locally converted Q4_K_M (5.03 BPW). Zero <think> content.
- **Harness lesson (recorded because Phase 6 re-runs must not repeat it):**
  the upstream llama.swiftui example was NOT measurement-grade. Defects
  found via cross-checking against the app's own bench + the Mac roofline:
  (1) per-token SwiftUI full-transcript re-layout + per-token console
  printing throttled generation itself (2048-token runs read 10.5–14.3 t/s
  vs a true ~31); (2) reported t/s divided by the length CAP, not actual
  tokens generated; (3) completion state (`is_done`, `n_decode`) never
  reset — every post-first generation in a session no-oped instantly;
  (4) `parse_special=false` would have tokenized the rendered template's
  markers as literal text; (5) batch hardcoded to 512 — our 852-token
  prefill prompt would overflow it. All patched (diffs in
  ~/Projects/llama.cpp, commented `qwen-metal P0A-1`); prompts now BUNDLED
  in-app after Universal Clipboard expiry silently substituted stale raw
  text for the rendered form twice (72-token signature caught it both times).
- **Metal API Validation costs ~17% decode** (tg128 26.11 validation-on
  rested vs 31.55 validation-off; burst runs confirm ~31–35 with it off).
  The MLX session ran validation ON → MLX's 39.2 and the 29.4 target are
  possibly UNDERSTATED. The committed target stands (hard rule 6 — it never
  loosens; if anything the true bar is higher). ACTION for Phase 6 (and
  opportunistically sooner): re-run MLX with validation off; benchmark
  protocol addendum — all future rows record the validation setting, default
  OFF.
- **Roofline interpretation finding (seeds BW-1):** rough BPW math puts both
  engines at ~96–102% of the 43.84 GB/s triad figure (llama.cpp 32.44 ×
  ~1.3 GB/token ≈ 42 GB/s; MLX 39.2 × ~1.14 ≈ 44.7). Decode is
  read-dominated; a 2-read+1-write triad understates read-mostly achievable
  DRAM bandwidth, so the roofline denominator is likely conservative by
  ~5–10% for decode-shaped traffic. Backlog task BW-1 added: read-only
  streaming bandwidth microbench variant to bound this properly. Until then,
  roofline fractions quoted against 43.84 carry this caveat.
- Bench-after-sustained observation: pp512 fell to 265 ± 23 (from 431 ± 51
  rested) — prefill (compute-bound) throttles much harder than decode;
  auxiliary bench runs must record thermal state.
- Remaining P0A-1: energy method dry-run (MLX, sustained); phys_footprint
  for llama.cpp not captured — grab the gauge peak during any later run.

## 2026-08-22 — P0A-1 CLOSED: energy method validated; Phase 0 exit complete

- **Energy dry-run (one cycle PER ENGINE, per PRD acceptance): method
  VALIDATED.** MLX 0.104 net J/token (81→71%, 1055 s, 32,840 tokens,
  3.67 W gross / 3.24 W net); llama.cpp 0.131 (69→59%, 1057 s, 26,132
  tokens, 3.66/3.23 W). Idle floor 1%/15 min ≈ 0.43 W (LLMEval foregrounded,
  scaled pro-rata — method detail pinned). SoC quantization ⇒ ~±12% error
  bars. Both inside the 3–9 W anchor. Full table in benchmarks/results.md.
  Recorded deviation: llama.cpp band 69→59% (not 80→70) — fine for method
  validation; Phase 6 comparative rounds use the pinned band.
- **Findings folded into the protocol for Phase 6:** (1) energy/sustained
  measurements DETACHED only (attached −30% "thermal" was partly harness);
  (2) validation overhead is engine-dependent (+1% MLX vs ~17–21%
  llama.cpp) — setting recorded per row, default OFF; MLX validation-off
  spot check 39.6 confirms the 29.4 target wasn't materially understated;
  (3) phys_footprint corrections: MLX 1.02 GB (gauge; earlier 923 MB was
  the app's activeMemory meter), llama.cpp 307 MB — an mmap accounting
  artifact, NOT an efficiency claim (invariant 3's asymmetry, now measured);
  (4) sustained thermal is engine-level: MLX −9% vs llama.cpp −45% at
  identical ~3.7 W (llama.cpp ends below the 29.4 target — sustained
  framing matters).
- **PRD-phase-0 acceptance walk (all criteria MET):**
  1. Model repo+revision pinned before first baseline row ✓ (PIN-1).
  2. Baseline table, 2 engines, prefill/decode/memory, burst+sustained,
     cold+warm, full annotations ✓; energy ≥1 validated battery-delta row
     per engine ✓. Honest gaps (recorded, non-blocking, PROVISIONAL rows):
     cold captured for decode only (prefill cold differs only via TTFT);
     decode rates are app-reported overall rates, not strictly windowed;
     llama.cpp memory subject to the mmap accounting caveat.
  3. Measured iPhone DRAM bandwidth ✓ (43.84 GB/s; PLAN.md derives from it).
  4. Absolute target committed ✓ (29.4 tok/s = 0.75 × 39.2).
  5. saxpy/matmul/triad + dual-timing XCTest green on macOS ✓ (30 tests).
  6. All Phase 0 rows marked PROVISIONAL ✓.
- **Phase 0 is fully exited** (P0B-1..4 done earlier; P0A-1 done now).
  SPEC-P2 remains blocked on P1-5 only. Harness provenance: local branches
  qwen-metal-p0a1 (llama.cpp 97e552a+bceddff+…, mlx-swift-examples
  47c36a0+992118b) archived as benchmarks/patches/*.patch.

## 2026-08-22 — P1-2 landed: safetensors parser + config loader (engine io module)

- **Shipped:** Sources/QwenMetalEngine/IO/{SafetensorsFile,ModelConfig}.swift
  + SafetensorsFileTests (15) + ModelConfigTests (11); full suite 56 tests,
  0 failures. All 7 enumerated edge cases from phase-0-1.md covered in the
  same change as the code. No numeric gate involved: both upcasts are exact
  (bf16 = fp32 top half by bit shift; fp16→fp32 exact by construction), so
  the tests assert with == rather than a tolerance.
- **Parser pins honored:** hand-written (no swift-safetensors), raw mmap()
  (PLAN invariant 5), single-file only — any `*.index.json` sibling triggers
  a loud reject whose message names tools/consolidate_shards.py as the
  remedy. Validation includes offset bounds, dtype/shape byte-size match,
  overflow-checked shape products, and overlapping-range detection.
- **Family-flag conventions (P1-4 relies on these):** `attention_bias`
  honors an explicit config key, else defaults by family (qwen3 → false,
  qwen2 → true, unknown family → hard error rather than a guess).
  `usesQKNorm` is DERIVED from `model_type == "qwen3"` — HF configs carry no
  explicit qk-norm key, so the architecture fork pinned in the PIN-1 entry
  is encoded here. `head_dim` falls back to hidden_size / num_attention_heads
  when absent (Qwen 2.5-style configs); the pinned Qwen3 config is explicit.
- **Scope note:** fp32 materialization per tensor (~7 GB resident for Phase 1
  on the Mac) is the accepted phase-0-1.md behavior; raw fp16/bf16 views for
  GPU upload are a Phase 2 concern and deliberately not built (YAGNI +
  just-in-time spec rule).

## 2026-08-23 — P1-3 gate pre-committed: sgemm wrapper vs triple-loop tolerance

- Gate, set BEFORE the test was written or run (METHODOLOGY rule 2): per output
  element, |sgemm − tripleLoop| <= 2·γ_K·Σ_p |a_ip|·|b_pj|, where
  γ_K = K·u/(1−K·u) with unit roundoff u = 2^-24 (fp32), and the Σ|a||b| term
  is accumulated in the test alongside the triple-loop reference. Inputs are
  fp32 drawn from [-1, 1] via the seeded deterministic generator (SplitMix64,
  same as the P0B tests). Rationale: both sides consume identical fp32 bits and
  accumulate in fp32, so the only legitimate divergence is reduction order.
  γ_K·Σ|a||b| is the classical rigorous forward-error bound on a K-term fp32
  dot product; naive left-to-right summation (the test oracle) satisfies it,
  and Accelerate's blocked/SIMD/wider-accumulator variants satisfy it or
  tighter, so twice the bound covers the worst-case sum of both sides' errors
  with no eyeballed headroom constant. The absolute-floor term used by earlier
  gates is unnecessary here: the bound already scales to zero exactly when the
  products are all zero, where both sides are exact. Any transpose /
  leading-dimension / stride bug produces O(1) errors, orders of magnitude
  above the bound. Per the standing rule this tolerance never loosens.
- Exactness carve-out (== assertions, not a tolerance, same species as the
  P1-2 upcast tests): small-integer cases whose every product and partial sum
  is exactly representable in fp32 must match the triple loop bit-for-bit.
- API convention (reversible, convention-following — surfaced, not blocking):
  the wrapper is row-major fp32 with transposeA/transposeB flags so later
  callers (per-head QK^T, tied-embeddings lm_head) never materialize
  transposes; all four flag combinations are covered by the validation test.
  The naive triple loop lives ONLY in the test target (phase-0-1.md build
  step 3), exactly like the P0B-3 toy-kernel oracle.

## 2026-08-23 — P1-3 landed: Accelerate sgemm wrapper (hard rule 8 path)

- **Shipped:** Sources/QwenMetalEngine/BLAS/Sgemm.swift (BLAS.sgemm, the
  single matmul path for the CPU reference) + SgemmTests (7 tests: exact
  small-integer case, odd shapes 67x129x45 and 301x257x173, all four
  transpose-flag combinations on 35x53x29, vector-shaped decode edges,
  input non-mutation, explicit input-validation errors). Full suite
  63 tests, 0 failures. The pre-committed gate (previous entry) held
  unmodified on the first run.
- **Implementation notes:** classic cblas_sgemm interface with Int32
  dimensions — the SDK does not expose __LAPACK_int to Swift; realistic
  model dimensions sit far below Int32 range. Input validation reuses
  KernelInputError (its doc comment widened to cover the BLAS wrapper).
  The test-side triple loop also accumulates Σ|a||b| per element as the
  gate's error-bound basis, so the oracle is ~20 lines rather than the
  spec's ~15 — the extra lines are the bound computation, not model logic.

## 2026-08-23 — P1-4 gate pre-committed: per-module activation tolerances

Set BEFORE any P1-4 test was written or run (METHODOLOGY rule 2), following
the P0B-2/P0B-3/P0B-4/P1-3 precedent of agent-committed derived gates.
Each Swift module is fed its reference input slice from
tests/fixtures/qwen3-1.7b/activations/ (prompt short_english, seq 5) and
diffed against its reference output slice — modules are isolated, so a
failure names the module, not the pipeline.

- **Hidden-state slices** (layer0_pre_attn_norm_output, layer0_attn_output,
  layer0_block_output, last_layer_output, final_norm_output): per element,
  |Δ| <= max(5e-5 · M, 1e-6), where M = max|ref| over that slice.
  Derivation, not a new number: the phase's already-committed logit gate is
  1e-3 absolute at logits whose typical magnitude is ~20 (phase-0-1.md
  correctness harness) — a relative resolution of 5e-5. These modules sit
  strictly earlier in the network than the logits, where legitimate
  reduction-order divergence has compounded less, so granting them the
  end-to-end relative resolution is conservative in the right direction.
  Scaling by slice max-abs (not per-element |ref|) is the same species as
  the P0B-3 gate's |ref| term: dot-product error scales with the magnitude
  of the accumulated terms, and near-zero outputs legitimately carry
  absolute error inherited from large terms in the same reduction. The
  1e-6 floor covers a hypothetical all-near-zero slice; it is orders of
  magnitude above fp32 noise at these magnitudes either way. Bug-scale
  errors (wrong rotation half, transposed projection, bad GQA head map,
  missing causal mask) are O(1) relative — 3-4 orders above the gate.
- **Embedding lookup: exact ==, no tolerance** (same species as the P1-2
  upcast and P1-3 small-integer carve-outs). It is a row copy of
  exactly-upcast bf16 weights; HF's fp32 load performs the identical exact
  upcast, so any difference is a bug.
- **Isolated lm_head check** (final_norm_output fixture in, full-vocab
  logits out, vs logits_step0000): the committed phase gate applies
  unchanged — |Δ| <= 1e-3 absolute. No new number is introduced for
  logit-shaped output, and the single-matmul divergence in this isolated
  check is far below the end-to-end budget.

Per the standing rule (hard rule 6), none of these loosen — a failure is
a bug signal, never a tolerance-adjustment signal.

## 2026-08-23 — P1-4 landed: CPU reference modules (Qwen3 family) + activation oracle tests

- **Shipped:** Sources/QwenMetalEngine/Model/{ModelError, Embedding, RMSNorm,
  RoPE, Attention, MLP, TransformerBlock, QwenModel}.swift +
  ActivationFixtureTests (7 per-module oracle tests vs the dumped HF fp32
  slices, isolated: each module is fed its reference INPUT slice) +
  ModelModuleUnitTests (13 synthetic/error-path tests needing no checkpoint).
  Full suite 83 tests, 0 failures. **All pre-committed gates (previous entry)
  held unmodified on the first run** — embedding matched exactly; every
  hidden-state slice passed 5e-5·max|ref|; the isolated lm_head check passed
  the committed 1e-3 logit gate.
- **Family encoding:** Qwen3 only, per PIN-1 — per-head Q/K RMSNorm applied
  BEFORE RoPE (HF order), half-split rotation with fp32 angle tables
  mimicking HF's fp32 path, no QKV biases. QwenModel refuses any config that
  is not Qwen3-shaped (unsupportedFamily) instead of growing an architecture
  registry (PLAN.md non-goals). Tied embeddings honored: lm_head shares the
  embedding table storage.
- **Hard rule 8 honored:** QKV/o/MLP/lm_head projections AND per-head
  QK^T / PV all route through BLAS.sgemm; HF's [out, in] weight storage is
  consumed via transposeB so no transpose is ever materialized. Elementwise
  logic (RMSNorm, RoPE, softmax max-subtracted, SiLU, embedding lookup) is
  hand-rolled fp32 Swift.
- **Checkpoint handling in tests:** ActivationFixtureTests uses the
  local-only consolidated artifact (models/qwen3-1.7b-70d244cc.safetensors),
  verifies its __metadata__.source_revision equals the PIN-1 revision before
  trusting it, and XCTSkips with a regeneration hint when the file is absent.
  The pinned config.json values are mirrored inline in the test (the
  from-disk config path gets exercised by P1-5's CLI).
- **Measured (dev-loop, Mac, debug build):** one-time fp32 materialization of
  the checkpoint dominates the suite at ~190s; everything after loads in
  milliseconds-to-seconds (28-layer seq-5 forward ≈ 3.2s, lm_head ≈ 1.7s).
  Seeded follow-up IO-1 (vectorized upcast) rather than optimizing in-diff.
- P1-5 flipped to ready (last Phase 1 task: decode loop + CLI + full
  logit-match suite + tokenizer equivalence).

## 2026-08-23 — IO-1 rank bumped 25 → 11.5 (decided by James)

The upcast-vectorization follow-up runs BEFORE P1-5 rather than as filler:
P1-5's dev loop re-pays the ~190s debug-mode checkpoint materialization on
every `swift test` iteration, while IO-1 is a <~1h fix that amortizes within
a handful of runs. It is pulled forward as pure dev-loop economics, NOT as a
phase gate: Phase 1's exit criteria are unchanged, and if IO-1 stalls it is
skipped, not fought. Constraint reaffirmed for whoever picks it up: the
upcast must stay exact (the P1-2 == tests are the gate) and weights stay
mmap-only (PLAN.md invariant 5).

## 2026-08-23 — IO-1: upcast vectorized via Accelerate (measured)

- **Change:** `SafetensorsFile.fp32Values` now converts through Accelerate
  instead of a scalar Swift loop: fp16 via `vImageConvert_Planar16FtoPlanarF`
  (hardware widening, exact), bf16 via an all-exact vDSP chain
  (`vDSP_vfltu16` → ×2^16 `vDSP_vsmul` → `vDSP_vfixu32`, i.e. bits<<16
  materialized through exact fp32 arithmetic), chunked at 1M elements.
  Accelerate is prebuilt, so debug (-Onone) test runs no longer pay the
  unoptimized-loop tax. Sources with a 2-byte-misaligned data offset (legal
  per the format) take the retained scalar path.
- **Exactness gate held:** the P1-2 `==` tests pass unmodified, and a new
  exhaustive sweep (all 65,536 bit patterns per dtype, bit-for-bit vs the
  scalar reference — NaN payloads, infs, subnormals included) plus an
  odd-offset fallback test landed with the change. No tolerance introduced.
- **Invariants intact:** weights still load via mmap only (PLAN.md inv. 5);
  conversion reads straight from the mapping; no dequant/layout change.
- **Measured (dev-loop, Mac M2 Pro, debug build):** checkpoint-loading
  oracle suite (ActivationFixtureTests, 7 tests incl. full 1.7B fp32
  materialization) 190s → **11.0s**; full 89-test suite now 21.0s cold /
  16.1s warm. The P1-5 edit-test iteration tax this rank bump targeted is
  gone (~17× on the materialization).

## 2026-08-23 — P1-5 gates pre-committed: logit-suite scalars, top-64, tie epsilon

Set BEFORE any P1-5 test was written or run (METHODOLOGY rule 2), following
the P0B-2/P0B-3/P1-3/P1-4 precedent of agent-committed derived gates. All of
these are derivations of the phase's single committed number — the 1e-3
absolute full-vocab logit gate (phase-0-1.md; DECISIONS 2026-08-20 item 6).
No new tolerance is introduced.

Premise for every-step (not just checkpoint-step) assertions: the suite
teacher-forces the REFERENCE argmax token at each step, so both engines see
byte-identical token prefixes at every one of the 50 steps. Divergence at any
step is therefore pure reduction-order noise on one forward pass — the same
species the 1e-3 checkpoint gate already bounds — and never compounds across
steps through the discrete token channel. (Teacher-forcing also keeps steps
after a hypothetical tie-exempt argmax flip comparable; our own argmax is
asserted separately, and the CLI decode loop self-feeds.)

- **Full-vocab checkpoints (steps 0, 1, 24, 49):** per element
  |Δlogit| <= 1e-3. The committed phase gate, applied unchanged.
- **Per-step scalar fingerprints (all 50 steps, float64 over the fp32
  vector, matching the manifest protocol):**
  - |Δ logsumexp| <= 1e-3 — logsumexp is 1-Lipschitz in the sup norm.
  - |Δ mean| <= 1e-3 — the mean of per-element deviations each <= 1e-3.
  - |Δ std| <= 2e-3 — std is 2-Lipschitz in the sup norm (mean shift and
    deviation shift each contribute at most the element bound).
- **Per-step top-64 (all 50 steps):** our logits gathered at the REFERENCE
  top-64 indices, per element |Δ| <= 1e-3 vs the stored values. Same gate,
  same species; asserting at reference indices (rather than comparing our
  own top-64 set) keeps the assertion permutation-free under legitimate
  near-tie reordering.
- **Tie-aware argmax (all 50 steps):** exact top-1 match wherever the
  recorded top1-vs-top2 margin >= epsilon_tie = 2e-3; below it, assert our
  top-1 is in {reference top-1, reference top-2}. Derivation: with both
  sides within 1e-3 per element of the true logits, an argmax flip requires
  the reference margin < 2 x 1e-3; a flip at any larger margin cannot be
  reduction-order noise and stays a hard failure. The current fixture set's
  minimum margin is 4.8e-3 (1 of 250 steps below 1e-2 — P1-1 entry), so NO
  step is exempt today; the exemption path gets a synthetic unit test so it
  is exercised code, not dead code.

Per the standing rule (hard rule 6), none of these loosen — a failure is a
bug signal, never a tolerance-adjustment signal.

## 2026-08-23 — P1-5 landed: decode loop + CLI + logit suite + tokenizer equivalence — Phase 1 EXITED

- **Shipped:** Sources/QwenMetalEngine/Decode/{DecodeLoop,QwenModel+Decode}
  .swift (greedy first-index-tie-break argmax; EOS/max-token/context stops;
  full re-forward per step — KV cache stays Phase 2; temperature NOT built,
  spec-optional), Tokenizer/TextTokenizer.swift (swift-transformers adapter,
  local-folder load only), IO/ModelDirectory.swift (--model-dir resolution
  with named-what's-missing errors), ModelConfig gains optional
  eos_token_id parsing, CLI `generate` subcommand. Tests: LogitMatchSuiteTests
  (5 prompts), TokenizerEquivalenceTests, DecodeLoopTests,
  ModelDirectoryTests, +3 ModelConfig eos tests, TieAwareArgmaxRuleTests
  (synthetic exemption-path pin). ActivationFixtureTests' checkpoint plumbing
  extracted to a SharedCheckpoint helper so both oracle classes share ONE
  ~7 GB fp32 model (RoPE table 64 → 256; gates untouched).
- **Full logit-match suite PASSED, first run, all gates unmodified**
  (the pre-committed P1-5 gates entry above): all 5 prompts x 50
  teacher-forced steps — full-vocab |Δ| <= 1e-3 at steps {0,1,24,49},
  per-step float64 fingerprints (lse/mean <= 1e-3, std <= 2e-3), per-step
  top-64 at reference indices <= 1e-3, argmax exact top-1 at every one of
  250 steps (no step was tie-exempt at epsilon 2e-3, as predicted from the
  recorded margins). Full suite: **118 tests, 0 failures** in 1936s
  (debug; logit suite dominates — follow-up DEV-1 seeded).
- **Tokenizer observations (spec-required):** swift-transformers pinned
  **exact 1.3.3** (Package.swift; latest release, 2026-05-16). Encoding is
  id-identical to the pinned Python tokenizers 0.22.2 dump on all 5 fixture
  prompts, including the fully rendered chat_template string with the empty
  think block — zero disagreements, so the "log and match Python" clause was
  never exercised. eosTokenId resolves to 151645 (<|im_end|>) from
  tokenizer_config.json, matching config.json's eos_token_id; byte-level
  BPE decode round-trips the raw prompt exactly. Chat templating stays
  Python-side (fixtures record rendered strings); Swift renders none.
- **Local model-dir convention:** the CLI consumes a directory holding
  exactly one .safetensors + config.json + tokenizer.json +
  tokenizer_config.json. models/ now carries the three JSONs downloaded at
  the pinned revision 70d244cc (local-only, never committed, like the
  checkpoint): config.json sha256 1ddb5b89…, tokenizer.json aeb13307…,
  tokenizer_config.json d5d09f07… (full hashes reproducible via
  `shasum -a 256 models/*.json`).
- **CLI verified (exit criterion):** `generate --prompt "The capital of
  France is" --max-tokens 24` → " Paris. The capital of Italy is Rome. The
  capital of Spain is Madrid. …" — coherent and consistent with the fixture
  argmax continuation. Load 10.8s; decode 0.33 tok/s (debug CPU reference,
  no KV cache — expected-slow by design). Edge cases: empty prompt → usage
  error exit 2; nonexistent dir → clear error exit 1; 5001-token prompt →
  "context limit is 4096" error exit 1, no crash. Phase 1 CLI context cap =
  min(4096, max_position_embeddings), enforced in DecodeLoop.
- **Family note:** nothing new beyond P1-4 — the decode layer is
  family-agnostic; Qwen3-specific logic stays in the module stack.
- **Phase 1 exit criteria walked:** CLI coherent text ✓; logit suite <=1e-3
  all 5 prompts ✓; per-module activation tests ✓ (P1-4, re-green);
  enumerated edge-case tests ✓ (parser/config from P1-2, decode 8-9 + CLI
  edge inputs + tokenizer equivalence 11 this task; temperature case 10
  n/a — not built); DECISIONS.md updated ✓ (this entry). AUDIT-1 and
  SPEC-P2 flipped to ready.

## 2026-08-23 — AUDIT-1: full-depth code audit run — 5 verified findings, 3 tasks seeded

- **Method:** /code-audit at full depth, MLE lens auto-enabled (transformers/
  mlx oracle toolchain). Architecture map (read-only explorer) fed to 4
  parallel specialist reviewers (swift, silent-failure, mle, python), then
  cross-reviewer dedup and adversarial verification: independent skeptics
  prompted to refute, default-refuted-if-uncertain, 3 votes on
  CRITICAL/HIGH and 1 on MEDIUM/LOW. All reviewers landed; 22/22 agents
  completed.
- **Result:** 12 candidates → 5 confirmed (2 HIGH, 2 MEDIUM, 1 LOW), 7
  refuted. Full report incl. refuted-candidates appendix: docs/AUDIT.md
  (fresh snapshot; overwritten on each audit rerun by design).
- **Confirmed, in brief:** (F1, HIGH) ModelConfig.positiveInt boundary bug —
  `asDouble <= Double(Int.max)` admits 2^63 since Double(Int.max) rounds UP
  to 2^63; NSNumber.intValue wraps to Int.min; reproduced end-to-end traps
  at QwenModel.swift:53/:60 from a doctored config.json (crash instead of
  thrown ModelConfigError). (F2, HIGH) decode stop set is {151645} only;
  the pinned checkpoint's generation_config.json lists [151645, 151643] and
  is read nowhere — HF generate() consults it, so this is oracle-parity
  skew the teacher-forced logit suite structurally cannot see. (F3, MED)
  double() lacks finiteness/sign checks for rms_norm_eps/rope_theta →
  silent NaN logits. (F4, MED) intList lacks positiveInt's upper bound →
  garbage eos ids pass. (F5, LOW) tokenizer.json/tokenizer_config.json
  hashes exist only as prose in this ledger, never checked programmatically.
- **Decision:** seeded CFG-1 (F1+F3+F4, one validation-layer diff), EOS-1
  (F2), TOK-1 (F5) into docs/PRIORITIES.yaml at ranks 26-28 — after the
  phase chain per the audit SOP (rank after current max); James may re-rank
  (e.g. EOS-1 before Phase 2 decode work) as with IO-1's bump. No gates,
  fixtures, or pinned invariants touched by the audit; no code changed.
- **Note on refuted candidates:** 7 plausible-but-wrong findings are
  recorded in docs/AUDIT.md's appendix so audit reruns don't resurface
  them without new evidence (notably: fixture-write atomicity is covered
  by manifest sha256 verification; the 4-of-50 full-vocab checkpoint
  structure is the pre-committed gate design, not a coverage gap).

## 2026-08-23 — Audit tasks re-ranked ahead of Phase 2 (decided by James)

- CFG-1/EOS-1/TOK-1 bumped 26/27/28 -> 13.1/13.2/13.3: all three audit
  fixes land before any Phase 2 decode work. Rationale: CFG-1 and EOS-1 are
  HIGH-severity correctness/robustness gaps in the exact load/decode paths
  Phase 2 builds on; EOS-1 in particular affects free-running stop behavior
  that SPEC-P2's top-1-agreement gate will measure, and TOK-1 pins the
  tokenizer lineage the Phase 2 diffs depend on. Fractional ranks after
  AUDIT-1 (13), IO-1 precedent — phase chain unrenumbered; SPEC-P2 stays
  ready at rank 14 and simply picks up after the three fixes.

## 2026-08-23 — CFG-1: ModelConfig numeric validation hardened (audit F1+F3+F4)

- **What landed:** one diff in the config validation layer
  (Sources/QwenMetalEngine/IO/ModelConfig.swift) + 6 new edge-case tests
  (ModelConfigTests, red-first). No public API signature changed; the init
  strictly narrows what it accepts — configs that previously crashed
  (SIGTRAP in QwenModel.init) or silently produced NaN logits now throw
  ModelConfigError.invalidValue naming the key.
- **Validation semantics chosen:** integer fields validate via
  `Int(exactly: NSNumber)` — no Double round-trip. This closes the F1
  boundary (2^63 no longer admitted; NSNumber.intValue wrap to Int.min is
  unreachable) while still accepting Int.max itself, which the audit's
  alternative (strict `< Double(2^63)` compare) would wrongly reject
  because Double cannot distinguish Int.max from 2^63 — a test pins this
  (testPositiveIntStillAcceptsIntMax). intList (eos_token_id) uses the
  same Int(exactly:) bound (F4). rms_norm_eps/rope_theta go through a new
  `positiveFiniteDouble` (`isFinite && > 0`), so negative/zero/inf values
  throw instead of NaN-ing RMSNorm/RoPE (F3).
- **Observation:** literal `1e999` never reaches the finiteness guard on
  this platform — JSONSerialization rejects it as malformed JSON. The test
  (testDoubleFieldsRejectOverflowingLiterals) asserts only "throws a
  ModelConfigError", so the invariant (no non-finite value passes) holds
  under either parser behavior.
- **No gates touched:** these are exact throw-behavior tests, not numeric
  tolerances; no fixtures or pinned invariants involved.
- **Verification:** ModelConfigTests 20/20 green (9 red before the fix,
  matching the audit's claims exactly); full suite minus the ~32-min logit
  phase-exit gate (untouched layer): 119 tests, 0 failures, 21s.

## 2026-08-23 — EOS-1: generation_config.json eos ids join the decode stop set (audit F2)

- **Context (audit F2, HIGH, upheld 3-0):** the pinned checkpoint's
  generation_config.json — the file HF generate() itself consults for
  stopping — lists eos_token_id [151645, 151643], but the engine stop set
  was built from config.json ∪ tokenizer only = {151645}. <|endoftext|>
  (151643) never stopped decode: silent post-EOS garbage on
  completion-style prompts, invisible to the teacher-forced logit suite
  (it structurally cannot see free-running stop behavior) and hidden in
  CLI output by skipSpecialTokens.
- **Decision: fix, not scope-out** — unioning the file's ids restores
  oracle parity with HF generate(). Landed as:
  ModelDirectory.generationConfigURL (optional file — absence is not an
  error); new GenerationConfig (IO module) parsing exactly the one field
  decode consumes, eos_token_id, through the same Int(exactly:)-validated
  intList as config.json (CFG-1 bounds), present-but-malformed failing
  loudly via ModelConfigError, sampling keys ignored (greedy is a
  protocol pin); CLI stop set = config.json ∪ tokenizer ∪
  generation_config.json.
- **Stop-set observation (continues the P1-5 tokenizer entry):** effective
  stop set for the pinned checkpoint is now {151645, 151643}. Verified
  end-to-end: CLI loads models/generation_config.json and generates
  normally. Note for SPEC-P2: stop-set assembly currently lives in the
  CLI (thin, ~5 lines); Phase 2's decode redesign should decide its final
  home in the engine, since free-running stop behavior feeds the
  top-1-agreement gate design.
- **No gates touched:** throw/stop-behavior tests only; no numeric
  tolerances, fixtures, or pinned invariants involved.
- **Verification:** red first (build fails on the missing API), then +10
  tests green: DecodeLoopTests regression with the real [151645, 151643]
  pair (stops right after 151643) plus a test documenting the pre-fix
  miss; 2 ModelDirectory optional-file cases; 6 GenerationConfig parse
  edge cases. Full suite minus the logit phase-exit gate: 129 tests,
  0 failures, 21s.

## 2026-08-23 — TOK-1: tokenizer artifacts pinned programmatically (audit F5)

- **What:** TokenizerEquivalenceTests.setUpWithError now sha256-verifies the
  local-only models/ tokenizer artifacts against pinned constants, mirroring
  SharedCheckpoint's source_revision check. Present-but-drifted is a thrown
  error (loud failure), never a skip; absence still skips cleanly as before.
- **Full pins recorded** (previously prefix-only prose in the P1-5 entry;
  both recomputed this session from models/ and matching those prefixes):
  tokenizer.json
  aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4,
  tokenizer_config.json
  d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101.
- **Scope note:** models/config.json (1ddb5b89…) intentionally NOT pinned —
  the test oracle chain never reads it (SharedCheckpoint inlines the pinned
  config values; the file only feeds manual CLI runs), and audit F5 names
  only the two tokenizer files.
- **Verification:** red first — an all-zeros placeholder pin failed all 3
  tokenizer tests with the full mismatch message; real pins green (3 tests).
  Full suite minus the logit phase-exit gate: 129 tests, 0 failures, 17.5s.
  No gates, fixtures, or engine code touched.

## 2026-08-23 — Phase 2 gates pre-committed: fp16 GPU-vs-fp32-CPU tolerances + agreement gate

Set BEFORE any Phase 2 code or test exists (PLAN.md invariant 4; the eng
review's SPEC-P2 obligation, OV#5/#11). Spec: docs/phases/phase-2.md. Premise
for every derivation: GPU weights are the raw bf16 checkpoint bits (spec D1),
bit-identical to what the CPU reference upcasts — so the ONLY divergence
sources are fp16 activation rounding (unit roundoff u16 = 2^-11; activations
fp16 between kernels, fp32 accumulation inside, spec D2) and reduction order.
Notation: M = max|ref| over the compared slice; per-step M64 = max|ref top-64
value| at that step (a lower bound on the step's full-vector max-abs, i.e.
using it is conservative in the tighter direction).

- **Tier K — kernel-level, synthetic unit-scale inputs (every kernel, before
  any optimization, hard rule 3):** |Δ| <= max(2^-9·M, 2^-11). The P0B-3
  species (fp16 operands, fp32 accumulation, 4·u16 headroom over per-operand
  rounding), M-scaled per the P1-4 species; the 2^-11 floor covers near-zero
  slices.
- **Exact (==) surfaces, no tolerance:** embedding-lookup output (row copy +
  exact bf16 upcast + correctly-rounded fp16 store — GPU fp16 must equal
  fp16(ref fp32) bitwise); kv-append readback; residency mode B (wired copy)
  vs mode A (mmap) full-pipeline output (same bits, same kernels); the
  bit-shift bf16→fp32 upcast itself.
- **Tier M — isolated module tests vs the P1 activation fixtures (module fed
  its reference input slice):** norm/MLP/block-internal slices
  |Δ| <= max(2^-8·M, 2^-11) — input + output rounding + at most ~6 internal
  fp16 rounding events ⇒ 8·u16 = 2^-8. Attention (layer0_attn_output):
  |Δ| <= max(2^-7·M, 2^-11) — stated-assumption derivation: post-QK-norm
  q/k are RMS-normalized so raw score magnitudes are O(γ²), budgeted <= 2^3;
  score absolute error ≈ 2·u16·|score| <= 2^-10·2^3 = 2^-7; softmax
  sensitivity <= 2× on the simplex; the PV convex combination and o_proj keep
  the output error within ~2^-7 of the slice scale.
- **Tier E — full-stack (compounded) surfaces:** last_layer_output /
  final_norm_output slices |Δ| <= max(2^-5·M, 2^-11); teacher-forced
  full-vocab logit checkpoints (steps {0,1,24,49} × 5 prompts)
  |Δ| <= 2^-5·M_step; per-step top-64 gathered at reference indices
  |Δ| <= 2^-5·M64 (all 250 steps); per-step float64 fingerprints
  |Δ lse| and |Δ mean| <= 2^-5·M64, |Δ std| <= 2^-4·M64 (1- and 2-Lipschitz
  in the sup norm, P1-5 structure). Derivation of 2^-5: per-layer
  contribution <= 8·u16 = 2^-8 relative; ~58 sequential rounding sites
  (2 module boundaries × 28 blocks + head/tail); independent-error (RMS)
  accumulation √58 ≈ 7.6 ⇒ 2^-8·7.6 ≈ 2^-5.1, committed 2^-5. Worst-case
  LINEAR compounding (~0.4·M) was rejected as vacuous — bug-scale errors are
  O(1)·M and the two-tier design places fine resolution in the isolated
  tests, where no compounding argument is needed.
- **Top-1 agreement gate, N = 250 teacher-forced steps (OV#5/#11):** exact
  top-1 match at every step whose recorded top1-vs-top2 margin >=
  epsilon_tie = 2·(2^-5·M64) = 2^-4·M64; below that, our top-1 must be in
  {reference top-1, reference top-2}. No step is unasserted. With today's
  fixtures (measured this session: margin min 0.0048 / median 4.17 / max
  27.36; 52 of 250 steps < 1.0): 176 of 250 steps are hard-asserted, 74 sit
  in the exemption band. Teacher-forcing keeps all 250 steps comparable (the
  P1-5 premise); a flip at margin >= 2·δ cannot be rounding noise under the
  committed per-element bound δ and stays a hard failure.
- **Free-running divergence is REPORTED, not gated:** 128 free-running greedy
  steps × 5 prompts, GPU vs CPU reference; first-divergence index and both
  texts recorded in DECISIONS.md at P2-4. Rationale: with per-logit deviation
  legitimately up to 2^-5·M, any near-tie step can flip and permanently fork
  a self-fed trajectory — a numeric free-run gate is either vacuous or a
  false-failure generator. The teacher-forced gate above is the strongest
  agreement statement that composes across steps.

Honest flag (surfaced for James, veto window = before P2-EXEC starts): the
Tier K/M structure is the established derived-gate species
(P0B-3/P1-4/P1-5 precedent), but two Tier E constants are judgment-derived
rather than pure derivations — the √L independent-error compounding model
behind 2^-5, and the 2^3 attention score budget behind 2^-7. They are
recorded here BEFORE any Phase 2 test exists; once P2 tests exist, hard
rule 6 applies unmodified (failures are bug signals; these numbers never
loosen).

## 2026-08-23 — SPEC-P2: Phase 2 spec written; P2 build tasks seeded

- **Spec landed: docs/phases/phase-2.md** (naive Metal port + minimal KV
  cache, on-device). All four eng-review Part 4 obligations are covered:
  pre-committed fp16 tolerances + top-1-agreement-over-N gate (previous
  entry), mmap-vs-wired-copy sustained-stability bench (OV#9, task P2-7),
  "before" row per parity pins incl. validation-off recording (OV#12, P2-7).
- **Design decisions (D1-D8, reversible/convention-following — details and
  rationale in the spec):** GPU weights = raw bf16 checkpoint bits via ONE
  mmap-backed no-copy MTLBuffer + per-tensor byte offsets, upcast in
  registers by bit-shift (no weight-rounding term in any gate; wired-copy
  variant is the same bits, so the OV#9 comparison isolates residency);
  activations fp16 between kernels / fp32 accumulation / fp32 softmax and
  logits; KV cache one preallocated 448 MiB fp16 buffer, head-major
  [28][K|V][8][4096][128] (hard rule 4); naive one-thread-per-output
  kernels; ONE command buffer per token with dual timing + dispatch count
  (wall−GPU = the Phase 4 overhead metric, hard rule 7); sequential
  per-token prefill (batched GEMM stays Phase 5); CPU-side argmax sharing
  DecodeLoop's tie-break; stop-set assembly moves from CLI into the engine
  (closing the EOS-1 note); QwenMetalApp/ thin SwiftUI shell added as the
  planned top-level target (build by agents, deploy/run by James only).
- **Backlog:** P2-1..P2-7 seeded at ranks 14.1-14.7 (P2-7 owner: james —
  device rows); P2-EXEC now depends on them and stays the SPEC-P3 milestone.
  Phase 2's memory high-water mark (~4.0 GB bf16 + KV) is inside the
  Increased-Memory-Limit envelope and is itself the OV#9 test regime.
- **Architecture PDF staleness surfaced (upkeep rule):** docs/architecture.pdf
  was last regenerated at Phase 0 exit (v1.3) — Phase 1 exit, the audit
  fixes, and this spec are not reflected. Not silently ignored: follow-up
  DOC-1 seeded to regenerate it.

## 2026-08-23 — SPEC-P2 review refinements (decided by James)

Two refinements from James's review of the Phase 2 spec; folded into
docs/phases/phase-2.md (D8) and docs/PRIORITIES.yaml (P2-6, P2-EXEC):

- **Architecture PDF regeneration rides the phase-exit milestone.** The
  CLAUDE.md upkeep rule already mandates regeneration at each phase end, but
  Phase 1 slipped (DOC-1 is the catch-up). Enforcement placement fixed:
  P2-EXEC's close-out — and future *-EXEC milestones — explicitly include
  the PDF regen, so it cannot silently slip again.
- **QwenMetalApp tester-friction controls:** one-tap quick-load buttons for
  the two pinned benchmark prompts (bundled in-app — the P0A-1 clipboard
  lesson) and a Regenerate button re-running the last generation from the
  same prompt (the manual form of the pinned sustained regenerate-loop
  protocol). Reduces friction for the on-device runs James performs.

## 2026-08-23 — DOC-1: architecture PDF regenerated (v1.4, Phase 1 exit state)

- **docs/architecture.pdf v1.3 → v1.4** (12 pages), regenerated per the
  CLAUDE.md upkeep rule from the DECISIONS.md entries added since Phase 0
  exit. Content updates: §6 gains the Phase 1 outcome paragraph (all gates
  held first run; tokenizer id-identical at pinned 1.3.3; AUDIT-1 hardening
  — CFG-1/EOS-1/TOK-1; Phase 2 fp16 gates pre-committed, free-run
  report-not-gate); roadmap figure/table mark Phase 1 DONE and Phase 2
  next-with-spec; oracle figure's "cached ≡ uncached" sub-label corrected
  to the actual Phase 2 design (vs CPU ref @ fp16 gates; CPU-quant from
  P3); risks table closes the tokenizer-mismatch row; lineage adds
  phase-2.md, dated 2026-08-23.
- **Figure 3 memory budget updated from generic ~1.5–2B planning estimates
  to pinned-config DERIVED numbers** (PIN-1 entry: 0.97 GB packed 4-bit
  weights incl. scales, 448 MiB fp16 GQA KV @4K), explicitly labeled
  derived-not-measured with the Phase 2 bf16 ~4.0 GB high-water note;
  Phase 2/3 on-device phys_footprint rows replace them.
- **Verification:** both generator scripts ran clean; pypdf extraction
  confirms all v1.4 content markers present and no stale v1.3/planning-
  estimate strings remain (pypdf added to .venv as a verification-only
  dev dependency).

## 2026-08-23 — P2-1: GPU weight residency landed (no-copy mmap buffer + wired-copy variant)

- **Landed: Sources/QwenMetalEngine/Metal/GPUWeights.swift** + 3 internal
  raw-mapping accessors on SafetensorsFile + GPUWeightsTests (8 tests).
  Spec D1 as written: ONE `makeBuffer(bytesNoCopy:)` over the whole mmapped
  file (base page-aligned by mmap; length rounded up to a page multiple —
  safe because mmap maps whole pages), per-tensor ABSOLUTE byte offsets
  (data-section start + header `data_offsets`) handed to kernels as
  arguments, never as `setBuffer` offsets (whose alignment rules the
  format's 2-byte packing can violate). Wired-copy = same bytes memcpy'd
  into a heap `MTLBuffer` (the copy dirties the pages — that is the
  residency delta OV#9 measures).
- **Lifetime decision:** the no-copy buffer's deallocator closure retains
  the SafetensorsFile, so the munmap cannot run while ANY holder of the
  buffer is alive — not just holders of GPUWeights. (Surfaced by the
  pre-commit review; a caller keeping `buffer` alone would otherwise alias
  unmapped pages.) Test pins the GPUWeights-outlives-file-reference case.
- **All P2-1 gates were the pre-committed EXACT (==) surfaces** — bf16
  bit-shift upcast (NaN payloads included, asserted bitwise), fp16 widening
  (finite + inf patterns), wired-vs-mmap byte identity AND same-kernel
  output identity, odd data-section-offset path (spec edge case 7, via
  byte-assembled 16-bit loads in the test kernels). Held unmodified first
  run; nothing loosened, no new gate needed.
- **Measured/pinned for P2-2:** the real consolidated checkpoint's tensor
  offsets are 2-byte aligned (spot-check test asserts it, plus exact
  readback of model.norm.weight through the no-copy buffer). Production
  kernels may therefore use typed `ushort` loads; the byte-assembled form
  stays test-only.
- Test-only upcast kernels live in the test file, not the engine (P2-2 owns
  production kernels — YAGNI). Suite minus logit gate: 137 tests, 0
  failures (was 129).

## 2026-08-23 — P2-2: naive decode kernel set landed (Tier-K gates held first run)

- **Landed: Sources/QwenMetalEngine/Metal/DecodeKernels.swift** — the six
  non-attention decode kernels of spec D4 (embedding-lookup, rmsnorm, matvec
  fp16- and fp32-store, rope, swiglu, residual-add), all naive
  one-thread-per-output, runtime-compiled from one source string (P0B
  convention). + tests/QwenMetalEngineTests/DecodeKernelTests.swift (17
  tests). Suite minus logit gate: 154 tests, 0 failures (was 137).
- **API shape (reversible, anticipates D5):** kernels expose
  `encode(into encoder:)` methods rather than self-dispatching — P2-4 packs
  ~500 dispatches into ONE command buffer per token on a default (serial)
  compute encoder, and the Tier-K tests drive the same methods through
  `MetalContext.timedDispatch` (hard rule 7). Sequential-dispatch ordering
  is the serial encoder's guarantee; no barriers needed until Phase 4
  touches encoder strategy.
- **Weight addressing:** whole-checkpoint buffer + per-tensor ELEMENT
  offsets (never `setBuffer` offsets), typed `ushort` loads per the P2-1
  alignment pin; host wrappers throw on odd byte offsets. bf16→fp32 is the
  same registered bit-shift as P2-1's exact-gated form.
- **RoPE table sharing:** the GPU kernel consumes the CPU `RoPE`'s fp32
  cos/sin tables verbatim — `RoPE` gained read-only `cosValues`/`sinValues`
  accessors (no behavior change to the frozen oracle; its stored tables were
  already computed). Angle drift is therefore structurally impossible; the
  position-p unit test oracles against CPU full-recompute (spec edge case 3
  in its targeted unit form).
- **Fast-math default retained** (P0B convention, options: nil): fp32
  `exp`/`sqrt` few-ulp error is ~2^-20 relative, orders below the Tier-K
  gate's 2^-9; if a Tier-M/E suite later implicates it, `precise::`
  variants are the fix lever — the gates do not move (hard rule 6).
- **Gate outcomes:** every Tier-K diff (matvec vs BLAS.sgemm incl. odd
  shapes/near-zero floor/nonzero offset, rmsnorm single-row + per-head
  rows, rope p=0 and p=9, swiglu, residual-add, swiglu→residual one-buffer
  chain) passed at max(2^-9·M, 2^-11) unmodified first run;
  embedding-lookup passed the EXACT bitwise gate including fp16-boundary
  patterns (±0, bf16 subnormal→0, overflow→±inf, min-normal, subnormal
  result, +inf). Nothing loosened.
- lm_head needs no transposed kernel: the tied [vocab, hidden] embedding
  table IS [out, in] for logits = E·x, so the standard matvec consumes it
  directly (nothing materialized, hard rule 1).

## 2026-08-23 — P2-3: KV cache + attention kernels landed (exact + Tier-K gates held first run)

- **Landed: Sources/QwenMetalEngine/Metal/KVCache.swift +
  AttentionKernels.swift** — the preallocated decode cache (spec D3) and the
  four attention kernels of spec D4 (kv-append, attn-scores, softmax fp32,
  attn-pv), naive one-thread-per-output, encoder-based API (P2-2
  convention). + KVCacheTests (6) + AttentionKernelTests (9). Suite minus
  logit gate: 169 tests, 0 failures (was 154).
- **Cache shape as specced:** ONE fp16 buffer
  [layers][K|V][kvHeads][maxContext][headDim], head-major; allocated in
  full at init, no grow path exists in the API (hard rule 4 structurally
  enforced). Size formula overflow-checked; 448 MiB verified by test at the
  pinned dims (28/8/4096/128). Slot addressing via element offsets
  (never `setBuffer` offsets — P2-1 convention), bounds-validated on the
  host before any dispatch.
- **Context-limit stop:** append at position >= maxContext throws
  `KVCacheError.contextFull` BEFORE encoding — tested that the 
  last in-bounds append succeeds, the next throws, and the cache bytes are
  bit-identical after the refusal (spec edge case 5, no OOB possible).
- **Softmax is out-of-place** (scores -> probs), a deliberate deviation
  from the CPU module's in-place loop: under one-thread-per-element every
  thread reads its whole row, so in-place would race with concurrent
  writes. Same fp32 max-subtract/exp/normalize formula, same sequential
  reduction order per thread (rmsnorm redundant-recompute pattern). P2-4
  carries one extra [numHeads][maxContext] fp32 probs buffer (256 KB at
  real dims, negligible vs the 448 MiB cache).
- **GQA mapping** computed host-side (groupSize = numHeads/kvHeads,
  validated, `gqaMismatch` on non-divisible) and passed to kernels; the
  pattern test pins KV head h serving Q heads {2h, 2h+1} exactly (headDim=4
  makes scale=1/2 exact, so a repeat-interleave or off-by-one mapping is a
  hard value mismatch, not a tolerance question).
- **Gate outcomes:** kv-append passed the pre-committed EXACT gate
  (exhaustive whole-buffer bitwise map over three scattered slots, incl.
  NaN-payload/±inf/subnormal patterns; everything else sentinel-untouched).
  attn-scores and attn-pv passed Tier K max(2^-9·M, 2^-11) vs BLAS.sgemm
  oracles on odd shapes (hard rule 8); softmax passed Tier K vs the CPU
  formula; the p=4 append→scores→softmax→pv chain in ONE command buffer
  passed Tier K vs CPU full-recompute (spec edge case 3); p=0 decode
  reproduced single-token attention exactly (probs == 1.0, output bitwise
  == the mapped V row — spec edge case 4). Nothing loosened.

## 2026-08-24 — P2-4: GPU pipeline wired; ALL Phase 2 Tier-M/E gates held first run; free-run divergence: none

- **Landed: Sources/QwenMetalEngine/Metal/GPUModel.swift** — the P2-1/2/3
  pieces wired into a full per-token forward: ONE command buffer per token
  (21 dispatches/layer × 28 + head/tail ≈ 591, spec D5) through
  `MetalContext.timedDispatch`, so dual timing rides every step (hard rule
  7; `lastStepTiming` is the P2-5 hook). fp16 activations / fp32
  accumulation / fp32 logits (D2); RoPE kernel consumes the CPU `RoPE`'s
  fp32 tables; KV cache preallocated at init (hard rule 4); argmax stays
  CPU-side in the shared `DecodeLoop`. Loader validates every tensor's
  shape AND dtype up front (new `ModelError.badWeightDtype` — the register
  upcast is bf16-specific, a non-bf16 checkpoint must fail at load).
- **Decode-loop conformance is INCREMENTAL:** `lastPositionLogits(ids:)`
  runs only the suffix when `ids` strictly extends the cached prefix, else
  resets and replays; logits computed at the last position only (D6).
  Pinned by tests: incremental == fresh replay BITWISE; prefix-mismatch
  reset; contextFull at the preallocated bound before any dispatch.
- **Gate outcomes (all pre-committed 2026-08-23, none touched, all held
  unmodified first run):** Tier M — embeddings exact-bitwise fp16 vs
  fixture; layer0_pre_attn_norm_output ≤ max(2⁻⁸·M, 2⁻¹¹);
  layer0_attn_output ≤ max(2⁻⁷·M, 2⁻¹¹) (isolated harness fed reference
  inputs, sequential per-position over a 1-layer cache). Tier E —
  last_layer_output and final_norm_output ≤ max(2⁻⁵·M, 2⁻¹¹) through the
  full 28-layer wired stack; the teacher-forced logit suite
  (GPULogitSuiteTests, 5 prompts × 50 steps): full-vocab checkpoints
  ≤ 2⁻⁵·M_step, per-step float64 fingerprints (lse/mean ≤ 2⁻⁵·M64, std ≤
  2⁻⁴·M64), top-64 ≤ 2⁻⁵·M64, tie-aware top-1 at ε_tie = 2⁻⁴·M64 — all
  250 steps asserted, zero failures. GPU suite wall time 98 s debug (vs
  ~32 min for the CPU suite — the KV cache at work).
- **Free-running divergence REPORT (committed protocol: 128 greedy steps ×
  5 prompts, GPU vs CPU reference, no stop set, release build):** first
  divergence = NONE on all five prompts — the GPU trajectory is
  token-identical to the CPU reference for all 640 free-running steps, so
  "both texts" collapse to one identical text per prompt (e.g.
  short_english continues " Paris. The capital of Italy is Rome. …").
  Reproduce: `QWEN_FREE_RUN_REPORT=1 swift test -c release --filter
  FreeRunReport` (opt-in harness, tests/…/FreeRunReportTests.swift;
  18.8 min, CPU side dominates). The report stays a report: nothing about
  this result gates future runs (the 2026-08-23 rationale stands).
- **Stop set moved into the engine (spec D7, closes the EOS-1 note):**
  `ModelDirectory.stopTokenIds(config:tokenizerEOSTokenId:)` = config.json
  ∪ tokenizer ∪ generation_config.json; CLI and (Phase 2) app consume the
  one implementation. Tests pin the pinned-directory result {151645,
  151643}, the no-generation-config and nil-tokenizer unions, and loud
  failure on a malformed generation_config.json. EOS-1's DecodeLoop
  regressions still pass, and the stop semantics are re-pinned against the
  real GPU backend (scripted-free: whatever token greedy emits first,
  adding it to the stop set stops decode right after it).
- **CLI `--backend gpu`** (default cpu, behavior unchanged): loads
  GPUModel at the 4096 pinned context (448 MiB cache). Verified: coherent
  text (" Paris. The capital of Italy is Rome. …", 2.33 tok/s M2 Pro
  debug — naive by design, P2-5 measures properly); empty prompt and
  >4K prompt produce the same errors as the CPU backend; no-Metal machines
  get `MetalHarnessError.noDevice`, and all GPU test classes skip cleanly
  (spec edge cases 8–10).
- **Suite: 196 tests, 0 failures** (full suite minus the CPU logit gate,
  +27 new; 1 skip = the env-gated free-run harness). GPU tests also skip
  cleanly when the local checkpoint is absent (SharedCheckpoint pattern;
  new SharedGPUModel shares one mmap-residency pipeline across suites).
- **Follow-up seeded: DK-1** — pre-existing generic `setBytes` warning in
  DecodeKernels/AttentionKernels surfaced by the release build.

## 2026-08-25 — P2-5: per-token instrumentation landed + first Mac GPU sanity row

- **Landed:** `DispatchCounter` (increments at the exact `dispatchThreads`
  call sites inside DecodeKernels/AttentionKernels — the count is MEASURED,
  never derived from pipeline structure, so Phase 4 fusion changes the
  reported number automatically); `GPUModel.lastStepDispatchCount` next to
  the existing `lastStepTiming`; `Decode/DecodeInstrumentation.swift`
  (`TokenStepRecord`, `DecodeTimingCollector`, `CanonicalDecodeWindow`) —
  engine-side aggregation so CLI and the P2-6 app report identical numbers;
  CLI `--backend gpu` now prints the per-token block + decode rates
  (cpu-backend output byte-unchanged).
- **Canonical-window semantics (pinned in code + tests):** window rate =
  384 tokens ÷ (wallEnd of the forward producing generated token 512 −
  wallEnd of the forward producing token 128), tokens 1-based — the natural
  reading of PLAN.md's "generated tokens ÷ decode wall time, canonical
  window = tokens 128–512". Completion-to-completion spans include host
  work between command buffers (argmax, loop) — the honest cadence.
  Below 512 generated tokens the window is reported n/a, never
  extrapolated. Overhead metric = median of PER-TOKEN wall−GPU deltas
  (not medianWall − medianGPU; a test pins the distinction).
- **Dispatch count verified:** 591/token with logits (21×28 + embedding +
  final norm + lm_head), 589 without the tail; tiny 1-layer synthetic
  model measures 24/22 — exact-value tests would break on any missed or
  double-counted dispatch site.
- **MEASURED (Mac dev-loop sanity row, PROVISIONAL — benchmarks/results.md
  Phase 2 section):** M2 Pro, release, decode-essay (84 pinned tokens),
  640-token burst: median GPU 218.44 ms/token, median wall 218.83 ms,
  median wall−GPU **0.391 ms** (the Phase 4 overhead metric — negligible
  on Mac at 591 dispatches/token), canonical window **4.56 tok/s**,
  overall 4.56. ~9% of the Mac naive roofline (178.19 GB/s ÷ 3.44
  GB/token) — expected for one-thread-per-output matvec; optimization
  stays Phase 3–5 (hard rule 3 discipline held: no kernel touched).
  Repeatability: two same-session runs agree to 3 digits.
- **Measurement footgun recorded:** shell `$(cat file)` strips the rendered
  prompt's trailing `\n\n` → 83 tokens, not the pinned 84. The results.md
  note carries the workaround; follow-up CLI-1 seeded (a `--prompt-file`
  flag that feeds exact bytes) so P2-7/Phase 6 Mac-side reproduction can't
  drift.
- **Suite: 212 tests, 0 failures, 1 skipped** (env-gated free-run harness;
  full suite minus the CPU logit gate), +16 over P2-4 (13 instrumentation
  arithmetic, 3 GPUModel dispatch/timing sanity). No numeric gates added
  or touched — instrumentation tests are structural, and the Phase 2 gate
  set is untouched.

## 2026-08-25 — P2-6: QwenMetalApp thin iOS shell landed (build-verified; device runs are James's)

- **Landed:** `QwenMetalApp/` (the committed top-level target planned in
  CLAUDE.md, spec D8) + an engine-side `Bench/` module so the app stays
  thin: `BenchGenerationRunner` (instrumented single generation — the P2-5
  collector wiring factored engine-side, + stop-reason inference and a
  prefill-time field), `SustainedLoop` (the pinned ≥5-min regenerate
  protocol; per-generation tok/s sequence kept — the OV#9 bimodality
  signal; refuses to spin on empty generations), `BenchmarkReport`
  (row-field export: PROVISIONAL marker, dual-timing medians + wall−GPU
  overhead, dispatches/token, canonical-window labeling, operator
  placeholders), `MemoryFootprint` (task_info phys_footprint — explicitly
  a cross-check; the Xcode gauge stays the metric of record per the
  protocol pin), and `BenchDefaults` (burst cap 640 = P2-5 Mac row
  precedent; sustained minimum 300 s = PLAN.md pin; both pinned by test).
- **DecodeLoop API extension (reversible):** `generate` gained an optional
  `shouldStop` closure polled BEFORE each forward — token-boundary
  cooperative stop for the app's Stop control and the sustained loop's
  duration bound. Default nil; all existing call sites and behavior
  unchanged (existing suites re-ran green).
- **Model-transfer decision (reversible, convention):** the app finds the
  model in its Documents folder (Finder file sharing; UIFileSharingEnabled)
  — any subfolder that validates via the engine's `ModelDirectory`; a
  missing model produces a clear error listing the expected files. No
  bundled checkpoint (3.44 GB would bloat every install).
- **App structure:** hand-authored `QwenMetalApp.xcodeproj` (Xcode 16
  synchronized-folder format, local package dep on the repo root, shared
  scheme with Run = Release per the benchmark protocol), explicit
  Info.plist, Increased Memory Limit entitlement wired via
  CODE_SIGN_ENTITLEMENTS (applies when James signs). SUPPORTED_PLATFORMS
  = iphoneos only (simulator support is a non-goal). Screens per D8:
  Generate (quick-load pinned prompts, Regenerate, Stop) and Benchmark
  (burst with prompt picker — prefill-summarize serves the prefill row;
  sustained pinned to decode-essay; residency mmap/wired toggle that
  drops + reloads the model, since residency is baked in at load per D1;
  share/copy row export).
- **Pinned prompts ride in the bundle as copies** of the rendered forms;
  a new drift test (AppBundledPromptTests) pins them byte-identical to
  benchmarks/prompts/rendered/ incl. decode-essay's trailing "\n\n"
  (the CLI-1 lesson) — a drifted copy fails the suite loudly.
- **Honest limitations recorded:** the export's prefill figure spans
  generation start → first generated token's completion (sequential
  prefill + one decode forward — labeled as such in the export text);
  Stop during a sustained loop aborts without a report (status line says
  so); battery health and cold/warm are operator-entered fields, never
  guessed.
- **Verified:** `xcodebuild -scheme QwenMetalApp -destination
  'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build` → BUILD SUCCEEDED
  (Release, iPhoneOS 26.5 SDK, Xcode 17F113 — the release Xcode); both
  prompts confirmed inside the built .app. Engine suite minus the CPU
  logit gate: 232 tests, 0 failures, 1 skipped (+20 over P2-5: 14
  harness, 4 report export, 2 bundled-prompt drift). No numeric gates
  added or touched. Seeded CLI-2 (dedupe the CLI's inline P2-5 wiring
  onto BenchGenerationRunner). P2-7 (James, on-device) is now ready.

## 2026-08-25 — P2-7 MEASURED: on-device "before" rows; attached-run pitfall; residency decision (James)

Full rows in benchmarks/results.md (Phase 2 iPhone section). Two sessions
were run; the first was invalidated and rerun — both are recorded.

- **Protocol incident (recorded, method-level lesson):** session 1 was
  launched via Xcode's Run button — Metal API validation ON + debugger
  attached — which inflated per-token GPU time **1.4–1.9×** on this
  591-dispatch/token workload (median GPU 217 ms attached vs 112–159 ms
  detached at identical settings). Rows kept, marked ATTACHED/non-
  comparative per the P0A-1 validation-off pin; the full detached rerun
  (session 2) supplies the valid rows. The P0A-1 finding that validation
  cost is strongly engine-dependent (MLX ~1%, llama.cpp 17–21%) now has
  our datapoint: many tiny dispatches ⇒ large penalty. Detached launches
  are mandatory for every future device row (already in the runbook;
  reaffirmed the hard way).
- **Decode "before" (iPhone 15 Pro, protocol row: warm burst, canonical
  window, detached): 6.74–6.92 tok/s (mmap)** — 23% of the committed 29.4
  target, 17.6% of MLX's 39.2. Fast-state runs in the same session
  reached 7.81–8.64; honest headline is the **range 6.7–8.6 tok/s**.
- **Run-to-run device-state variance ~1.4× (detached, same settings):**
  median GPU clustered at ~112 vs ~158 ms/token; cold/warm is not the
  driver (the cold burst was fastest). Cause unidentified (device
  power/thermal governor state). CONSEQUENCE for Phase 3+: comparative
  rows use repeats/interleaving and report ranges — folded into SPEC-P3's
  obligations (PRIORITIES note added).
- **Dispatch overhead (the Phase 4 metric): 1.9–2.0 ms/token at 591
  dispatches ≈ 3.4 µs/dispatch — stable across ALL 13 runs, both
  sessions, both residencies** (Mac: 0.39 ms). The only number the state
  variance never touched. ~6% of a Phase 3-scale 33 ms token.
- **Prefill "before": 8.2–10.7 tok/s sequential** (852-token prompt:
  103.5 s to first token) — Phase 5's target number.
- **Residency (OV#9): speed comparison UNRESOLVED — and that is the
  result.** Attached session: mmap 42% slower sustained. Detached rerun:
  mmap FASTER (window 8.64 vs 7.46). Both directions observed; deltas
  inside the state-noise band. What DID measure cleanly: phys_footprint
  (Xcode gauge) **mmap ~536 MB vs wiredCopy ~4.3 GB** (the llama.cpp
  307 MB accounting asymmetry quantified on our side, per invariant 3;
  in-app task_info cross-check within ~2% of the gauge) and load time
  **1.5 s vs 9.7 s**.
- **DECISION (James, 2026-08-25): default residency for Phase 3+ stays
  `mmap`.** Rationale: no consistent speed penalty established, and mmap
  is strictly better on footprint (536 MB vs 4.3 GB) and load (1.5 s vs
  9.7 s). wiredCopy remains a toggle; Phase 3 re-runs the comparison on
  the ~0.97 GB packed weights with an interleaved-repeats protocol before
  treating the question as closed.
- **Determinism across modes/sessions (free evidence):** prefill-summarize
  produced exactly 462 tokens then EOS in both sessions; the decode-essay
  greedy trajectory EOSes at generated token 1601 (observed in the mmap
  sustained run; all shorter runs — 640/930/1211/1595 — consistently
  EOS-free). The sustained loop's regenerate-on-EOS path executed
  correctly in the field.
- **Roofline reframe (plan-level):** fast-state decode ⇒ ~30.8 GB/s
  weight traffic ≈ 70% of the measured 43.84 GB/s roofline — on-device
  the naive engine is close to memory-bound, unlike the Mac sanity row
  (M2 Pro ~9% of its roofline; its "iPhone == Mac per-token" coincidence
  was a validation artifact). Phase 3 packed roofline = 45.2 tok/s; at
  the observed 50–70% efficiency ⇒ ~22–32 tok/s, bracketing the 29.4
  target — Phases 4–5 remain load-bearing.
- **Annotations:** battery field in exports recorded SoC start→end;
  Battery Health shows "Normal" (no %) on iOS 26.5.2 — ≈85% last measured
  at P0A-1. Thermal: phone notably cooler than the Phase 0 MLX/llama.cpp
  sustained cycles at today's speeds; expect that to change as Phases 3–5
  approach the roofline.

## 2026-08-25 — P2-EXEC: Phase 2 exit criteria walked — Phase 2 EXITED

- **Exit-criteria walk (PLAN.md phase table / phase-2.md §Exit criteria —
  all five MET, evidence cited to the entries above):**
  1. *Preallocated K/V, append per step, naive unfused attention;
     incremental decode from the first on-device build* ✓ — P2-3/P2-4:
     one 448 MiB fp16 buffer allocated at model load with no grow path in
     the API (hard rule 4 structurally enforced; size verified by test at
     the pinned dims); per-step kv-append; incremental prefix decode
     pinned BITWISE-equal to a fresh replay.
  2. *Pre-committed fp16 gate passes vs CPU reference* ✓ — every Tier
     K/M/E gate (committed 2026-08-23, before any Phase 2 code or test
     existed) held UNMODIFIED on the first run (P2-1..P2-4 entries),
     including the exact (==) surfaces and the tie-aware top-1 agreement
     gate over all 250 teacher-forced steps. The free-running divergence
     report (report, not gate, per the committed rationale): NONE — GPU
     token-identical to the CPU reference on all 5 prompts × 128 steps.
  3. *"Before" benchmark row recorded per parity pins* ✓ — P2-7 detached
     session: protocol headline warm-burst canonical window
     6.74–6.92 tok/s (mmap), honest range 6.7–8.6 tok/s; prefill
     8.2–10.7 tok/s; greedy, pinned prompts, per-engine token counts,
     phys_footprint, validation setting recorded (the attached session
     was invalidated and fully rerun — the pin worked); PROVISIONAL
     markers on every row.
  4. *mmap vs wired-copy sustained-stability comparison recorded; default
     residency decided in DECISIONS.md* ✓ — comparison recorded with the
     speed question honestly UNRESOLVED (both directions observed inside
     the ~1.4× device-state noise band); footprint and load time measured
     cleanly (mmap 536 MB / 1.5 s vs wired 4.3 GB / 9.7 s); DECISION:
     mmap default for Phase 3+ (James, P2-7 entry), re-test interleaved
     on packed weights before closing the question.
  5. *DECISIONS.md entries for everything decided/measured, incl. the
     free-run report* ✓ — the 2026-08-23/24/25 P2-1..P2-7 entries above,
     plus the attached-run protocol lesson.
- **Judgment-derived Tier-E constants:** the honest-flag veto window
  ("before P2-EXEC starts", gates entry 2026-08-23) closed UNEXERCISED —
  James raised no veto, and both flagged constants (√L compounding model
  behind 2⁻⁵, the 2³ attention-score budget behind 2⁻⁷) were never needed
  as slack: all gates held first run. Hard rule 6 continues to bind them.
- **Verification at exit:** full suite minus the CPU logit gate —
  232 tests, 0 failures, 1 skipped (env-gated free-run harness), 127 s —
  identical counts to the P2-6 baseline. Backlog drift test green.
- **Architecture PDF regenerated v1.4 → v1.5** per the upkeep rule
  (folded into this milestone by the 2026-08-23 SPEC-P2 review
  refinement): §1 gains the Phase 2 standing (6.7–8.6 tok/s before-row),
  §4 the measured phys_footprint asymmetry + residency decision, §5.2 the
  Phase 2 roofline position (50–70%, dispatch overhead 1.9–2.0 ms/token @
  591, Phase 3 projection ~22–32 tok/s bracketing 29.4), §6 the Phase 2
  oracle outcome (all gates first run; free-run divergence none), §8
  roadmap P2 EXITED / P3 NEXT, §10 risk rows (memory measured; ~1.4×
  device-state variance → SPEC-P3 obligation). Figure 3 carries the
  measured footprints; Figure 5 gains the measured P2 point
  (3.44 GB/token, 6.7–8.6 tok/s); Figure 7 marks P2 done. Verified via
  pypdf extraction: all v1.5 markers present, no stale v1.4 strings.
- **Phase 2 is EXITED.** SPEC-P3 flipped to ready (rank 16) — the next
  action; its notes already carry the P2-7 obligations (repeats/
  interleaving protocol for device rows; packed-weights residency
  re-test). No new follow-ups this session; existing fillers (BW-1,
  DEV-1, DK-1, CLI-1, CLI-2) stand.

## 2026-08-25 — App signing moved to gitignored Local.xcconfig; scheme validation semantics pinned (decided by James)

Housekeeping of the P2-7 device-session residue in QwenMetalApp/, decided
by James with reproducibility as the criterion (a clone should build and
run the diagnostics with no foreign signing state baked in):

- **Signing home:** DEVELOPMENT_TEAM removed from the committed pbxproj.
  Both app build configs now use `baseConfigurationReference` →
  QwenMetalApp/Base.xcconfig (committed; contains only an optional
  `#include? "Local.xcconfig"` + instructions), and the per-developer
  QwenMetalApp/Local.xcconfig (gitignored) carries the team id. Verified:
  `-showBuildSettings` resolves DEVELOPMENT_TEAM through the chain, and
  the clone-equivalent unsigned build (`CODE_SIGNING_ALLOWED=NO`, no
  Local.xcconfig consulted) still returns BUILD SUCCEEDED.
- **.gitignore fix:** the old `*.xcodeproj/xcuserdata/` pattern was
  root-anchored (contains a slash) and never matched the app project one
  level deep — replaced with `**/xcuserdata/`; `QwenMetalApp/
  Local.xcconfig` added. Both P2-7 xcuserdata dirs now ignored.
- **Scheme attribute semantics PINNED (trap recorded so it is not
  "fixed" backwards):** in .xcscheme files, `enableGPUValidationMode`
  ABSENT = Metal API validation ON (Xcode default for attached runs);
  `enableGPUValidationMode = "1"` = validation DISABLED — it is what
  Xcode writes when the Diagnostics checkbox is UNCHECKED (confirmed
  against James's scheme UI showing unchecked). The P2-7-modified shared
  scheme (which carries ="1") is therefore committed deliberately: a
  fresh clone's attached Run now defaults to validation OFF, matching
  the P0A-1 validation-off protocol pin, where the previously committed
  scheme (no attribute) silently defaulted it ON — the exact session-1
  trap. Xcode's rewrite also dropped the hand-authored Run=Release
  comment (rewrites always strip comments; the Release setting itself
  survived and stays pinned by the scheme).

## 2026-08-25 — Phase 3 gates pre-committed: layered quant oracle + microbench fraction + quality band

Set BEFORE any Phase 3 code or test exists (PLAN.md invariant 4; the eng
review's SPEC-P3 obligations OV#4/#11/#12). Spec: docs/phases/phase-3.md.
Premise for every tier reuse: the packer chooses codes against the
fp16-rounded scale/bias as stored (spec D2), and q·scale is exactly
representable in fp32 (≤4-bit int × 11-bit fp16 significand ⇒ ≤15
significand bits), so `q*scale + bias` is ONE correctly rounded fp32
operation — CPU and GPU dequant produce bit-identical fp32 values
regardless of fma contraction. Both sides of every diff therefore consume
identical weight values, exactly the Phase 2 situation (bit-identical bf16
upcast), and the Phase 2 divergence analysis (fp16 activation rounding +
reduction order only) transfers unchanged.

- **Layer 1 — dequant tile: EXACT (==), no tolerance.** GPU fp32 tile dump
  vs CPU dequant of the same packed bytes, bitwise, including degenerate
  (scale = 0), extreme-scale (fp16 max/min-normal, subnormal), and
  negative-heavy groups. Embedding-gather fp16 store == fp16(exact fp32
  dequant) bitwise (the Phase 2 embedding species).
- **Layer 2 — fused dequant-matvec: Phase 2 Tier K reused unchanged,**
  |Δ| ≤ max(2⁻⁹·M, 2⁻¹¹) vs the CPU-quant oracle (BLAS.sgemm over the
  identical dequantized fp32 weights, hard rule 8). No new constant: same
  divergence species as the Phase 2 matvec (fp32 accumulation both sides,
  fp16 store). Applies to every optimization iteration of the kernel
  (hard rule 3: correctness test first, re-passed after each change).
- **Full-model tiers: Phase 2 Tier M and Tier E constants reused verbatim**
  with the oracle swapped to the CPU-quant reference, computed live at the
  same slice points and suite structure (isolated modules at
  max(2⁻⁸·M, 2⁻¹¹) / attention max(2⁻⁷·M, 2⁻¹¹); full-stack slices
  max(2⁻⁵·M, 2⁻¹¹); teacher-forced 5×50-step logit suite at 2⁻⁵ bounds
  with fingerprints at 2⁻⁵/2⁻⁴·M64; tie-aware top-1 at ε_tie = 2⁻⁴·M64
  with margins from the CPU-quant logits). Free-running divergence
  (128 steps × 5 prompts, GPU-quant vs CPU-quant) stays REPORTED, not
  gated — the 2026-08-23 rationale stands.
- **Quality gate vs mlx-lm 4-bit (OV#12 named metrics + slice; grades the
  packing recipe via the CPU-quant reference).** Metrics, all teacher-forced
  on the reference fp32 argmax sequences, fp32 logits, float64 statistics:
  (1) top-1 agreement vs the fp32 argmax over the 250 fixture steps;
  (2) mean full-vocab KL(P_fp32 ‖ P_engine), nats, over the 250 steps;
  (3) perplexity on the named slice — WikiText-2 `wikitext-2-raw-v1` TEST
  split at a dataset revision pinned in tools/pins.py when P3-3 lands,
  standard document concatenation, first 4096 tokens under the pinned
  tokenizer, ppl = exp(mean NLL) over positions 1..4095. Band-setter
  numbers (mlx's own agreement A_mlx, KL_mlx, Δppl_mlx vs the same fp32
  reference) are measured by the extended tools/ dump and recorded here
  BEFORE any metric of ours is computed. **Gates:** A_ours ≥ A_mlx − 4
  percentage points; KL_ours ≤ 1.5 × KL_mlx; Δppl_ours ≤ 1.5 × Δppl_mlx
  + 0.01 (Δppl = ppl_engine − ppl_fp32; the +0.01 floor guards a
  near-zero mlx delta). Out-of-band ⇒ the packer is the suspect and oracle
  layer 1 arbitrates (Issue 2 diagnosis rule). The 2026-08-22 argmax-level
  mlx dump is NOT reused for the band — it predates these definitions.
- **Microbench bandwidth fraction (OV#11):** the standalone fused
  dequant-matvec microbench (one token's worth of real packed matvecs,
  28×7 + lm_head = 197 dispatches, weights-only) must achieve aggregate
  weight-stream rate ≥ **0.70 × 43.84 GB/s = 30.7 GB/s on the pinned
  iPhone** (best across the pinned repeats protocol, GPU-timestamp basis,
  wall recorded alongside per hard rule 7). Aggregate = total packed bytes
  (q + scales + biases ≈ 0.967 GB) ÷ summed kernel GPU time; per-shape
  rates reported but not gated. Mac rows are dev-loop sanity, PROVISIONAL,
  never gated. Grounding (measurements, not aspiration): the naive
  full-model Phase 2 engine already ran at ~50–70% of roofline on-device
  (P2-7), and MLX/llama.cpp full decode sit at ~96–102% of the triad
  figure (P0A-1) — a weights-only streaming kernel clearing 70% is a real
  bar below the ecosystem ceiling. BW-1's caveat (triad likely understates
  read-mostly bandwidth ~5–10%) makes the fraction conservative in the
  strict direction.
- **Device-row protocol addendum (from P2-7's measured ~1.4× device-state
  variance; extends the benchmark protocol for ALL Phase 3+ rows):**
  single-config rows = ≥3 same-session repeats, report median AND range;
  A-vs-B comparisons = interleaved A,B,A,B,A,B (≥3 per side), a
  directional claim requires non-overlapping ranges, else recorded
  "unresolved at n=3"; detached launches mandatory (validation OFF,
  recorded). P3-7 re-runs mmap-vs-wired on the packed weights under this
  protocol and closes the residency question with its own entry.

Honest flag (surfaced for James, veto window = before P3-EXEC work starts,
the SPEC-P2 precedent): the tier reuses and the exactness arguments are
derivations, but FOUR constants in this entry are judgment-derived — the
0.70 microbench fraction, the 4-percentage-point agreement margin, the
1.5× KL/Δppl multipliers, and the choice of the WikiText-2 first-4096
slice. Also pinned by this spec and flaggable in the same window: the
packed-layout schema itself (D1 — u32 nibble order, group 64, fp16
scale/bias triplet naming) becomes a pinned invariant once P3-1 lands.
Per hard rule 6, once P3 tests exist these numbers never loosen; failures
are bug signals.

## 2026-08-25 — SPEC-P3: Phase 3 spec written; P3 build tasks seeded

- **Spec landed: docs/phases/phase-3.md** (4-bit quant + fused
  dequant-matvec). All four eng-review Part 4 obligations covered:
  microbench bandwidth fraction pre-committed (OV#11, previous entry),
  perplexity eval slice named (OV#12, WikiText-2 test first-4096),
  mlx-community 4-bit provenance handled (OV#12 — verified at PIN-1 via HF
  file history; the quality-gate dump re-verifies revision 3b1b1768
  programmatically at run time), embeddings + lm_head quantized per the
  MLX recipe (OV#4 — the tied embedding is stored once as a packed
  triplet; 1-D norm vectors stay bf16, matching the MLX recipe's
  linears+embeddings coverage).
- **P2-7 obligations folded in:** the repeats/interleaving device-row
  protocol is pinned (spec D8 + gates entry), and P3-7 re-runs the
  mmap-vs-wired comparison interleaved on the ~0.97 GB packed weights,
  closing the residency question P2-7 left unresolved.
- **Design decisions (D1–D8, rationale in the spec):** packed layout =
  4-bit grouped affine, group 64 along the reduction dim, per-group fp16
  scale+bias, `{name}.q` u32 (8 codes per word, low nibble first) +
  `.scales`/`.biases` fp16, norms pass through bf16, one safetensors-format
  file with 4-byte-aligned `.q` offsets validated at pack and load;
  packer is Swift engine code (`qwen-metal-cli pack`) so parser/upcast/
  dequant arithmetic are shared with the loaders — tools/ Python stays
  oracle-side; CPU-quant reference materializes fp32 dequant weights
  through the FROZEN CPU module path (explicitly the PLAN invariant 4
  oracle carve-out — hard rule 1 continues to bind engine GPU + app
  unqualified); fused dequant-matvec may be optimized within the phase
  (after its correctness tests pass, re-passed per iteration);
  quantization error vs kernel bugs separated by the layered oracle
  (Issue 2); attention/KV untouched this phase.
- **Backlog:** P3-1..P3-7 seeded at ranks 16.1–16.7 (P3-7 owner: james —
  device rows); P3-EXEC re-pointed at them. Memory high-water drops to
  ~1.5 GB (packed 0.97 GB + KV 448 MiB) — the packed roofline is
  ~45.2 tok/s and P2-7's efficiency range projects ~22–32 tok/s,
  bracketing the 29.4 target.
- **NOTE for James (veto window before P3-EXEC):** four judgment-derived
  gate constants + the packed-layout schema pin are flagged in the gates
  entry above.

## 2026-08-26 — Phase 3 veto window CLOSED: gates + schema approved (decided by James)

James reviewed the flagged items from the 2026-08-25 gates entry and
approved them explicitly (window closed early by decision, not by the
passive SPEC-P2-style expiry):

- **All four judgment-derived constants stand as committed:** the 0.70
  microbench bandwidth fraction, the 4-percentage-point top-1 agreement
  margin, the 1.5× KL and 1.5×+0.01 Δppl multipliers, and the WikiText-2
  test first-4096 perplexity slice. Discussed and resolved: the quality
  band is a bug tripwire (floor), not an aspiration ceiling — measured
  metrics are reported side-by-side regardless, and parity-or-better vs
  mlx-4bit remains the expected outcome of matching the MLX recipe.
  A tighter parity-as-gate form was considered and rejected (one-way
  gates + legitimate rounding-choice wobble = false-failure risk).
- **Packed-layout schema (spec D1) approved as pinned:** q4g64,
  `{name}.q` u32 with 8 codes/word low-nibble-first, separate fp16
  `.scales`/`.biases`, norms pass through bf16, fp16-rounded scale/bias
  before code selection.
- **Committed success metric unchanged:** decode ≥ 29.4 tok/s
  (0.75 × MLX measured) stays the bar; raising it was considered and
  declined — the phase structure already rewards overshooting, and the
  end-to-end decode-vs-roofline judgment stays in Phase 4 per the plan.

Hard rule 6 now binds all of the above unmodified. P3-1 may proceed with
no open questions on the format or gates.

## 2026-08-26 — Process refinements: phase-exit doc surface + spec veto reporting (decided by James)

Two standing-process changes, folded into the operating docs so they bind
every future session:

- **Phase-exit close-outs refresh the newcomer-facing surface.** Every
  *-EXEC milestone now also updates README.md (Status, stale phrasing,
  doc tables) and any other doc a curious engineer would check when
  stumbling on the repo — same trigger as the architecture-PDF regen.
  Folded into CLAUDE.md ("Architecture document upkeep") and the P3-EXEC
  backlog note (pattern carries to all future *-EXEC milestones).
  **Catch-up done this session:** README.md was stale at "Phase 0 —
  starting; no engine code yet" (two exited phases behind); Status
  rewritten to the current Phase 2-exited state with measured numbers,
  and the "once it exists" test bullet corrected.
- **SPEC-Pn write-outs report veto-flagged decisions in-conversation.**
  Judgment-derived gate constants and new pins/schemas are reported
  directly in the session report, item by item with values and rationale
  — never only as a pointer into DECISIONS.md. Folded into
  AGENT_OPERATION.md step 11 and the SPEC-P4/P5/P6 backlog notes.
  (Retroactive instance: the Phase 3 flagged items were walked
  in-conversation 2026-08-26 and approved — see the veto-window entry
  above.)

## 2026-08-26 — P3-1: q4g64 packed layout + Swift packer landed; packed artifact produced

The pinned schema (D1/D2, approved 2026-08-26) is now code:
Sources/QwenMetalEngine/Quant/{Q4G64,Q4Packer,PackedCheckpoint}.swift +
CLI `pack` subcommand + 31 new tests (Q4G64Tests 12, Q4PackerTests 18,
ModelDirectoryTests +1). Full suite minus the CPU logit gate: 262 tests,
0 failures.

- **Packed artifact produced:** `models/qwen3-1.7b-70d244cc-q4g64.safetensors`
  (local-only), **968,083,288 bytes (0.968 GB)** — the spec's ~0.97 GB;
  197 packed matrices + 113 bf16 pass-through norms.
  **sha256 = 0feaa7ce4382cde9f9e3f7f08dbfad52fb4bae9a8ce8e94113930bdd8f0c93f3.**
  Determinism verified for real: two independent full packs of the 1.7B
  checkpoint are byte-identical (same sha256). Pack time ~21 s release.
- **Discovery — the consolidated checkpoint materializes the tie:** the
  4.06 GB source file carries `lm_head.weight` BYTE-IDENTICAL to
  `model.embed_tokens.weight` (verified by memcmp over the mapped ranges,
  622,329,856 bytes). Schema D1 stores the tied matrix once, so the packer
  omits `lm_head.weight` only after verifying byte identity (an untied
  lm_head packs normally; both behaviors pinned by tests). First pack
  without the dedupe measured 1.143 GB / 198 matrices — the dedupe
  recovers the ~0.17 GB the spec's 0.97 GB figure assumes.
- **Rounding-rule pin (schema implementation detail):** the D2 recipe's
  `round` is **round-half-away-from-zero** — the C/Metal `round()`
  semantics the P3-4 GPU kernels will share. Pinned in code comment and by
  an exact boundary test (q = 0.5 → 1). fp16-round-FIRST order pinned by a
  hand-computed test (scale 1/15: value 0.4999 → code 8 against the stored
  fp16 scale, 7 against the unrounded one).
- **Defensive addition beyond the spec list:** a group whose fp16 scale or
  bias would be non-finite (|w| beyond fp16 range — impossible in the
  pinned checkpoint) aborts the pack loudly (`groupRangeOverflow`), same
  spirit as the non-finite-weight abort. A failed pack removes its partial
  output file (a ~1 GB half-written artifact looks complete at a glance).
- **ModelDirectory discovery change:** bf16 checkpoint discovery now
  ignores `*-q4g64.safetensors` — the packed artifact legitimately lives
  beside the bf16 checkpoint in models/ (D2 pins the name) and was
  tripping the exactly-one rule (caught by the existing pinned-directory
  test the moment the artifact landed). P3-5 adds explicit packed-model
  loading; until then `generate` keeps resolving the bf16 checkpoint
  (verified end-to-end, GPU backend).
- **Parser:** TensorDType gained U32 (byteWidth 4) so the packed file rides
  the existing mmap parser; `fp32Values` rejects U32 with a clear error
  (packed tensors load via PackedCheckpoint only).

## 2026-08-29 — SUGGESTION (out-of-scope per non-goals): smarter quantization grouping for quality

Noted at James's request, flagged per hard rule 2 (quant-format changes are
on the PLAN.md non-goals list — recorded here as a suggestion, NOT seeded
as a task; picking any of it up is James's call and would need a scope
decision in this ledger).

q4g64 groups are purely positional (64 consecutive elements along the
reduction dim). Mechanisms known to recover quality over positional
grouped-affine, roughly in order of effort:

- **GPTQ-style error-compensated rounding:** keep the q4g64 layout
  byte-identical, choose codes by propagating rounding error across
  columns (Hessian-weighted) instead of independent nearest-round. File
  format unchanged — only the packer's code-selection changes — so
  loaders/kernels would not move; but dequant would no longer be "nearest
  to source weight", and the round-trip bound test + determinism pins
  would need re-derivation.
- **AWQ-style activation-aware scaling:** per-channel scales chosen from
  activation statistics before grouping; needs a calibration pass in
  tools/ and a schema addition (per-channel pre-scale tensor) — a schema
  re-pin.
- **Codebook/k-means (non-uniform levels):** replaces the affine map with
  a 16-entry LUT per group/tensor; schema re-pin + kernel LUT loads.
- **Smaller groups (g32) or super-blocks (K-quant style):** finer range
  adaptation at +overhead bytes/element (g32 doubles scale/bias overhead
  to ~0.625 B/elem, moving the roofline).

Constraints that make ALL of these expensive here: the q4g64 schema is a
PINNED invariant (2026-08-26 approval); any change re-pins the schema,
regenerates the packed artifact + sha256, and re-runs every oracle layer
(exact tile, Tier K/M/E, quality band vs mlx-lm 4-bit — whose premise
"measured identically to the MLX recipe" weakens if our recipe diverges).
Sensible trigger, if ever: P3-3's quality gate lands OUT of band, or a
later phase shows quality (not bandwidth) is the binding constraint.

## 2026-08-29 — P3-2: CPU-quant reference landed (packed → fp32 through the frozen CPU model)

The Phase 3+ oracle (spec D3, the PLAN invariant 4 carve-out) is now code:
`QwenModel` gained a `WeightSource` loading front end — `SafetensorsFile`
is the unchanged bf16 path (delegating convenience init, byte-identical
behavior), `PackedCheckpoint` (new `Quant/PackedWeightSource.swift`) is the
fp32 dequant materialization. The frozen forward modules cannot tell which
front end fed them; no CPU module changed. Hard rule 1 (register-only
dequant) continues to bind engine GPU + app unqualified — nothing GPU-side
was touched this task.

- **Tests:** PackedModelTests (5 synthetic tests: bit-exact `==` dequant
  through the front end vs the pinned group primitives, bit-identical bf16
  norm pass-through, tied-lm_head mapping, config/shape mismatch and
  untied-config-vs-tied-artifact loud rejects) + PackedModelSmokeTests
  (real artifact, pinned-revision load, skips cleanly if absent). All
  assertions exact — NO new tolerance constants, nothing veto-flaggable.
  Suite minus CPU logit gate: 268 tests, 1 skipped (env-gated free-run
  report), 0 failures.
- **Tied lm_head composes with no aliasing:** the packed artifact stores
  the tied matrix once (P3-1) and a tied config never requests
  lm_head.weight; an UNTIED config against the tied artifact fails with a
  loud tensorNotFound (pinned by test) — never a silent substitute.
- **Measured — the P3-1 wall-time warning was real:** materializing all
  197 matrices through the scalar per-element loop cost 229–262 s in a
  debug build (the IO-1 story repeating). `dequantMatrix` rewritten:
  per-group 16-entry LUT (each entry computed via the same pinned
  `Q4G64.dequant`) + `concurrentPerform` over disjoint group ranges —
  bit-identical outputs by construction (LUT alone changed nothing:
  262 s; parallelism is the lever). Now **42.7 s debug / 2.6 s release**;
  smoke decode tokens identical across scalar/LUT/parallel variants and
  both build modes. Exactness stays pinned by the P3-1/P3-2 `==` suites
  (all green unmodified).
- **Smoke decode (structural, no numeric gate — the quality band is
  P3-3):** short_english prompt "The capital of France is" → " Paris. The
  capital of France is Paris" — coherent; finite logits every step; all
  ids in-vocab; EOS/cap termination clean.
- **Dev-loop note:** the smoke test adds ~73 s to a debug `swift test`
  (28 s release). Escape hatch: `--skip PackedModelSmokeTests`; oracle
  suites consuming the CPU-quant reference should run release-mode and
  share ONE materialized model (SharedCheckpoint pattern) — note added to
  P3-5 in the backlog.

## 2026-08-30 — P3-3 (part 1): quality-gate band-setters MEASURED + dataset pin

Recorded BEFORE any metric of ours is computed (the ordering rule in the
2026-08-25 gates entry). Generator: tools/dump_quality_gate.py (new; pinned
venv, pyarrow==25.0.1 added to tools/requirements.txt). Committed outputs:
tests/fixtures/qwen3-1.7b-quality/ (band.json + ppl-slice blobs, sibling of
the FROZEN P1-1 set, which is untouched); drift test:
tests/test_quality_fixtures.py (stdlib-only, 7 tests).

- **Dataset pin (mandated by the gates entry to land with P3-3):**
  Salesforce/wikitext @ b08601e04326c79dfdd32d625aee71d232d685c3
  (unchanged upstream since 2024-01-04), config wikitext-2-raw-v1, TEST
  split, file wikitext-2-raw-v1/test-00000-of-00001.parquet, sha256
  5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91.
  Pinned in tools/pins.py. Slice protocol (documented in the dump script,
  per spec D6): `"".join` of the parquet text rows in stored order (rows
  carry their own trailing newlines), pinned HF tokenizer with
  add_special_tokens=False, first 4096 ids, one causal window,
  ppl = exp(mean NLL) over positions 1..4095, float64 NLL over fp32
  logits, KV-cached chunked prefill (chunk 512).
- **Reference-logits artifact (local-only, 151.9 MB):**
  models/qwen3-1.7b-70d244cc-ref-logits-250.bin — HF fp32 teacher-forced
  full-vocab logits, [5, 50, 151936] fp32 LE in fixture_prompts.json
  order, sha256
  b8a157f39972cb08917b388821f0f196de674ff76df8a1508b28b01268dd1261.
  Integrity at dump time: rows at the committed checkpoint steps
  {0,1,24,49} x 5 prompts came out BYTE-IDENTICAL to the P1-1 fixtures,
  and all 250 argmax steps reproduced steps.json — the oracle
  environment still regenerates Phase 1 exactly.
- **Band-setters (mlx-community/Qwen3-1.7B-4bit @ 3b1b1768, revision
  requested explicitly at download — measured per the gates-entry
  definitions, float64 stats over fp32 logits):**
  - **A_mlx = 219/250 = 0.876** (per prompt: short_english 45,
    multi_sentence 43, code_snippet 47, non_ascii 42, chat_template 42).
  - **KL_mlx = 0.114917 nats** (mean full-vocab KL(P_fp32 ‖ P_mlx)).
  - **ppl_fp32 = 13.996141; ppl_mlx = 16.470062; Δppl_mlx = 2.473921.**
- **The pre-committed gate formulas therefore resolve to:** A_ours ≥
  0.836; KL_ours ≤ 0.172375; Δppl_ours ≤ 3.720882. Constants unchanged
  from the 2026-08-25 entry (hard rule 6); our CPU-quant metrics are
  computed AFTER this entry, by the Swift QuantQualityGateTests.
- The 2026-08-22 argmax-level mlx dump remains unused for the band, as
  the gates entry requires (it predates these definitions).

## 2026-08-30 — P3-3 (part 2): quality gate OUT OF BAND; root cause = D2 recipe, not the packer implementation

Our CPU-quant metrics, measured by QuantQualityGateTests (release, Swift
float64 stats over fp32 logits, identical protocol to the band-setter dump)
AFTER the part-1 entry landed (commit 76b8351):

- **A_ours = 206/250 = 0.824** — gate ≥ 0.836: **FAIL** (−1.2 pp).
- **KL_ours = 0.357424 nats** — gate ≤ 0.172375: **FAIL** (3.11× mlx).
- **ppl_ours = 19.023943, Δppl_ours = 5.027802** — gate ≤ 3.720882:
  **FAIL** (2.03× mlx's Δppl).

Gates untouched (hard rule 6). Per the pre-committed diagnosis rule the
packer was the suspect; the layered-oracle arbitration ran the same day
(scratchpad experiments, all numbers below measured this session):

1. **Weight space exonerates the packer IMPLEMENTATION:** independent
   numpy dequant of the artifact reproduces the pinned schema, and
   per-tensor reconstruction RMSE vs fp32 is UNIFORMLY (slightly) BETTER
   than the mlx-community stored weights on ALL 197 packed tensors
   (ours/mlx ratio 0.984–0.988); tensor coverage identical both ways.
2. **Weights, not engines, cause the gap:** teacher-forced KL on
   short_english with the SAME HF fp32 engine, only weights swapped —
   our dequant weights 0.188103, mlx dequant weights 0.089006 nats. The
   latter matches mlx's own-engine number (0.086505), so the band-setter
   measurement and our Swift suite are both faithful.
3. **Locus:** swapping ONLY the MLP projections to mlx's closes most of
   the gap (0.188 → 0.086); embedding swap is negligible (0.186 / 0.081);
   attention swap intermediate (0.139).
4. **Root cause — the D2 recipe differs from MLX's actual recipe:**
   mx.quantize (probed on controlled groups) snaps the grid to an INTEGER
   ZERO-POINT (bias/scale exactly integer whenever 0 is in the group's
   range; e.g. b/s = −8.000, s = max(max/(15−z), −min/z)), so zero and
   near-zero weights reconstruct with ~zero systematic offset. Our pinned
   pure min/max affine (scale=(max−min)/15, bias=min) leaves zero
   OFF-GRID: the dense near-zero mass of trained weights picks up
   group-correlated residuals that add coherently across the 2048/6144
   reduction dims — lower per-element RMSE, but ~2–3× the distributional
   damage. The spec's premise (D1/D2 "matches the MLX recipe") was wrong
   at the code-selection level; logged per the conflicts-with-reality
   rule rather than worked around.
5. **Candidate fix validated end-to-end:** a zero-point-aligned q4g64
   recipe (z = clamp(round(−min·15/range), 0, 15), s covering both sides,
   b = −z·s; SAME schema, SAME group size, applied to the same fp32
   source) measures KL 0.092300 on the swap harness — at ecosystem
   parity (mlx 0.089) — despite higher RMSE (3.20e-3 vs ours 3.06e-3 on
   l0 down_proj). RMSE is the wrong objective; grid-zero alignment is
   what the band actually grades.

**Status/decision:** P3-3 CANNOT close in-band without amending the
pinned D2 packing recipe. The schema (D1: tensor triplet, u32 low-nibble
order, group 64, fp16 scales/biases) is NOT implicated and needs no
change; the amendment is to scale/bias/code SELECTION only. Because D2
was veto-approved 2026-08-25/26, the amendment waits for James
(AGENT_OPERATION pause rule): seeded QR-1 (decision, owner james) and
QR-2 (implementation: packer amendment + repack + P3-1/P3-2 re-verify)
in the backlog; P3-3 re-blocked on QR-2. The failing band tests are
committed as-is — they are the phase gate detecting a real defect, and
they skip cleanly on machines without the local artifacts. Full detail
for the decision: the fp16-rounding-before-code-selection principle and
the exact-dequant (fma) argument both SURVIVE the amendment unchanged —
only the (scale, bias) choice per group changes, so the Phase 3 gate
premises (bit-identical CPU/GPU dequant) are unaffected.

Artifacts of record this session: ref-logits artifact sha256 b8a157f3…
(part-1 entry), band.json (committed), Swift suite QuantQualityGateTests
(+ 6 synthetic metric unit tests, green), scratchpad diagnosis scripts
(diag_recipe/diag_all/diag_swap/diag_class/diag_class2/diag_mx_recipe/
diag_zp — session-local; numbers preserved in this entry).

## 2026-08-30 — QR-1 DECIDED (James): D2 amended to zero-point-aligned selection (option A1)

James reviewed the P3-3 part-2 diagnosis and the decision directions
(walked in-conversation with trade-offs) and chose **A1 — our own clean
zero-point-aligned selection rule**, rejecting A2 (bit-emulating
mx.quantize, with its reverse-engineered corner behaviors) and deferring
C (GPTQ/AWQ/codebook/g32 quality mechanisms).

- **The amended D2 recipe (binding from this entry on):** per group of 64
  fp32 values with min m, max M, range R = M − m:
  - Degenerate (R = 0): scale = 0, bias = fp16(m), all codes 0 — unchanged.
  - Else: zero-point z = clamp(round(−m·15/R), 0, 15) (round
    half-away-from-zero, the existing rounding pin); scale s = max of the
    applicable endpoint covers, M/(15−z) when z < 15 and −m/z when z > 0
    (both endpoints stay covered — no mlx-style clipping); fp16-round the
    scale FIRST, then bias b = fp16(−z·s16); codes
    q = clamp(round((w − b16)/s16), 0, 15) against the values AS STORED.
  - Everything else survives unchanged: schema D1 (triplet, u32
    low-nibble-first, group 64, fp16 scale/bias storage, alignment),
    the fp16-round-before-code-selection principle, the exact-dequant
    fma argument (q·s + b unchanged), all gate constants (hard rule 6).
  - Known fine print (accepted): b16 = fp16(−z·s16) is not always exactly
    −z·s16 (4+11 mantissa bits), so grid-zero can sit off by up to
    ~2⁻¹¹·|b| ≈ 2e-5 — vs up to ~1.5e-3 under the old recipe.
- **Basis:** validated end-to-end this session at KL 0.0923 vs mlx 0.089
  on the HF swap harness (part-2 entry, point 5).
- **C is deferred, not dead (James):** in a LATER optimization phase the
  part-2 suggestion-entry mechanisms (2026-08-29: GPTQ-style
  error-compensated rounding first among them) may be picked up with the
  explicit goal of beating mlx-4bit quality, not just matching it. Until
  that phase is chartered, quant-format/recipe upgrades beyond A1 remain
  on the non-goals list; recorded in TODOS.md so it cannot silently drop.
- **Execution:** QR-1 marked done; QR-2 (packer amendment + repack +
  P3-1/P3-2 re-verify) unblocked and proceeds now; P3-3 band suite
  re-runs against the SAME committed band.json (band-setters measure mlx
  and fp32, not us — they remain valid).

## 2026-08-30 — QR-2: amended recipe implemented; artifact repacked (new sha256)

- **Code:** Q4G64.packGroup (scale, bias) selection replaced with the QR-1
  A1 formulas (z = 0 pins bias to +0 explicitly); dequant, packWords,
  nibble order, error paths untouched. phase-3.md D2 carries a dated
  AMENDED note pointing at the QR-1 entry (superseded text kept, labeled).
- **Tests (red-first):** 3 new Q4G64Tests pin the amended selection with
  hand-derived fp16 constants — zero-straddling group (z = 4; b16 = −4·s16
  is fp16-EXACT there, so the 62 zero elements dequant to 0.0 bitwise),
  all-positive (z = 0, bias +0, grid [0, M]), all-negative (z = 15, grid
  top within the 2⁻¹¹·|bias| fine-print slack of zero). All three failed
  against the old recipe first (verified red), pass after. Every
  pre-existing pin survived UNMODIFIED — including fp16-round-first and
  half-away rounding (their min = 0 fixtures make old and amended recipes
  coincide), the scale-relative round-trip bound, determinism, and both
  overflow rejects. Quant suites: 15 + 17 + 5 green (debug), full 38 incl.
  real-artifact smoke green (release).
- **Artifact repacked:** models/qwen3-1.7b-70d244cc-q4g64.safetensors,
  968,083,288 bytes (size unchanged — schema identical), sha256
  **d073af49b3a37dd53be397d9c8045e36e38cc820701efcb1dc20b79692eef3ce**
  (supersedes 0feaa7ce…, which was the old-recipe pack). Two consecutive
  packs byte-identical (determinism re-verified); tied lm_head again
  memcmp-verified and stored once; 197 matrices + 113 bf16 pass-through
  norms; 22.6 s release.
- **Smoke note (honest):** the 8-token smoke continuation changed vs the
  old artifact (" Paris. The capital of France is Paris" → " in the city
  of ...? The question"). Structural assertions all pass; argmax-path
  flips are expected under a recipe change, and the committed quality
  judgment is the P3-3 band suite, not the smoke — running next.

## 2026-08-31 — QR-3 DECIDED (James): D2 selection re-amended to SNAP-SCALE; A1 measured 3.5% out of band

**A1 band result (the QR-2 artifact, QuantQualityGateTests release):**
A_ours = 214/250 = 0.856 (gate ≥ 0.836: PASS), Δppl_ours = 2.985137
(gate ≤ 3.720882: PASS), **KL_ours = 0.178458 (gate ≤ 0.172375: FAIL at
1.553× mlx vs the allowed 1.5×)**. Gates untouched (hard rule 6);
arbitration re-ran the same day:

- The HF swap harness reproduces the Swift number EXACTLY (0.178458) —
  measurement pipelines agree; the excess is recipe-borne. Per-prompt:
  short 0.0950 / multi 0.1782 / code 0.0766 / non_ascii 0.0768 /
  **chat_template 0.4657** (2.01× mlx's 0.2315) — the failure is carried
  by one prompt class, and a class-swap on chat pins it ENTIRELY on the
  MLP projections (swapping only MLP to mlx: 0.4657 → 0.1744; embedding
  and attention swaps make it WORSE — our A1 beats mlx on those classes).
- Root cause of the residual: A1 inflates the group step (up to ~50% in
  asymmetric groups) to keep BOTH endpoints on-grid; mlx's actual stored
  parameters (inspected on down_proj L13: z concentrated at 8–9,
  bias = min exactly, s ≈ |edge|/round(|edge|·15/range)) show the
  ecosystem rule keeps the step ≈ range/15 (optimal) and SNAPS the
  dominant-|.| endpoint onto the grid, letting the far endpoint clip by
  ≤ ~one step in rare cases. Asymmetric groups concentrate in the MLP
  tensors — exactly the located class. An argmin-z variant of A1 was
  tested and rejected (chat 0.4657 → only 0.4476; not the mechanism).
- **Snap-scale emulated end-to-end** (emulator bit-faithful: reproduces
  the A1 artifact's per-prompt KLs to 6 decimals): per-prompt KL
  0.091766 / 0.121147 / 0.075019 / 0.058372 / 0.213670 → **overall
  0.111995, BELOW mlx's 0.114917 (0.975×)**; agreement 219/250 — exactly
  mlx's count. down_proj L13 RMSE 3.1365e-3 ≈ mlx stored 3.1383e-3.

**Decision (James, in-conversation, after a full walk of the mechanism,
step-size significance, why snap edges out mlx — fp16 vs bf16 scale
storage + grid-consistent code selection — and both options):** amend the
D2 selection to snap-scale. **Binding formulas (supersede the 2026-08-30
QR-1 A1 selection):** per group with min m, max M, range R = M − m:
- Degenerate (R = 0): scale = 0, bias = fp16(m), codes 0 — unchanged.
- Else: s₀ = R/15; edge = m if |m| ≥ |M| else M (dominant-magnitude
  endpoint); q₀ = max(round(|edge|/s₀), 1) (round half-away-from-zero,
  the standing rounding pin; q₀ deliberately NOT clamped to 15 —
  single-sign groups keep the fine step, zero isn't in their range);
  s = |edge|/q₀; scale = fp16(s) FIRST; bias = fp16(k·scale) with
  k = −q₀ when m dominates (edge at code 0) and k = q₀ − 15 when M
  dominates (edge at code 15); codes q = clamp(round((w − bias)/scale),
  0, 15) against the values AS STORED. Selection arithmetic is evaluated
  in fp32 (as every prior recipe was) — the emulator that produced the
  validation numbers above uses identical fp32 arithmetic, so q₀ ties
  resolve the same way in both.
- Zero sits on the stored grid for EVERY zero-straddling group (there
  q₀ ≤ 15 always); the non-dominant endpoint may clip by |δ|·15·s₀/q₀
  ≤ ~one step (δ = the q₀ rounding residue) on at most one extreme per
  group. Schema D1, fp16-round-first, the fma exactness argument, and
  all gate constants remain untouched.
Honest accounting also recorded: the A1 proposal was validated on
short_english only — the least step-sensitive prompt — which is how the
covering-rule flaw slipped through; the snap validation above covers all
250 steps.

## 2026-08-31 — QR-3 landed + P3-3 CLOSED IN-BAND: all three quality gates PASS

**QR-3 implementation:** Q4G64.packGroup selection replaced with the
snap-scale formulas (red-first: the two single-sign A1 pins re-derived
and verified failing before the change; the zero-straddling pin is
grid-identical under both rules and stayed green by derivation). One
hand-derived expectation was corrected to the packer's actual fp32
arithmetic (fp32(1/15) rounds up, so the all-positive fixture's q₀ is 22,
not the real-number 22.5→23 — the emulator agrees bit-for-bit; noted in
the QR-3 formulas entry). Round-trip bound extended to one full step
(clipped-endpoint term, derivation in the test comment). Quant suites:
37 green debug, 38 green release incl. real-artifact smoke — which is
back to " in the city of Paris, and the" on the short_english prompt.

**Artifact (supersedes d073af49… / 0feaa7ce…):**
models/qwen3-1.7b-70d244cc-q4g64.safetensors, 968,083,288 bytes, sha256
**d03b3fe3a2a7a3ad393e42f195f790787ab746103f309545d0320c4727632fcc**;
two consecutive packs byte-identical; tied lm_head memcmp-verified and
stored once; 21.3 s release.

**P3-3 band verification (QuantQualityGateTests, release, 255 s —
2 tests, 0 failures) against the committed band.json band-setters:**
- **Top-1 agreement: A_ours = 218/250 = 0.872 ≥ 0.836 — PASS**
  (A_mlx 219/250; one argmax flip apart).
- **Mean KL: 0.115575 nats ≤ 0.172375 — PASS with 33% margin**
  (1.006× KL_mlx 0.114917 — parity).
- **Δppl: 2.500193 ≤ 3.720882 — PASS with 33% margin**
  (ppl_ours 16.496334 vs ppl_fp32 13.996141; 1.011× Δppl_mlx 2.473921).
All gates held at the pre-committed constants — nothing was loosened at
any point in the three-recipe arc (hard rule 6 observed throughout; the
two intermediate failures are recorded above as bug signals that were
fixed in the packer, exactly as the 2026-08-25 diagnosis rule
prescribed). Real-artifact numbers sit ~3% above the fp32-emulated
projection (KL 0.1156 vs 0.1120) — consistent with the engine-side
≤1e-3 CPU-vs-HF divergence the emulator does not carry; comfortably
inside the band either way.

**P3-3 exit criterion (spec D6 / exit table) is MET: the packing recipe
is in-band vs mlx-lm 4-bit on all three pre-committed metrics.** The
oracle chain now reads: HF fp32 (byte-verified) → CPU-quant reference
(quality-banded, this entry) → GPU-quant (P3-4/P3-5, pending). Unblocks
nothing yet (P3-7 also needs P3-5/P3-6); next ready task by rank is
P3-4.

## 2026-09-02 — P3-4 landed: fused q4g64 GPU kernels; layer-1 exact + Tier-K gates held first run

Landed `Sources/QwenMetalEngine/Metal/QuantKernels.swift` (spec D4) +
`QuantKernelTests` (13 tests): the dequant-tile dump (oracle layer 1,
test support), fused dequant-matvec (fp16- and fp32-store variants), and
embedding-gather-dequant. Dequant is in registers via the pinned
`float(q)·float(scale)+float(bias)` (hard rule 1; single-rounding
argument in the 2026-08-25 gates entry). Kernels take (buffer, byte
offset) per triplet tensor — synthetic tests bind three small buffers,
P3-5 binds the whole-checkpoint `GPUWeights` buffer three times; wrappers
convert to element offsets and reject misalignment (.q mod 4,
scales/biases mod 2), non-multiple-of-64 in-dims, short buffers, and
out-of-range token ids pre-dispatch.

**Gate outcomes (all pre-committed 2026-08-25, none touched):**

- Layer 1 EXACT: GPU tile dump bitwise == CPU dequant on hand-built
  adversarial fixtures (asymmetric nibble pattern, two-group boundary at
  elements 63/64/65, per-row group indexing, degenerate scale=0, fp16
  max-normal/min-normal/SUBNORMAL scale+bias, negative-heavy) and on
  packer-produced random matrices. Embedding-gather fp16 store bitwise
  == fp16(exact fp32) including rows whose dequant overflows fp16 to
  ±inf on store. Real-artifact check: layers.0 q_proj [2048, 2048] from
  the pinned packed artifact (d03b3fe3…) dequants bitwise-identical to
  `PackedCheckpoint.dequantMatrix` through the mmap GPUWeights buffer at
  real file offsets.
- Layer 2 Tier K (Phase 2 constant reused, max(2⁻⁹·M, 2⁻¹¹)): fused
  matvec vs BLAS.sgemm over the identical dequant fp32 weights (hard
  rule 8) — odd out-dims, both store variants, near-zero floor case,
  tied-embedding-as-lm_head structural case (edge 10), and the real
  q_proj shape. All held unmodified first run.
- Test-teeth check: deliberately swapping the kernel's nibble read to
  high-first failed the suite; reverting re-greened it (red evidence for
  an all-new exact suite).

Kernel is deliberately naive (one thread per output element, P2-2
shape): hard rule 3 puts correctness first; spec D4's optimization
license (simdgroup reductions, vectorized loads) is exercised against
the P3-6 microbench with these suites re-passed per iteration.

Verification: full suite minus the CPU logit gate, **release mode: 292
tests, 1 skipped (the QWEN_FREE_RUN_REPORT opt-in harness), 0
failures** (690 s). Release chosen deliberately: a debug-mode run
stalled >40 CPU-min inside P3-3's ppl-slice band test (the DEV-1/P3-2
wall-time species — 4096-token CPU-quant forward at -Onone); noted on
DEV-1 in the backlog rather than seeded as a new task. Gates are
build-mode-independent (exact == and Tier K on fp32 values).

No new pins, no schema changes, no judgment-derived constants
introduced. Unblocks P3-5 (pipeline wiring) and P3-6 (microbench).

## 2026-09-04 — P3-5 landed: GPU-quant pipeline wired; all Tier-M/E gates held first run; free-run divergence NONE

Landed the packed decode path end to end (spec D5). `GPUModel` gained a
q4g64 initializer (`init(packed: PackedCheckpoint, ...)`): the P3-4 fused
kernels serve the embedding gather and all matvecs (QKV/o/gate/up/down +
tied lm_head as [out, in], no transpose); 1-D bf16 pass-through norms ride
the Phase 2 RMSNorm kernel; attention/RoPE/SwiGLU/residual kernels
untouched (spec scope). Internally one `MatrixRef` enum (bf16 byte offset
vs q4 triplet byte offsets into the single whole-checkpoint GPUWeights
buffer, bound three times per triplet as P3-4 anticipated); the bf16 path
is the same code 1:1 and the full Phase 2 suite re-proves it unchanged.
Wrong-format loads both directions fail with clear errors naming the right
loader (edge 9, GPU side). Plumbing: CLI `--weights bf16|q4g64` (gpu OR
cpu — cpu+q4g64 is the CPU-quant reference), `ModelDirectory`
packed-artifact discovery (`packedCheckpointURL` /
`requirePackedCheckpoint()`, loud missing/multiple errors), app weights
toggle (bf16/q4g64 segmented control; **default q4g64** — P3-7 rows run
packed; reversible one-line default), `BenchmarkReport` records the
weights format (q4g64 rows export as "Phase 3 row export").

**Gate outcomes (constants reused verbatim per the 2026-08-25 gates entry;
none touched, all held FIRST RUN, release mode, live CPU-quant oracle):**

- Tier M: embedding-gather fp16 EXACT bitwise vs fp16(CPU-quant fp32) on
  the real packed triplet at real file offsets; layer-0 input RMSNorm at
  2⁻⁸ (bf16 norm weights read from the PACKED file); layer-0 attention
  module at 2⁻⁷ (quant matvecs for q/k/v/o + shared Phase 2 attention
  kernels).
- Tier E: full-stack slices at 2⁻⁵ (last_layer_output + final_norm_output,
  5 tokens); teacher-forced 5×50-step logit suite vs live CPU-quant logits
  — full-vocab checkpoints (steps {0,1,24,49}) at 2⁻⁵·M_step, float64
  fingerprints at 2⁻⁵·M64 (std 2⁻⁴·M64), top-64 at CPU-quant indices at
  2⁻⁵·M64, tie-aware top-1 at ε_tie = 2⁻⁴·M64 with margins from CPU-quant
  logits. Teacher-forcing uses the committed reference fp32 argmax
  sequences (identical prefixes, P1-5 premise; the P3-3 band suite
  teacher-forces the same sequences). GPU-quant suite block: 18 tests,
  847.8 s release (dominated by 250 live CPU-quant re-forwards).
- **Free-running divergence (REPORTED, not gated): NONE on all 5 prompts ×
  128 steps — GPU-quant and CPU-quant are token-identical** (the Phase 2
  result carries to the quantized fork). Texts coherent per prompt
  (short_english "…city of Paris…", chat_template a correct Rayleigh
  answer with <|im_end|>). Report harness: QWEN_FREE_RUN_REPORT=1, and
  QWEN_FREE_RUN_REPORT_FILE=<path> writes the report to a file — the
  current swift-test runner captures in-test stdout and does NOT replay
  it, so print-only reports are silently lost (cost one blind 30-min
  re-run; Phase 2's FreeRunReportTests has the same fragility → DEV-2).

**Measured (not derived):** dispatches/token on the packed path = **591**
(DispatchCounter at the dispatchThreads call sites — unchanged from bf16;
kernels replaced 1:1; tiny-model exact-count tests pin 22/24 at 1 layer).
Mac dev-loop observation (PROVISIONAL, not a row): CLI gpu+q4g64 median
GPU ~29 ms/token (debug host) vs bf16's 218.44 ms (P2-5) — the ~3.6×
weight-byte drop already showing before any kernel optimization; wall−GPU
~1.1 ms at 591 dispatches, consistent with P2-5.

CLI verified end-to-end: gpu+q4g64 coherent text + per-token block;
empty-prompt usage error, bad --weights value, and missing-packed-artifact
errors all clear; bf16 default output unchanged (byte-stable paths
untouched). iOS app target builds clean (Release, generic device,
unsigned) with the weights toggle; deploy stays James's.

Verification: **full suite minus the CPU logit gate, release: 316 tests,
2 skipped (the two opt-in free-run harnesses), 0 failures (1416.2 s)** —
including the complete Phase 2 bf16 GPU suites against the refactored
GPUModel (bitwise/gate behavior preserved). Tiny-model packed wiring
suite: 9 tests (incremental-replay bitwise, EOS/context stops, dispatch
counts, load rejects). New tests this task: 34 (GPUQuantModelTests 9,
GPUQuantTierM 3, GPUQuantTierE 1, GPUQuantLogitSuite 5, QuantFreeRunReport
1 opt-in, ModelDirectory +3, BenchmarkReport +2, plus the free-run
file-write path exercised by the report run).

No new pins, no schema changes, no numeric-gate constants introduced or
modified. Unblocks nothing new by itself (P3-7 waits on P3-6); next ready
by rank is P3-6 (bandwidth microbench).

## 2026-09-04 — P3-6 landed: dequant-matvec bandwidth microbench (D7) + Mac sanity row

Landed as Bench/QuantMatvecMicrobench.swift + CLI `microbench` subcommand +
an app Benchmark-screen microbench mode (report text assembled engine-side
via `exportText` so CLI and app print identical fields). The harness runs
ONE command buffer of one token's worth of REAL packed matvecs — 197
dispatches at the pinned dims (28 × {q,k,v,o,gate,up,down} + lm_head),
weights-only by construction — through the P3-4 `QuantKernels.encodeMatvec`
exactly as the decode pipeline binds them (fp16-store projections,
fp32-store lm_head, tied embedding triplet as [out, in], whole-checkpoint
GPUWeights buffer bound three times per triplet). No new kernels; no
kernel changes; the D4 optimization license remains unexercised (kernel
still naive) — this task builds the grader, P3-7 applies the gate.

**Metric implementation (per the pinned D7 definition):** aggregate
weight-stream rate = total packed bytes ÷ the aggregate command buffer's
GPU duration. Byte accounting is pinned by test at exactly **967,753,728
bytes** (= 0.5625 B/element × 1,720,451,072 matrix elements — the gates
entry's "q + scales + biases ≈ 0.967 GB"); the 197-dispatch count is
MEASURED via DispatchCounter, not derived. Dual timing everywhere (hard
rule 7); wall−GPU at 197 dispatches recorded per iteration.

**Conventions chosen (reversible, surfaced in-plan; NOT gates):**
- In-run protocol default 2 warmup + 10 measured (triad P0B-4 shape);
  the on-device gate consumes the BEST aggregate across the D8 repeats
  protocol (≥3 same-session runs), independent of per-run iteration count.
- Per-shape rates measured in separate per-role command buffers (8 roles)
  and reported alongside; the aggregate figure comes only from the
  197-dispatch single command buffer. Reported, never gated.
- Each site writes its own output buffer (573 KB fp16 + 608 KB fp32
  total) so Metal hazard tracking adds no false serialization; inputs are
  deterministic fp16-exact synthetic activations shared per in-dim.
- Pre-report spot check: layer-0 q_proj GPU output vs sgemm over
  `dequantMatrix` (hard rule 8) at the REUSED Tier-K gate; failure throws
  and withholds the figure (P0B-4 refuse-to-report precedent).

**Mac sanity row (PROVISIONAL, benchmarks/results.md Phase 3 section):**
M2 Pro, release, mmap, artifact d03b3fe3…: aggregate median **58.81 GB/s**
(best 60.05, range 58.59–60.05 over 10 iterations), overhead ~0.4 ms @
197 dispatches; spot check max |Δ| 0.000486 ≤ 0.0034. Observation: ~33%
of the Mac triad figure vs the Phase 2 bf16 naive matvec's ~9% — ~3.7×
closer to roofline before any optimization. Per-shape medians: q/o 46.3,
k/v ~23.6, gate/up 75.4, down 49.2, lm_head 110.5 GB/s — the small
per-layer matvecs individually underperform exactly as the gates entry
anticipated. Mac fraction is NOT a device predictor (Phase 2 precedent:
Mac 9% vs iPhone 50–70%); the 30.7 GB/s gate is decided only by P3-7.

Verification: new tests 12 (byte-accounting pins incl. the real-artifact
967,753,728/197 check, dual-timing sanity, per-shape/aggregate arithmetic
on hand-built results, spot-check teeth, iteration/shape validation edges,
untied/wrong-shape rejects, exportText fields). **Full suite minus the CPU
logit gate, release: 328 tests, 2 skipped (opt-in free-run harnesses),
0 failures (1672.5 s).** iOS app Release build verified (generic device,
unsigned); device runs stay James's (P3-7).

No pins, schema changes, or numeric-gate constants introduced or modified.
Unblocks P3-7 (now ready: P3-3, P3-5, P3-6 all done — owner james).
