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
