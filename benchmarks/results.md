# benchmarks/results.md — measurement ledger

Rows are append-only (METHODOLOGY rule 7); never overwrite, add a new dated row.
Engine baseline rows (MLX / llama.cpp / ours) follow the PLAN.md benchmark
protocol and parity pins. Phase 0 rows are PROVISIONAL per the staleness rule.

## Bandwidth microbench (triad, P0B-4)

Kernel: STREAM triad `a[i] = b[i] + s·c[i]`, float4, fp32. Pinned protocol
(DECISIONS.md 2026-08-21): 3 × 384 MiB buffers = 1.125 GiB streamed/iteration,
2 warmup discarded + 10 measured, sustained = median of measured (GPU-timestamp
time base, wall recorded alongside; GB = 10^9 bytes). Reproduce with:
`swift run -c release qwen-metal-cli bandwidth`.

| Date | Device | OS | Toolchain | Sustained GB/s | Min–max GB/s | Mean dispatch overhead | Notes |
|---|---|---|---|---|---|---|---|
| 2026-08-21 | Apple M2 Pro (Mac, dev machine) | macOS 26.5.1 (25F80) | Xcode 26.6 (17F113), release | 178.19 | 172.47–179.85 | ~0.2 ms | PROVISIONAL, dev-loop sanity only — NOT the roofline denominator. ~89% of M2 Pro's rated 200 GB/s, confirming the working set defeats the SLC. |
| 2026-08-22 | Apple iPhone 15 Pro (A17 Pro, pinned device) | iOS 26.5.2 | Xcode 26.6 (17F113), Release, scratch device shell (benchmarks/device-shell/) | 43.84 | 42.19–44.45 | ~0.6–1.1 ms | PROVISIONAL. **THE roofline denominator** (PLAN.md invariant 1); recorded in DECISIONS.md 2026-08-22. Battery health 85%, >50% charge, rested-to-ambient (procedural check — unplugged + idle; no instrumented readout). ~85.6% of A17 Pro's rated 51.2 GB/s (cf. Mac row's ~89% of rated — same fraction ballpark ⇒ DRAM, not cache). Repeatability: three prior Debug-config runs, medians 43.28 / 43.67 / 43.89 (spread 42.43–44.36, overhead ~0.5–0.7 ms) — GB/s basis is GPU timestamps, unaffected by host build config. |

## Phase 0a baselines — MLX (PROVISIONAL)

### 2026-08-22 — MLX LLMEval, iPhone 15 Pro

Session: iOS 26.5.2, battery health 85%, >50% charge, rested-to-ambient
(procedural), Xcode 26.6 (17F113), **Release**, run from Xcode (memory =
phys_footprint gauge peak). Engine: LLMEval, mlx-swift-examples @ `378f2449`
+ the 2 pinned parity edits (revision-pinned checkpoint, temperature 0 — see
DECISIONS.md 2026-08-22); deps as built (xcworkspace Package.resolved):
mlx-swift-lm 3.31.3 (`1c05248b`), mlx-swift 0.31.4 (`dc43e62d`). Checkpoint:
mlx-community/Qwen3-1.7B-4bit @ `3b1b1768`. Feeding mode: raw user text,
LLMEval applies its own template, non-thinking default; outputs verified free
of `<think>` content. Greedy determinism observed: identical 1488-token
output across all three decode runs. Rate caveat: decode tok/s is the
app-reported overall generation rate, NOT the canonical 128–512 window
(windowed instrumentation arrives with our engine; Phase 6 re-measures).
Token-count note: LLMEval's template renders ~10 tokens more than the pinned
rendered form (app 862 vs pinned 852 on prefill-summarize — likely a default
system message); decode-essay app-side count not captured, ≈94 estimated
(pinned form = 84).

| Run | Prompt (tokens) | Mode | Cold/warm | TTFT | Generated | Total time | Decode tok/s | phys_footprint |
|---|---|---|---|---|---|---|---|---|
| 1 | decode-essay (≈94 app-side, est.) | burst | cold (load <1 min, not instrumented) | 1531 ms | 1488 | 37.9 s | 39.2 | 923 MB |
| 2 | decode-essay | burst | warm | 309 ms | 1488 | 37.8 s | 39.4 | 923 MB |
| 3 | decode-essay | burst | warm | 307 ms | 1488 | 38.7 s | 38.4 | 923 MB |
| 4 | prefill-summarize (862 app-reported) | prefill | warm | 2328 ms | 409 | 10.6 s | 38.5 | 923 MB |

Prefill tok/s ≈ 862 / 2.328 s ≈ **370** (TTFT includes one decode step ~26 ms;
prefill-only ≈ 374).

| Sustained (regenerate on EOS; LLMEval resets context per generation — matches pinned reset policy) | Value |
|---|---|
| Duration / generations | 5 min / 6 |
| First generation decode tok/s | 38.3 |
| Last generation decode tok/s | 26.7 (−30% — smooth thermal decline, no stutter) |

**Warm-burst median decode = 39.2 tok/s** → absolute success target committed
in DECISIONS.md: **0.75 × 39.2 = 29.4 tok/s**. Roofline context: 39.2 ≈ ~89%
of the ~44 tok/s naive ceiling (43.84 GB/s ÷ ~1.0 GB/token).

## Phase 0a baselines — llama.cpp (PROVISIONAL)

### 2026-08-22 — llama.swiftui, iPhone 15 Pro

Session: iOS 26.5.2, battery health 85%, start ~74% charge, rested, Xcode
26.6 (17F113), **Release**, **Metal API Validation OFF** (scheme diagnostics
— note: the MLX session ran with it ON; a validation-off MLX spot-check is
owed before Phase 6). Engine: llama.swiftui example @ llama.cpp b9999
(`47c78692`) + measurement/parity patches (greedy sampler; parse_special
tokenization; batch 512→2048; burst cap n_len 640; per-token print/UI-flush
removal; t/s computed from actual n_decode — upstream divided by the cap;
per-completion state reset — upstream no-ops every 2nd run; bundled pinned
prompts; all diffs live in ~/Projects/llama.cpp, documented in DECISIONS.md).
Checkpoint: locally converted models/qwen3-1.7b-70d244cc-Q4_K_M.gguf (5.03
BPW, sha256 in DECISIONS.md). Feeding mode: pinned RENDERED prompts
(bundled), no app-side template. Prompt counts: 84 / 852 (llama.cpp = HF
counts here). Zero <think> content. phys_footprint: not captured this
session (gauge not observed — follow-up). Harness-defect anecdotes (NOT
rows): two earlier 2048-cap runs measured 10.47 and 14.26 t/s — UI
re-layout + console printing + mis-computed rate, since fixed.

| Run | Prompt (tokens) | Mode | Cold/warm | TTFT | Generated | Total time | Decode tok/s |
|---|---|---|---|---|---|---|---|
| 1 | decode-essay (84) | burst | cold | 414 ms | 556 (cap) | 17.96 s | 30.95 |
| 2 | decode-essay (84) | burst | warm | 223 ms | 556 (cap) | 17.14 s | 32.44 |
| 3 | decode-essay (84) | burst | warm | 225 ms | 556 (cap) | 16.01 s | 34.73 |
| 4 | prefill-summarize (852) | prefill | warm | 1917 ms | 617 (EOS) | 21.80 s | 28.31 |

Prefill tok/s ≈ 852 / 1.89 s ≈ **452** (TTFT minus one decode step). Run 4's
lower decode rate (28.31) reflects the deeper KV (852-token prompt) —
consistent with bytes/token growth.

| Sustained (repeat decode-essay sends, same session) | Value |
|---|---|
| Duration / generations | 5 min / 13 |
| First generation decode tok/s | 31.42 (TTFT 227 ms) |
| Last generation decode tok/s | 20.88 (TTFT 226 ms; −34% — smooth thermal decline) |

Auxiliary (app Bench button, validation OFF, run AFTER the sustained loop —
thermally loaded): pp512 265.11 ± 22.88 t/s (vs 430.83 ± 51.08 rested with
validation ON — thermal state dominates prefill), tg128 31.55 ± 0.37 t/s
(vs 26.11 ± 0.12 validation ON — validation cost ~17% on decode).

**Warm-burst median decode = 32.44 tok/s ≈ 83% of MLX's 39.2.** Roofline
note (rough BPW math, see DECISIONS.md): 32.44 × ~1.3 GB/token ≈ 42 GB/s
≈ 96% of the 43.84 triad figure; the MLX equivalent lands ≈ 102%. Both
engines saturate ~triad-level bandwidth — decode is read-dominated and the
2R+1W triad likely understates read-mostly achievable bandwidth (follow-up
BW-1 seeded).

## Phase 2 — qwen-metal GPU backend, Mac dev-loop sanity (PROVISIONAL)

### 2026-08-25 — first instrumented decode rates, M2 Pro (P2-5)

NOT comparative, never a baseline (spec: "catches gross regressions before
device time is spent"). Session: macOS 26.5.1 (25F80), Xcode 26.6 (17F113),
`swift run -c release`, backend `gpu` (bf16 mmap weights, fp16 activations,
naive kernels, 448 MiB KV cache at the 4096 pinned context). Prompt =
pinned rendered decode-essay (84 tokens — note: `--prompt "$(cat file)"`
strips the rendered form's trailing `\n\n` and tokenizes to 83; restore it
with `$'\n\n'` or the count drifts). Greedy, stop set {151645, 151643},
burst cap 640 (llama.cpp-runbook cap). Rates are the engine's native
instrumentation (P2-5): medians over per-token dual timing, canonical
window per PLAN.md (completion of generated token 128 → completion of 512,
384 tokens). Repeatability: an earlier same-session run (83-token prompt
variant) measured median GPU 218.51 ms / window 4.56 — identical to 3
digits.

| Date | Device | Prompt (tokens) | Generated | Median GPU ms/tok | Median wall ms/tok | Median wall−GPU ms | Dispatches/tok | Window tok/s (128–512) | Overall decode tok/s | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-25 | Apple M2 Pro (Mac, dev machine) | decode-essay (84) | 640 (cap) | 218.44 | 218.83 | 0.391 | 591 | 4.56 | 4.56 | PROVISIONAL, burst, warm (2nd run of session). Naive-by-design: ~9% of the M2 Pro naive roofline (178.19 GB/s ÷ 3.44 GB/token ≈ 52 tok/s) — the one-thread-per-output matvec is the known Phase 3–5 target; dispatch overhead is tiny on Mac (0.39 ms of 218.8 ms). Output coherent (computing-history essay). |

## Phase 2 — qwen-metal GPU backend, iPhone 15 Pro "before" rows (P2-7, PROVISIONAL)

### 2026-08-25 — two sessions: Xcode-attached (invalidated) + detached rerun (James)

Conditions common to both sessions: iPhone 15 Pro (iPhone16,1), iOS 26.5.2,
QwenMetalApp Release, greedy, stop set {151645, 151643}, pinned rendered
prompts (bundled in-app), engine-native P2-5 instrumentation, Increased
Memory Limit entitlement, starting temps ambient. Battery figures in rows
are **SoC start→end** (the export's battery field was used for SoC; Battery
Health displays "Normal" on this iOS build — % not read; ≈85% at P0A-1).
Model load: **mmap 1.5 s** (lazy, file-backed) vs **wiredCopy 9.7 s**
(3.44 GB copy). phys_footprint, Xcode gauge (the pinned metric of record):
**mmap ~536 MB, wiredCopy ~4.3 GB** — in-app task_info cross-check within
~2% of the gauge in every run; the mmap figure excludes the 3.44 GB of
clean file-backed weight pages (the P0A-1 llama.cpp 307 MB accounting
asymmetry, now quantified on our side, per PLAN.md invariant 3). Thermal:
phone stayed notably cooler than the Phase 0 MLX/llama.cpp sustained
cycles.

**Session 1 was launched from Xcode's Run button** (established after the
fact): Metal API validation ON + debugger attached, so per the P0A-1
validation-off pin those rows are **INVALID for comparative/headline use**.
Kept below for the record — they quantify the attached-run penalty at
**1.4–1.9× on per-token GPU time** for this 591-dispatch/token workload
(cf. P0A-1: MLX ~1%, llama.cpp 17–21% — strongly engine-dependent).
**Session 2 was relaunched detached from the home screen (validation OFF)
— these are the valid "before" rows.**

#### Session 2 (detached — VALID): burst + prefill rows (cap 640)

| Date | Residency | Prompt (tokens) | Cold/warm | Generated | Prefill s (tok/s) | Median GPU ms/tok | Median wall ms/tok | Median wall−GPU ms | Disp/tok | Window tok/s (128–512) | Overall tok/s | SoC | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-25 | mmap | decode-essay (84) | cold | 640 (cap) | 14.44 (5.82) | 111.71 | 113.72 | 2.026 | 591 | 8.33 | 7.81 | 79→78 | PROVISIONAL. Fastest burst of the session despite being cold — cold/warm is not what separates the speed clusters. |
| 2026-08-25 | mmap | decode-essay (84) | warm | 640 (cap) | 7.88 (10.66) | 159.12 | 161.12 | 2.021 | 591 | 6.74 | 6.91 | 78→76 | PROVISIONAL. **Protocol headline row** (warm burst): window 6.74. |
| 2026-08-25 | mmap | decode-essay (84) | warm | 640 (cap) | 9.63 (8.72) | 157.48 | 159.39 | 1.997 | 591 | 6.92 | 7.01 | 76→75 | PROVISIONAL. Warm repeat: window 6.92 — agrees with the other warm burst to ~3%. |
| 2026-08-25 | mmap | prefill-summarize (852) | warm | 462 (eos) | 103.48 (8.23) | 184.62 | 186.59 | 1.983 | 591 | n/a (<512 gen) | 5.31 | 75→73 | PROVISIONAL — **the prefill row**: 852 tokens in 103.5 s = 8.23 tok/s sequential (Phase 5's target). Generated exactly 462 tokens then EOS — identical count to session 1's run (determinism across sessions). |
| 2026-08-25 | wiredCopy | decode-essay (84) | warm | 640 (cap) | 9.39 (8.94) | 111.87 | 113.79 | 1.985 | 591 | 7.81 | 7.53 | 72→71 | PROVISIONAL. Wired burst lands in the fast cluster — no residency penalty distinguishable from state noise. |

#### Session 2 (detached — VALID): sustained rows (5-min regenerate loop)

| Date | Residency | Loop | Median GPU ms/tok (last gen) | Median wall−GPU ms | Window tok/s (128–512) | Overall tok/s | Notes |
|---|---|---|---|---|---|---|---|
| 2026-08-25 | mmap | gen 0: 1,601 tokens in 282.8 s, **stop: eos**; gen 1: 43 tokens (truncated) — 1,644 tokens total | 150.66 (gen 1, shallow cache) | 1.903 | 8.64 (gen 0) | 5.96 (gen 0) / 6.52 (gen 1) | PROVISIONAL. The loop correctly regenerated on EOS (the decode-essay greedy trajectory ends at generated token 1601 — consistent with every shorter run never seeing EOS). Gen 0 overall 5.96 < window 8.64 reflects attention-depth growth over 1,601 tokens. |
| 2026-08-25 | wiredCopy | gen 0: 1,595 tokens in 300.0 s (truncated) | 174.73 (full gen, median depth ~880) | 1.910 | 7.46 | 5.46 | PROVISIONAL. 1,595 tokens EOS-free — consistent with the same trajectory (EOS at 1601). |

#### Session 1 (Xcode-attached, validation ON — kept for the record, NOT comparative)

| Date | Residency | Prompt (tokens) | Mode | Cold/warm | Generated | Prefill s (tok/s) | Median GPU ms/tok | Median wall−GPU ms | Window tok/s | Overall tok/s | SoC | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-25 | mmap | decode-essay (84) | burst | cold | 640 | 22.80 (3.68) | 217.16 | 1.987 | 4.53 | 4.51 | 54→54 | ATTACHED. |
| 2026-08-25 | mmap | decode-essay (84) | burst | warm | 640 | 17.30 (4.85) | 216.71 | 1.936 | 4.53 | 4.53 | 54→53 | ATTACHED. Export reported twice ("A warm"/"B1") — recorded once. |
| 2026-08-25 | mmap | decode-essay (84) | burst | warm (labeled) | 640 | 22.81 (3.68) | 216.89 | 2.023 | 4.54 | 4.53 | 69→69 | ATTACHED. Prefill/footprint match the cold signature (likely fresh launch after a recharge break). |
| 2026-08-25 | mmap | prefill-summarize (852) | burst | warm | 462 (eos) | 164.88 (5.17) | 398.25 | 1.565 | n/a | 2.54 | 51→48 | ATTACHED. Same 462-token EOS count as the detached rerun. |
| 2026-08-25 | wiredCopy | decode-essay (84) | burst | warm | 640 | 17.14 (4.90) | 217.77 | 1.975 | 4.43 | 4.44 | 66→65 | ATTACHED. |
| 2026-08-25 | mmap | decode-essay (84) | sustained | warm | 930 (truncated) | — | 323.86 | 1.766 | 2.92 | 3.29 | 69→68 | ATTACHED. The apparent "mmap 42% slower sustained" signal here did NOT reproduce detached — see analysis. |
| 2026-08-25 | wiredCopy | decode-essay (84) | sustained | warm | 1,211 (truncated) | — | 228.41 | 1.968 | 4.54 | 4.27 | 68→66 | ATTACHED. |

#### Analysis

- **Headline "before" decode (protocol row: warm burst, canonical window,
  detached): 6.74–6.92 tok/s (mmap)** = 23% of the committed 29.4 target,
  17.6% of MLX's 39.2. Same-session fast-state runs reached 7.81 (wired
  burst), 8.33 (cold burst), 8.64 (sustained gen 0) — honest reporting is
  the range **6.7–8.6 tok/s**, not a single number.
- **Run-to-run device-state variance ~1.4×** within the detached session:
  two clusters at ~112 and ~158 ms/token median GPU under identical
  settings; cold/warm is not the driver (cold was fastest). Cause
  unidentified (device power/thermal governor state). Consequence: Phase 3+
  comparative rows need repeats/interleaving — folded into SPEC-P3.
- **Dispatch overhead (the Phase 4 metric): 1.9–2.0 ms/token at 591
  dispatches ≈ 3.4 µs/dispatch in ALL rows of both sessions and both
  residencies** (vs Mac's 0.39 ms) — the one number untouched by session
  and state variance. 1.2–1.8% of today's token; ~6% of a Phase 3-scale
  33 ms token.
- **Residency (OV#9): UNRESOLVED for speed.** The attached session showed
  mmap 42% slower sustained; the valid detached session showed mmap
  *faster* (window 8.64 vs 7.46) — both directions observed, and the
  deltas sit inside the state-noise band. Footprint and load time DID
  measure cleanly (mmap 536 MB / 1.5 s vs wired 4.3 GB / 9.7 s). Default
  residency decision: DECISIONS.md 2026-08-25 (mmap; Phase 3 re-tests
  interleaved on packed weights).
- **Determinism across runs, modes, and sessions:** prefill-summarize
  generated exactly 462 tokens then EOS in both sessions; decode-essay's
  greedy trajectory EOSes at generated token 1601, consistent with the
  640-, 930-, 1,211-, and 1,595-token runs never reaching it.
- **Roofline position:** fast-state 111.7 ms/token ⇒ ~30.8 GB/s weight
  traffic ≈ **70% of the 43.84 GB/s roofline** — on-device decode is far
  closer to memory-bound than the Mac sanity row implied (M2 Pro 218 ms ≈
  9% of its roofline; the attached-run "iPhone == Mac" coincidence is
  dead). Mid-state ≈ 50%. Phase 3 packed roofline = 45.2 tok/s
  (43.84 ÷ 0.97); at the observed 50–70% efficiency ⇒ ~22–32 tok/s,
  bracketing the 29.4 target — Phases 4–5 remain necessary, not optional.
- **Prefill "before" (warm, detached): 8.2–10.7 tok/s** sequential
  (spec D6; Phase 5's target).

## Phase 0a — energy dry-run + corrections (PROVISIONAL)

### 2026-08-22 — sustained battery-delta cycles, iPhone 15 Pro (method VALIDATED)

Conditions (both cycles + idle): DETACHED from Xcode (home-screen launch),
airplane mode, minimum brightness, Auto-Lock Never, Background App Refresh
off, unplugged, rested; Metal API Validation OFF; battery health 85%
(capacity basis 12.6 Wh rated × 0.85 = 10.74 Wh ⇒ 1% SoC = 387 J); sustained
regenerate-loop (decode-essay, fresh context per generation) via the Loop
patches (benchmarks/patches/). Idle baseline measured once (LLMEval
foregrounded, no generation): 1% / 15 min ≈ 0.43 W, scaled pro-rata to each
cycle. SoC read quantization ±0.5%/reading ⇒ ~±12% on J/token.

| Engine | SoC band | Wall | Gens | Tokens | Gross W | Net W (idle-corr.) | **Net J/token** | first→last t/s |
|---|---|---|---|---|---|---|---|---|
| MLX LLMEval | 81→71% | 1055 s | 23 | 32,840 | 3.67 | 3.24 | **0.104** | 40.43 → 36.78 (−9%) |
| llama.cpp (llama.swiftui) | 69→59% | 1057 s | 47 | 26,132 | 3.66 | 3.23 | **0.131** | 32.81 → 18.17 (−45%) |

Both cycles inside the 3–9 W plausibility window ⇒ **battery-delta energy
method VALIDATED** (one cycle per engine; ≥3-repeat rounds are Phase 6).
Recorded deviation: the llama.cpp cycle ran 69→59%, not the pinned 80→70%
band (single-session sequencing); acceptable for method validation, Phase 6
comparative rounds use the pinned band. At identical ~3.7 W draw, MLX
delivered ~25% more tokens per joule. Thermal contrast under identical
conditions: MLX −9% vs llama.cpp −45% over ~17.6 min — engine-level
difference, not harness; note llama.cpp's end-state 18.17 t/s is below the
29.4 target, so sustained-regime framing matters for Phase 6.

### 2026-08-22 — validation-off MLX spot check + phys_footprint corrections

- MLX warm burst, Metal API Validation OFF, attached: **39.6 t/s** (TTFT
  319 ms, prompt 94, 1488 tokens, 37.5 s) vs 39.2 validation-on ⇒ validation
  overhead is ENGINE-DEPENDENT (~+1% MLX vs ~17–21% llama.cpp). Committed
  target (29.4 = 0.75 × 39.2) stands; 0.75 × 39.6 = 29.7 confirms it was not
  materially understated. Prompt length 94 confirms the earlier ≈94 estimate;
  1488-token determinism holds across validation settings.
- **phys_footprint corrections (Xcode memory gauge = the pinned metric):**
  MLX = **1.02 GB** (the earlier rows' "923 MB" was the app's own MLX
  activeMemory meter, mislabeled — rows stand, this addendum corrects the
  metric); llama.cpp = **307 MB**, an mmap accounting artifact: the 1.19 GB
  GGUF is clean file-backed pages largely excluded from phys_footprint,
  while MLX's weights are dirty/anonymous buffer memory. The 307 MB vs
  1.02 GB comparison is NOT an efficiency claim — exactly the iOS
  memory-accounting asymmetry PLAN.md invariant 3 anticipates (Phase 2
  mmap-vs-wired bench will quantify it for our engine).
- Sustained-decline reconciliation: earlier ATTACHED 5-min loops showed −30%
  (MLX) / −34% (llama.cpp); detached, airplane-mode cycles show −9% / −45%.
  MLX's earlier decline was substantially harness load (debugger/radio/
  brightness); llama.cpp's is genuinely thermal. Sustained rows must be
  measured detached.
