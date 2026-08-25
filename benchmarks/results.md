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
