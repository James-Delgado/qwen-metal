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

## Phase 3 — dequant-matvec bandwidth microbench (D7, P3-6)

Harness: one command buffer running one token's worth of REAL packed q4g64
matvecs (28 layers × {q,k,v,o,gate,up,down} + lm_head = 197 dispatches,
weights-only — no attention/norm/elementwise/KV), through the P3-4 fused
kernels exactly as the decode pipeline binds them. Aggregate weight-stream
rate = 967,753,728 packed bytes (q + scales + biases) ÷ command-buffer GPU
time; per-shape rates from separate per-role command buffers, reported but
never gated. 2 warmup discarded + 10 measured per run; dual timing per hard
rule 7; a Tier-K spot check vs the CPU-quant oracle withholds the figure on
failure. The pre-committed gate (aggregate ≥ 0.70 × 43.84 = 30.7 GB/s, best
across the D8 repeats protocol) applies to the pinned iPhone ONLY (P3-7,
James). Reproduce with:
`swift run -c release qwen-metal-cli microbench --model-dir models`.

### 2026-09-04 — Mac dev-loop sanity row, M2 Pro (P3-6)

| Date | Device | OS | Toolchain | Aggregate median GB/s | Best | Min–max | Median overhead (wall−GPU) | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-04 | Apple M2 Pro (Mac, dev machine) | macOS 26.5.1 (25F80) | Xcode 26.6 (17F113), release | 58.81 | 60.05 | 58.59–60.05 | ~0.4 ms @ 197 dispatches | PROVISIONAL, dev-loop sanity only — never gated (gate is on-device, P3-7). mmap residency, artifact d03b3fe3…. Spot check max \|Δ\| 0.000486 ≤ Tier-K 0.0034. ~33% of the Mac triad figure (178.19 GB/s) vs Phase 2's ~9% — the fused kernel moves ~3.7× closer to the Mac roofline than the bf16 naive matvec, still naive-by-design (D4 optimization license is open, gated by P3-7). Per-shape medians: q/o 46.3, k/v ~23.6, gate/up 75.4, down 49.2, lm_head 110.5 GB/s — small per-layer matvecs individually underperform exactly as the gates entry anticipated; the aggregate is the roofline-relevant number. |

### 2026-09-04 — On-device Phase 3 rows, iPhone 15 Pro (P3-7, James)

One session, detached launches (home-screen, no debugger), Metal API
validation OFF (recorded), greedy, pinned prompts, weights q4g64 (artifact
d03b3fe3…), iOS 26.6.1 (earlier rows: 26.5.2), Xcode 26.6 (17F113).
Battery health "Normal" / 100% max capacity per iOS (see the 2026-09-05
DECISIONS correction: the reports' battery fields carried state-of-charge,
not health — charge ran 77% → 56% across the session). All rows under the
D8 protocol: ≥3 same-session repeats, median AND range, A/B interleaved.
Memory rows only are Xcode-attached (not timing rows).

#### Microbench gate (D7): **PASS — best aggregate 35.29 GB/s ≥ 30.7**

5 detached repeats (run 1 cold, 2–5 same-session), mmap, 2+10 iterations
each; spot check passed identically on every run (max |Δ| 0.000486 ≤
0.0034, deterministic inputs).

| Run | Aggregate median GB/s | Best | Min–max | Overhead (wall−GPU) @197 |
|---|---|---|---|---|
| 1 (cold) | 34.84 | 35.29 | 33.10–35.29 | 0.8–1.5 ms |
| 2 | 34.71 | 34.98 | 34.49–34.98 | 0.7–1.6 ms |
| 3 | 34.15 | 34.43 | 33.08–34.43 | 0.7–1.6 ms |
| 4 | 34.65 | 35.02 | 34.30–35.02 | 0.7–1.5 ms |
| 5 | 34.40 | 35.03 | 33.92–35.03 | 0.7–1.6 ms |

Gate basis: best across repeats = **35.29 GB/s = 80.5% of the 43.84 GB/s
roofline** (median-of-medians 34.65 = 79%). Robust pass: every one of the
50 measured iterations (worst 33.08) individually clears 30.7. Run-to-run
median spread ~2% — far tighter than decode's ~1.4× device variance.
Per-shape medians across runs: k/v ~24–26, q/o ~29–32, lm_head ~31–32,
down ~34–35.5, gate/up ~35–37 GB/s (reported, never gated). Kernel still
naive — D4 optimization license unexercised.

#### Packed decode, warm burst (decode-essay 84, cap 640, mmap, ×3)

| Repeat | Window tok/s (128–512) | Overall | Median GPU ms/tok | Wall−GPU ms | Dispatches | Prefill tok/s |
|---|---|---|---|---|---|---|
| 1 | 20.88 | 20.78 | 45.01 | 1.990 | 591 | 23.79 |
| 2 | 20.61 | 20.61 | 45.74 | 1.972 | 591 | 28.64 |
| 3 | 20.47 | 20.45 | 45.37 | 1.998 | 591 | 28.42 |

**Window median 20.61 tok/s, range 20.47–20.88** — ~3.0× the Phase 2
"before" (6.74–6.92) against a 3.56× weight-byte drop; 70% of the 29.4
target; just below the P2-7 projection band (~22–32). Effective weight
stream during full decode ≈ 0.968 GB ÷ 45.4 ms ≈ **21.3 GB/s ≈ 49% of
roofline** vs the microbench's ~79–80% for matvecs alone — the ~17 ms/token
gap (attention, norms/elementwise, inter-dispatch time at 591
dispatches/token) is Phase 4's named target. Stop = maxNewTokens all runs.

#### Sustained 5-min loops, mmap vs wired INTERLEAVED (D8: A,B,A,B,A,B)

Canonical-window tok/s of completed generations (1297 tokens each, every
one stopping at EOS at exactly token 1297 — greedy determinism held
bitwise across modes and 35 min of thermal drift):

| # | Mode | Gen windows (tok/s) | In-app footprint | Charge at start |
|---|---|---|---|---|
| 1 | mmap | 20.85, 19.21, 16.21 | 551.0 MB | 74% |
| 2 | wired | 20.60, 17.60, 17.68 | 1476.5 MB | 70% |
| 3 | mmap | 20.58, 17.80, 17.45 | 559.4 MB | 66% |
| 4 | wired | 18.12, 16.70, 16.64 | 1483.8 MB | 63% |
| 5 | mmap | 17.00, 16.85, 16.80 | 562.9 MB | 59% |
| 6 | wired | 17.14, 16.86, 16.77 | 1473.7 MB | 56% |

Per-side stats (9 completed gens each): mmap median 17.45, range
16.21–20.85; wired median 17.14, range 16.64–20.60. **Ranges overlap ⇒
speed unresolved at n=3** (the D8 rule). The interleaving shows why:
first-gen windows decline monotonically across the session (20.85 → 17.14)
regardless of mode — session-scale thermal drift dominates any residency
effect, settling at a **~16.8–17.1 tok/s equilibrium** (sustained/burst
≈ 0.82). **No mmap bimodality**: no generation shows a page-fault-stall
signature; the 0.97 GB packed working set stays resident. Overhead steady
~1.9 ms/token throughout (last-gen outliers are 1–30-token samples; run
5's 68.3 ms last-gen median is the thermal trough, mode-independent).

#### Memory rows (attached — Xcode gauge is the metric of record) + load

| Residency | Xcode gauge (steady, decoding) | In-app phys_footprint | Load (fresh instance) |
|---|---|---|---|
| mmap | 537.8 MB | 539.3 MB | 0.5 s |
| wiredCopy | **1.43 GB** | 1463.1 MB | 2.6 s |

Gauge vs in-app agree within ~2% in both modes (validates the in-app
cross-check for detached sessions). **Memory-drop criterion met**: wired
(honest total-resident) 1.43 GB vs Phase 2's 4.3 GB (3.0× whole-process;
weight bytes 3.44 GB → 0.968 GB = 3.56×, the plan's "~4×" stated
honestly), landing on the derived ~1.5 GB budget. mmap's 538 MB
under-reports file-backed weights per the Phase 2 annotation precedent
(≈ KV 448 MiB + activations + app).

**Residency close-out (decided by James, 2026-09-05 — DECISIONS entry):
mmap stays the default for Phase 4+.** Speed unresolved at n=3 under the
interleaved protocol; mmap is ~1 GB lighter and loads 5× faster; wired
stays available via the app toggle for diagnostics.

## Phase 4 — per-stage GPU attribution + latency variance, Mac dev-loop (P4-1, PROVISIONAL)

Harness (P4-1, phase-4.md D1/D7): the diagnostic attribution mode replays
the SAME encode sequence split into one command buffer per class-contiguous
dispatch run (282 segments/token at real dims; classes matvec / attention /
norm+elementwise / head-tail), buffers committed back-to-back on the serial
queue, per-class GPU time from the buffers' own timestamps; every other
decode token runs the untouched production single-command-buffer step as the
cross-check reference (pre-committed sanity band 0.5–2.0×, DECISIONS.md
2026-09-08). DIAGNOSTIC — never a benchmark row; production-path invariance
is test-pinned (bitwise-identical logits, 591 dispatches). Latency variance
(D7, reported never gated): nearest-rank p50/p95/p99/max of
completion-to-completion per-token wall spans over the canonical window +
stall count (spans > 2× window p50). Reproduce:
`swift run -c release qwen-metal-cli attribute --model-dir models --prompt "$(cat benchmarks/prompts/rendered/decode-essay.rendered.txt)"$'\n\n'`
(and `generate … --max-tokens 640` for the variance row; note the `$(cat)`
trailing-newline restore, P2-5 annotation).

### 2026-09-08 — Mac "before" (naive-path) attribution breakdown, M2 Pro (P4-1)

Session: macOS 26.5.1 (25F80), release build, weights q4g64 (artifact
d03b3fe3…), residency mmap, prompt decode-essay (84 tokens), 32 attributed
+ 32 production forwards interleaved at cache depth 83–146. Mac fractions
do NOT predict device fractions (Phase 2 precedent) — the claim-grade
"before" breakdown is the on-device P4-5 export.

| Date | Device | matvec | attention | norm+elementwise | head/tail | Class-sum (median ms/tok) | Production GPU ms/tok @ dispatches | Sanity ratio | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-08 | Apple M2 Pro (Mac, dev machine) | 14.95 ms (49.4%) | 2.45 ms (8.1%) | 11.12 ms (36.7%) | 1.77 ms (5.9%) | 30.29 (span 30.52, wall 30.69) | 30.27 @ 591 | 1.00 (band 0.50–2.00) | PROVISIONAL, DIAGNOSTIC. Shallow-depth attention (83–146) — the on-device row at window depth will weight attention higher. Elementwise at 36.7% of token GPU time is the Mac-visible headline for the D3 fold set; matvec + head/tail (the class split puts lm_head in head/tail) ≈ 16.7 ms, matching the P3-6 microbench expectation for all 197 weight-streaming dispatches (0.968 GB at ~58.8 GB/s ≈ 16.5 ms). Split-mode overhead ≈ 0 on Mac (class-sum ≈ span ≈ production). |

### 2026-09-08 — Mac decode sanity row with D7 variance fields, M2 Pro (P4-1)

First Mac q4g64 decode row (the P2-5 Mac row was bf16); same session and
settings as above, burst cap 640.

| Date | Device | Prompt (tokens) | Generated | Median GPU ms/tok | Median wall ms/tok | Median wall−GPU ms | Dispatches/tok | Window tok/s (128–512) | Latency p50/p95/p99/max ms (window) | Stalls | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-08 | Apple M2 Pro (Mac, dev machine) | decode-essay (84) | 640 (cap) | 34.74 | 35.09 | 0.360 | 591 | 28.32 (overall 28.23) | 35.18 / 38.36 / 38.73 / 38.93 | 0 (n=384) | PROVISIONAL, burst, warm, dev-loop sanity only. Tight distribution (max/p50 ≈ 1.11, zero stalls) — the D7 stall detector's clean-baseline shape. Median GPU 34.74 ms at 640-token depths vs 30.27 ms at depth ~146 in the attribution row: the depth-dependent attention/append cost, consistent measured twice. Output coherent (computing-history essay). |

### 2026-09-11 — Mac fused "after" rows (P4-4, PROVISIONAL): attribution breakdown + decode row

Fused (P4-2 SDPA + P4-3 folds) is the packed-pipeline DEFAULT as of P4-4;
same protocol, machine, and artifact (d03b3fe3…) as the 2026-09-08 "before"
rows above. Reproduce: the same two commands (fused is now the default; add
`--kernels naive` for the pre-fusion structure). These are dev-loop sanity
rows: the naive-vs-fused comparison here is CROSS-SESSION and therefore
context-only — the claim-grade before/after is P4-5's in-session interleaved
A/B under the D8 + bookend protocol, and Mac fractions do not predict device
fractions (Phase 2 precedent).

| Date | Device | matvec | attention | norm+elementwise | head/tail | Class-sum (median ms/tok) | Production GPU ms/tok @ dispatches | Sanity ratio | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-11 | Apple M2 Pro (Mac, dev machine) | 12.16 ms (45.1%) | 2.32 ms (8.6%) | 10.74 ms (39.8%) | 1.77 ms (6.6%) | 26.99 (span 27.11, wall 27.30) | 26.50 @ 227 | 1.02 (band 0.50–2.00) | PROVISIONAL, DIAGNOSTIC, kernels FUSED, depth 83–146. Production GPU 26.50 ms vs naive 30.27 ms at the same depth (−3.8 ms on Mac). Mac-only observation: norm+elementwise stays ≈39.8% of class-sum despite 11 → 3 elementwise dispatches/layer — the M2 Pro appears launch-latency-bound on the small fused dispatches, NOT byte-bound; no design decision from Mac fractions (the norm-fold revisit stays gated on P4-5's on-device attribution, per the P4-3 rationale). |

| Date | Device | Prompt (tokens) | Generated | Median GPU ms/tok | Median wall ms/tok | Median wall−GPU ms | Dispatches/tok | Window tok/s (128–512) | Latency p50/p95/p99/max ms (window) | Stalls | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-11 | Apple M2 Pro (Mac, dev machine) | decode-essay (84) | 640 (cap) | 31.76 | 32.05 | 0.290 | 227 | 31.16 (overall 31.14) | 32.16 / 35.14 / 35.60 / 35.68 | 0 (n=384) | PROVISIONAL, burst, warm, kernels FUSED (the new default), dev-loop sanity only. vs the 2026-09-08 naive row: window 31.16 vs 28.32 tok/s, median GPU 31.76 vs 34.74 ms, wall−GPU 0.290 vs 0.360 ms at 591→227 dispatches — directional Mac signal only (cross-session). Tight distribution (max/p50 ≈ 1.11), zero stalls. Output coherent (same computing-history essay species). |

### 2026-09-11 — iPhone 15 Pro Phase 4 rows (P4-5, James on-device): bookended interleaved naive-vs-fused + gates + attribution + sustained

Session conditions (one session, one build, run in the listed order):
DETACHED home-screen launches, Metal API validation OFF (recorded);
device iPhone16,1 (= the pinned iPhone 15 Pro hardware identifier),
iOS 26.6.1; weights q4g64 (artifact d03b3fe3…), residency mmap (P3-7
default); prompt decode-essay (84 tokens), burst cap 640, greedy;
battery 80% at session start (state-of-charge per the 2026-09-05
battery-fields correction; sustained ran 79→75%). Kernel path via the
P4-4 app toggle; every export carries its `kernels` field. Bookends =
F1/F4 (same fused config opens and closes the A/B block, phase-4.md D8
addendum). phys_footprint: Xcode gauge captured only PRE-LOAD (26.3 MB
at app launch, attached deploy step) — loaded values below are the
in-app cross-check (533–551 MB, consistent with P2-7's mmap 536 MB);
the loaded gauge-of-record reading was not captured this session.

| Run | Kernels | Cold/warm | Window tok/s (128–512) | Median GPU ms/tok | Median wall ms/tok | Wall−GPU ms | Dispatches/tok | Latency p50/p95/p99/max ms (window) | Stalls | phys_footprint (app) |
|---|---|---|---|---|---|---|---|---|---|---|
| F0 | fused | cold | 22.79 (overall 22.81) | 41.00 | 42.53 | 1.522 | 227 | 43.92 / 46.46 / 46.93 / 48.69 | 0 (n=384) | 534.5 MB |
| F1 (bookend) | fused | warm | 22.82 (22.80) | 41.02 | 42.52 | 1.519 | 227 | 43.93 / 46.46 / 46.87 / 47.18 | 0 | 550.7 MB |
| N1 | naive | warm | 20.77 (20.61) | 45.35 | 47.03 | 2.061 | 591 | 47.85 / 52.35 / 53.41 / 53.52 | 0 | 545.5 MB |
| F2 | fused | warm | 22.72 (22.70) | 41.23 | 42.72 | 1.517 | 227 | 44.05 / 46.79 / 47.11 / 47.78 | 0 | 544.5 MB |
| N2 | naive | warm | 20.68 (20.56) | 45.37 | 47.25 | 2.056 | 591 | 48.40 / 52.78 / 53.42 / 53.55 | 0 | 544.0 MB |
| F3 | fused | warm | 22.56 (22.56) | 41.37 | 42.79 | 1.502 | 227 | 44.08 / 47.13 / 48.25 / 49.91 | 0 | 543.2 MB |
| N3 | naive | warm | 20.40 (19.52) | 45.45 | 47.33 | 2.078 | 591 | 48.53 / 57.33 / 60.70 / 61.13 | 0 | 543.6 MB |
| F4 (bookend) | fused | warm | 22.27 (21.59) | 41.89 | 43.34 | 1.498 | 227 | 44.69 / 49.29 / 51.55 / 51.89 | 0 | 544.0 MB |

**Gate verdicts (pre-committed 2026-09-05, veto-closed 2026-09-07; hard
rule 6 — no constant touched):**

- **Dispatch gate ≤ 300: PASS** — 227 measured (DispatchCounter) on
  every fused row; naive rows 591, both perfectly stable.
- **Overhead gate ≤ 1.2 ms: FAIL** — fused median wall−GPU **1.51 ms**
  (per-run medians 1.498–1.519). Two dispatch counts in one session fix
  the overhead model: 2.06 ms @ 591 and 1.51 ms @ 227 ⇒ **≈1.17 ms
  FIXED per-token cost + ≈1.5 µs/dispatch** (affine fit; retro-predicts
  the historical 1.9–2.0 ms @ 591). The 1.2 gate assumed pure
  per-dispatch scaling (3.3 µs/dispatch, zero intercept) from the
  single 591-dispatch operating point. Recorded as a finding, not a
  gate adjustment — see DECISIONS.md 2026-09-11 P4-5 entry.
- **Decode floor ≥ 24.0 tok/s: FAIL** — fused warm-burst window median
  **22.64 tok/s** (range 22.27–22.82, n=4). Root cause per the
  attribution pair below: the fold set removed dispatch count but only
  ≈0.25 ms of elementwise GPU time (launch-latency-bound small
  kernels), so the "halve the ≈17 ms non-matvec slice" derivation
  over-credited the folds. See DECISIONS.md.
- **Before/after (D8 + bookend): DIRECTIONAL — fused faster.** Fused
  median 22.64 (22.27–22.82) vs naive median 20.68 (20.40–20.77):
  ranges disjoint AND effect 1.96 tok/s > bookend drift 0.55 tok/s
  (F1→F4) ⇒ **fused +9.5% in-session** (−4.5 ms/token wall). Naive
  median 20.68 also reproduces P3-7's cross-session 20.61 baseline.
- **Latency variance (D7, reported):** zero stalls in all 8 runs; fused
  max/p50 ≈ 1.07–1.16.

**On-device attribution (DIAGNOSTIC, D1 — never benchmark rows; cache
depth 83–146, 32 attributed + 32 production interleaved):**

| Kernels | matvec | attention | norm+elementwise | head/tail | Class-sum (median ms/tok) | Production GPU ms/tok @ dispatches | Sanity ratio |
|---|---|---|---|---|---|---|---|
| fused ("after") | 21.41 ms (57.2%) | 1.86 ms (5.0%) | 8.86 ms (23.7%) | 5.28 ms (14.1%) | 37.48 (span 37.56, wall 38.46) | 37.57 @ 227 | 1.00 |
| naive ("before") | 22.22 ms (57.8%) | 1.90 ms (4.9%) | 9.11 ms (23.7%) | 5.23 ms (13.6%) | 38.64 (span 38.74, wall 39.59) | 38.60 @ 591 | 1.00 |

Reading (feeds the P4-EXEC decode-vs-roofline judgment): weight
streaming (matvec + lm_head-dominated head/tail) ≈ 26.4 ms ⇒ ≈36.7 GB/s
≈ 84% of the 43.84 GB/s roofline (consistent with P3-6's 35.29 best);
**norm+elementwise 9.11 → 8.86 ms despite 11 → 3 dispatches/layer** —
the small elementwise kernels are launch/latency-bound on-device
(≈105 µs/dispatch), the same signature the Mac rows showed; the fused
SDPA advantage is depth-dependent (≈1.0 ms of the naive-vs-fused GPU
gap at depth ~115 here vs ≈4.2 ms at window depths in the burst rows).

**Sustained (fused, ≥5-min regenerate loop, decode-essay):** 4
generations / 5.0 min, battery 79→75%: gen windows **22.24 → 19.44 →
19.16 → 19.48 tok/s** (−12.6% first-gen thermal step, then stable;
gens 0–2 stop: eos at 1316 tokens, gen 3 truncated by the duration
bound). Last-gen per-token: median GPU 53.05 ms / wall−GPU 1.483 ms @
227 (medians over all tokens incl. depths ≫ window — not comparable to
burst window medians); window latency p50/p95/p99/max
51.20/53.86/54.28/54.85 ms, stalls 0. phys_footprint (app) 547.3 MB.

### 2026-09-12 — Mac P4-6 rows (PROVISIONAL): norm→matvec folds landed, 171 dispatches/token

P4-6 (iterate round): the two block RMSNorms folded into their consuming
matvecs (input-norm → matvec3, post-norm → gate+up+SwiGLU) with a
cooperative per-threadgroup inverse-RMS reduction — the P4-6 diagnosis
measured the standalone block-shape rmsnorm dispatch itself at ~176
µs/dispatch on Mac (redundant O(dim²) per-thread reduction; NOT launch
latency — dependent tiny dispatches cost ~3.5–4.7 µs, hazard tracking
free), so the fold removes the reduction redundancy rather than the
dispatch count per se. Same machine, protocol, artifact (d03b3fe3…), and
commands as the 2026-09-11 P4-4 rows (fused is the default). Cross-session
Mac comparisons are dev-loop directional signal only; the claim-grade gate
verdicts are P4-11's on-device.

| Date | Device | matvec | attention | norm+elementwise | head/tail | Class-sum (median ms/tok) | Production GPU ms/tok @ dispatches | Sanity ratio | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-12 | Apple M2 Pro (Mac, dev machine) | 16.70 ms (78.3%) | 2.33 ms (10.9%) | 0.52 ms (2.4%) | 1.78 ms (8.3%) | 21.38 (span 21.43, wall 21.64) | 21.32 @ 171 | 1.00 (band 0.50–2.00) | PROVISIONAL, DIAGNOSTIC, kernels FUSED (P4-6 folds), depth 83–146. vs P4-4 fused: production GPU 26.50 → 21.32 ms (−5.2 ms), norm+elementwise 10.74 → 0.52 ms (the class collapsed — only the cluster remains), matvec 12.16 → 16.70 ms (the folded matvecs carry the on-the-fly normed-input arithmetic; a threadgroup-cached normed-x variant is P4-10-scope). Norm dispatches now ride the matvec attribution class (spec D5: the erased boundary is no longer sliceable). |

| Date | Device | Prompt (tokens) | Generated | Median GPU ms/tok | Median wall ms/tok | Median wall−GPU ms | Dispatches/tok | Window tok/s (128–512) | Latency p50/p95/p99/max ms (window) | Stalls | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-12 | Apple M2 Pro (Mac, dev machine) | decode-essay (84) | 640 (cap) | 26.59 | 26.88 | 0.285 | 171 | 37.10 (overall 37.08) | 26.97 / 30.15 / 30.44 / 30.53 | 0 (n=384) | PROVISIONAL, burst, warm, kernels FUSED (P4-6 folds), dev-loop sanity only. vs the P4-4 fused row: window 37.10 vs 31.16 tok/s, median GPU 26.59 vs 31.76 ms at 227→171 dispatches — directional Mac signal only (cross-session). Zero stalls, tight distribution (max/p50 ≈ 1.13). Output coherent (same computing-history essay species). |

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

### 2026-09-05 — capacity-basis correction (battery fields were state-of-charge)

Correction, not an overwrite (METHODOLOGY rule 7; the 2026-08-22 rows above
stand as recorded). Established during the P3-7 session (James): the
device's battery HEALTH has read "Normal" / 100% max capacity the whole
project — the "battery health 85%" recorded on 2026-08-22 (and the battery
fields in all app reports to date) was actually the state of charge at
measurement time. The energy dry-run's capacity basis (12.6 Wh × 0.85 ⇒
387 J per 1% SoC) is therefore wrong; the correct basis is **12.6 Wh ×
1.00 ⇒ 453.6 J per 1% SoC** (×1.172 on every absolute energy figure):

| Figure (2026-08-22 rows) | As recorded | Corrected |
|---|---|---|
| MLX net J/token | 0.104 | **~0.122** |
| llama.cpp net J/token | 0.131 | **~0.154** |
| MLX gross / net W | 3.67 / 3.24 | ~4.30 / ~3.80 |
| llama.cpp gross / net W | 3.66 / 3.23 | ~4.29 / ~3.79 |
| Idle baseline | ~0.43 W | ~0.50 W |

Unchanged: the method VALIDATION (both cycles rescale identically and stay
inside the 3–9 W plausibility window), the relative result (MLX ~25% more
tokens/joule), all SoC bands, the ±12% quantization estimate (relative),
and every timing/bandwidth/decode/memory row (none uses capacity). Phase 6
obligation (for SPEC-P6): re-pin the capacity basis from battery health
read at run time, and record health and charge as separate fields.
