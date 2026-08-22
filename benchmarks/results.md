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
