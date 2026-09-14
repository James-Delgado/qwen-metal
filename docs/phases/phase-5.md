# Phase 5 Spec — tiled prefill GEMM

Written 2026-09-14 (SPEC-P5), from Phase 4 results logged in DECISIONS.md.
Context sources: PLAN.md v2 (phase table row 5: "Prefill tok/s benchmarked
separately vs MLX; uses threadgroup memory + simdgroup_matrix"; invariants
1/2/4/5), docs/phases/phase-2.md (the "sequential per-token prefill
(batched GEMM stays Phase 5)" deferral), docs/phases/phase-4.md (D5
constant-mapping rule, D8 + bookend protocol — both carried forward),
DECISIONS.md 2026-09-14 entries (P4-11 / P4-EXEC: the measured Phase 5
inputs) and 2026-09-07 (north-star binding: SPEC-P5 quantifies remaining
headroom per component, not just the head-to-head). Numeric gates for this
phase are committed in DECISIONS.md ("Phase 5 gates pre-committed",
2026-09-14) — this file explains them; the DECISIONS entry is the binding
record.

---

## Purpose

Phase 4 left decode in a precisely measured state (DECISIONS.md
2026-09-14 P4-11/P4-EXEC): fused warm-burst window median **31.67 tok/s**
on-device, ≈30.1–30.3 ms/token GPU with weight streaming ≈27.1–27.7 ms
(≈95% of fused GPU time) — the decode engine is nearly pure
bandwidth-bound and the 29.4 target is exceeded.

Prefill is untouched since Phase 2: the prompt is processed
**sequentially, one token at a time**, through the same per-token
kernels. Every prompt position therefore re-streams the full packed
weights (967,753,728 B), which puts a hard structural ceiling on any
sequential prefill: at 100% of the measured 43.84 GB/s roofline,
0.968 GB/token ⇒ ≈22.1 ms/token ⇒ **≈45.3 tok/s — the best sequential
prefill can EVER do on this device**. The measured baselines make the
gap concrete:

- Our only on-device prefill row: **8.23 tok/s** (Phase 2, bf16 naive,
  prefill-summarize 852 tokens in 103.5 s). The packed/fused sequential
  prefill has never been rowed on-device (decode-essay sessions since);
  Mac PROVISIONAL packed rows showed 23.8–28.6 tok/s, and per-token cost
  ≈ decode's suggests ≈30 tok/s on-device today — an estimate, not a row
  (P5-1/P5-5 measure it).
- MLX prefill on the same device: **≈370 tok/s** (Phase 0 PROVISIONAL);
  llama.cpp ≈452 tok/s. Both are ≈an order of magnitude above the
  sequential structural ceiling — they batch.

Phase 5 removes the structural ceiling: process the prompt in batched
chunks so the weights are streamed **once per chunk instead of once per
token**, with the matmuls becoming real GEMMs (M prompt positions × the
weight matrix) implemented as tiled kernels using threadgroup memory +
simdgroup_matrix (the PLAN phase-table pin). Decode is not touched: the
Phase 4 fused decode path, its gates, and all its rows stand.

Per the 2026-09-07 north-star binding, this phase also produces the
project's first **measured compute-throughput denominator** (the GEMM
microbench's GFLOPS curve): batched prefill transitions from
bandwidth-bound at small M to compute-bound at large M, and the Phase 6
roofline analysis needs the measured crossover, not an assumed peak-FLOPs
figure (METHODOLOGY 4: if a spec needs an unmeasured number, measuring it
becomes a task).

## Scope

**In:** prefill-span instrumentation (dual-timed, metric-of-record
definition below), the tiled q4g64 dequant-GEMM kernel + its layered
correctness suite, the chunked batched prefill pipeline (batched
norm/RoPE/KV-append, causal attention over the chunk, last-position-only
lm_head), a sequential-vs-tiled prefill path toggle for the in-session
before/after row, an M-sweep GEMM microbench (bandwidth fraction gate at
M=8; GFLOPS curve reported), Tier-M/E + free-run re-verification with the
tiled path engaged, Mac PROVISIONAL rows, and the on-device Phase 5 rows
(James) with the prefill-vs-MLX judgment inputs.

**Out (unchanged non-goals + deferred):** any change to the decode path
or its kernels (regression re-walk only), batch size > 1 across requests
(prefill batching is within one prompt — the batching non-goal is about
multi-request serving and stands), pipelined decode (PIPE-1 — campaign,
blocked on SPEC-P7 per the PD-1 bindings), KV-cache format changes,
q4g64 schema/recipe changes, sampling beyond greedy, energy rounds
(Phase 6), speculative decoding. The CPU reference and CPU-quant oracle
stay frozen; the bf16 backend keeps sequential prefill permanently (the
Phase 2 correctness artifact, the P4 D4 precedent).

## Design decisions

### D1. Prefill metric of record + instrumentation (measure first)

The engine gains a dual-timed **prefill span**: from the start of
prompt processing to the point the last prompt position's output (the
logits/argmax feeding the first generated token) is available —
**excluding the first generated token's decode forward**. Prefill tok/s
of record = per-engine prompt token count ÷ prefill-span wall time; the
span's GPU time is recorded alongside (hard rule 7 — dual timing
everywhere), plus prefill dispatch count (DispatchCounter; reported,
never gated). The existing BenchGenerationRunner TTFT-style field (which
spans prefill + one decode forward, honestly labeled since P2-6) keeps
exporting for continuity; rows cite the new span. Rationale: at tiled
rates the one-decode-forward contamination (~31 ms inside a multi-second
span today) shrinks to ~1% but the metric should be exact and
engine-owned, on both paths, before any comparison row exists. The
sequential "before" Mac row (PROVISIONAL) lands with the
instrumentation; the on-device "before" is measured in-session at P5-5.

### D2. Chunked batched prefill architecture

The prompt is processed in fixed-size chunks of C positions (last chunk
ragged). Per chunk, per layer: batched projections (GEMM: C×2048 input
against each weight matrix), batched QK-norm + RoPE + KV-append over the
C positions, causal attention for the C query positions over
cache[0..chunkEnd] (masking within the chunk), batched MLP. After the
final chunk, final-norm + lm_head run for the **last position only** —
sequential prefill pays the ≈5.3 ms lm_head stream per prompt token;
batched prefill pays it once per prompt. Decode from the first generated
token onward is the unchanged Phase 4 fused path; the KV cache written
by prefill is the same preallocated buffer (hard rule 4 — no growth, no
new cache).

**C is a reported parameter, not a pin:** the task picks it by
measurement (the microbench's M-sweep + end-to-end timing), records it
on every row, and may expose it as a diagnostic option. Reference point:
mlx-lm's ppl harness used chunk 512 (DECISIONS.md 2026-08-30). All
prefill scratch (chunk activations, attention workspace) is preallocated
at model load, sized for the maximum chunk — no per-chunk allocation.
Budget: ≤ 64 MiB on top of the standing ≈1.5 GB (C=512 activations incl.
the 512×6144 gate/up pair ≈ 13 MiB fp16; no materialized full C×L score
matrix is permitted — attention workspace must be bounded, streamed or
tiled).

### D3. Tiled q4g64 dequant-GEMM kernel

The phase's core kernel: C×K activations (fp16) × q4g64 packed weights
(N×K) → C×N fp16 out, fp32 accumulation, using **threadgroup memory
tiles + simdgroup_matrix accumulation** (the PLAN pin). Dequantization
stays inside the consuming kernel per hard rule 1, with one recorded
clarification (veto-flagged in the gates entry): **staging dequantized
weight tiles in threadgroup (on-chip) memory inside the consuming GEMM
kernel is permitted** — the tile is transient scratch that dies with the
threadgroup, which is the standard tiled-GEMM pattern (MLX's quantized
Steel kernels do the same). What stays forbidden is what the invariant
was written against: materializing dequantized weights to a device/DRAM
buffer. Register-only dequant (each simdgroup dequantizing its own
fragment) is equally acceptable; the task chooses by measurement.

Hard rule 3 binds: the kernel's correctness suite (edge tests 1–2, vs
the CPU-quant oracle through the validated sgemm reference, hard rule 8)
passes BEFORE any optimization iteration, and every iteration re-passes
it. The q4g64 schema, packing recipe, and artifact are untouched.

### D4. Prefill attention strategy (reuse first, batch if it pays)

Within a chunk, position i attends cache[0..i] (causal). Two
implementations are in scope, chosen by measurement:

1. **Reuse the P4-7 split-K fused SDPA per position** (a loop of C
   dispatches per layer): zero new attention code; encode cost ≈1.22
   µs/dispatch (measured A17 Pro) ⇒ 852 positions × 28 layers ≈ 29 ms of
   one-time encode per prefill — acceptable at first landing.
2. **A batched causal SDPA kernel** (C query positions per dispatch) —
   taken only if the reuse path's measured share says it pays
   (attribution over the prefill span; the PLAN "do not guess at
   bottlenecks" rule).

Attention FLOPs at 852 tokens are small next to the projections/MLP
GEMMs; the projections are the phase's target, and D1's measurements
arbitrate. Whichever lands gates at the attention species constant (D6).

### D5. Path selection (before/after measurability)

Tiled prefill becomes the **default for the packed pipeline**;
sequential prefill stays selectable behind an engine-level option
surfaced as a CLI flag and app toggle (the P4 D4 pattern, same
rationale): the only honest before/after prefill comparison is
sequential-vs-tiled **interleaved in one session on one build** under
the D8 + bookend protocol. The bf16 backend keeps sequential prefill
permanently. Decode is unaffected by the toggle. Toggle removal is a
post-row follow-up decision, not part of this phase.

### D6. Correctness gates: constants reused, species mapping extended

**No new tolerance constants** (the P4 D5 rule extended to batched
spans, floor 2⁻¹¹ throughout; binding record in the gates entry):

| Batched span contains | Gate (abs Δ vs CPU-quant oracle) |
|---|---|
| GEMM / elementwise only | Tier K: max(2⁻⁹·M, 2⁻¹¹) |
| a norm (batched norm+GEMM, qk-norm/rope/append) | max(2⁻⁸·M, 2⁻¹¹) |
| attention (causal SDPA, either D4 form) | max(2⁻⁷·M, 2⁻¹¹) |

Pure copies/lookups stay EXACT (batched embedding gather bitwise;
v-side append while it remains a copy). **KV-cache contents after
batched prefill** gate against fp16(CPU-quant fp32 K/V) at the
norm-species constant for every prompt position (k-side; v-side exact
if a copy) — the P4 cluster precedent. Tier-M module slices keep their
constants where sliceable; Tier-E (teacher-forced 250-step logit suite,
fingerprints, top-64, tie-aware top-1 at ε_tie = 2⁻⁴·M64) reruns
verbatim with the tiled prefill path engaged. Free-running divergence
(128 steps × 5 prompts from tiled prefill) stays REPORTED, not gated.
Tiled-vs-sequential outputs are NOT required to match bitwise (GEMM
reduction order differs from matvec — the 2026-09-12 reduction-order
decision covers this); both gate against the same CPU-quant oracle.

### D7. Performance gates (pre-committed; binding record in DECISIONS.md)

All tripwire floors (gates catch bugs and non-delivery; aspirations are
reported — the 2026-08-26 principle):

- **GEMM microbench fraction gate (on-device, P5-5):** the tiled GEMM
  weight-sweep at **M=8** over all 197 packed matrices (the P3-6
  protocol shape) sustains effective weight-stream ≥ **0.70 × 43.84 =
  30.69 GB/s** — the Phase 3 D7 fraction applied to the new kernel at a
  batch size where it remains weight-bandwidth-dominated. Batching must
  not lose bandwidth the matvec already achieves (35.29 GB/s measured).
  The sweep additionally REPORTS GB/s + GFLOPS at M ∈ {8, 64, 512}
  minimum — the measured compute denominator for the roofline analysis
  (never gated).
- **Prefill floor (on-device, P5-5):** warm tiled prefill of the pinned
  prefill-summarize prompt (852 HF tokens), D1 span, median of ≥3
  repeats ≥ **135 tok/s** — ≈3× the 45.3 tok/s sequential structural
  ceiling (amended from the proposed 90 = 2× at the veto walk,
  TIGHTENED by James before any Phase 5 test existed; binding record:
  DECISIONS.md 2026-09-14 "Phase 5 veto window CLOSED"). A tiled path
  landing below 3× the ceiling more likely signals a half-engaged
  pipeline than a hardware limit. Still deliberately far below the
  ≈370 MLX aspiration, which is judged, not gated.
- **Decode regression floor (on-device, P5-5):** fused decode warm-burst
  window median ≥ **24.0 tok/s** in the same session (the committed
  Phase 4 constant reused as a regression tripwire — decode measured
  31.67 at P4-11; falling below 24.0 after prefill integration signals
  breakage, not noise).
- **Prefill-vs-MLX is JUDGED, not gated:** P5-EXEC records the measured
  prefill rows against MLX's Phase 0 PROVISIONAL ≈370 tok/s (staleness
  rule: the publishable head-to-head is Phase 6, same-session), with
  **remaining headroom quantified per component** (weight stream at the
  microbench rate, measured GFLOPS vs the M-sweep curve, attention
  share, elementwise share, per-chunk overhead) — the 2026-09-07
  north-star binding and the PLAN's "measuring, explaining, narrowing"
  framing.

### D8. Device protocol (carried verbatim)

The Phase 3 D8 pins + Phase 4 bookend rule apply to all Phase 5 device
rows unchanged: detached launches, validation OFF recorded, ≥3
same-session repeats with median + range, interleaved A/B with bookends
for the sequential-vs-tiled before/after claim (effect > bookend drift
or "unresolved (drift-dominated)"), mmap residency, decode-essay for the
decode regression rows and prefill-summarize for the prefill rows,
phys_footprint gauge-of-record, SoC/battery fields per the 2026-09-05
correction. Prefill thermal note: Phase 0 observed prefill throttling
harder than decode (compute-bound); repeats bound it — reported, not
gated.

## Memory budget

Packed weights ~0.97 GB (mmap) + KV 448 MiB + activations/logits
< 10 MB + **prefill scratch ≤ 64 MiB (preallocated at load, D2)** ⇒
≈1.5 GB unchanged at the headline level. No other new persistent
allocations.

## Enumerated edge-case tests (land with the code, hard rule 3 / METHODOLOGY 3)

1. **Tiled GEMM vs oracle:** odd synthetic shapes + every real weight
   shape, ragged M (M % tile ≠ 0), M=1 degenerate — Tier K vs the
   CPU-quant dequant × validated sgemm reference (hard rule 8).
2. **Packed-layout adversarial reads through the GEMM path:** group
   boundaries crossing K-tiles, nibble order, the P3-1 adversarial
   fixtures (degenerate/extreme/negative-heavy groups) — same gates.
3. **Causal masking exactness:** small-dims constructed K/V (P2-3
   headDim=4 precedent) proving position i never reads j > i within a
   chunk.
4. **Chunk-boundary continuity:** prompt split across ≥2 chunks —
   RoPE/position indexing at {0, 1, C−1, C, last}; chunk-2 attention
   reads chunk-1 cache correctly; outputs within species gates.
5. **KV cache after batched prefill:** all prompt positions vs
   fp16(CPU-quant fp32) within the norm-species gate (k-side); v-side
   exact while a copy.
6. **Ragged/short prompts:** prompt % C ≠ 0, prompt < C (single partial
   chunk), prompt = 1 token.
7. **Last-position-only lm_head:** logits computed exactly once per
   prefill; tiny-model dispatch pins for the prefill path (P2-5
   precedent; real-dims counts REPORTED).
8. **Decode handoff:** teacher-forced continuation after tiled prefill
   gates vs oracle; free-run report (128 × 5) from tiled prefill.
9. **Prefill-path toggle:** sequential and tiled both load and pass a
   shared smoke + Tier-E-shape spot check; DispatchCounter distinguishes
   the paths; bf16 backend rejects/ignores the tiled option cleanly.
10. **Edge behavior unchanged:** empty prompt, >4K prompt, prompt at the
    4096 boundary, Metal-unavailable, missing model — errors identical
    to Phase 4 behavior on both prefill paths.
11. **Scratch preallocation:** prefill buffers allocated at model load;
    a multi-chunk prefill performs no new buffer allocations (asserted
    via buffer identity; footprint cross-check stable).
12. **Prefill instrumentation sanity (D1):** span wall ≥ span GPU;
    token accounting equals the per-engine count; both paths report the
    same fields; the legacy TTFT-style field still exports.
13. **Microbench harness:** fraction computation verified on synthetic
    timings; M-sweep report shape pinned.

## Instrumentation & benchmark deliverables

- D1 prefill span (dual-timed) + prefill dispatch count in
  BenchmarkReport, CLI output, and the app export; sequential path
  reports identically.
- M-sweep microbench: CLI subcommand (P3-6 `bandwidth`/microbench
  precedent) printing GB/s + GFLOPS per M; Mac PROVISIONAL row + device
  row (James).
- Mac PROVISIONAL rows: sequential "before" prefill (P5-1), tiled
  "after" prefill + microbench sanity (P5-2..P5-4). Never gated — Mac
  fractions do not predict device fractions (standing precedent).
- **On-device rows (James, P5-5, D8 + bookend):** interleaved
  sequential-vs-tiled prefill before/after (the claim row), ≥3 warm
  tiled prefill repeats (floor gate), M=8 microbench gate row + M-sweep
  report, fused decode regression row (≥24.0 re-walk), phys_footprint,
  thermal/variance notes.
- DECISIONS.md entries per task: gate outcomes, the before/after result,
  chunk-size choice rationale, and at P5-EXEC the **prefill-vs-MLX
  judgment** with the per-component headroom decomposition (D7).

## Task breakdown (seeded in docs/PRIORITIES.yaml)

| Task | Deliverable | Depends on |
|---|---|---|
| P5-1 | Prefill-span instrumentation (metric of record, dual-timed, dispatch count) + CLI/app surfacing + Mac sequential "before" prefill row | — |
| P5-2 | Tiled q4g64 dequant-GEMM kernel (threadgroup + simdgroup_matrix) + edge tests 1–2 + M-sweep microbench harness + Mac sanity row | — |
| P5-3 | Chunked batched prefill pipeline (batched elementwise/norm/RoPE/append, causal attention per D4, last-position lm_head, preallocated scratch) + edge tests 3–8, 10–11 | P5-2 |
| P5-4 | Tiled prefill as packed-pipeline default + sequential toggle + Tier-M/E re-verification + free-run report + Mac "after" rows + edge tests 9, 12–13 close-out | P5-1, P5-3 |
| P5-5 (james) | On-device rows: prefill floor + M=8 microbench + decode regression gates, interleaved sequential-vs-tiled before/after with bookends, M-sweep report, exports | P5-4 |

## Exit criteria (PLAN.md phase table, walked)

- Tiled prefill GEMM using threadgroup memory + simdgroup_matrix ✓ =
  P5-2/P5-3 land with the layered correctness suite (tests 1–8), gates
  held at the reused constants, re-passed through every optimization
  iteration.
- Prefill tok/s benchmarked separately vs MLX ✓ = P5-5's device prefill
  rows (D1 metric) + the P5-EXEC judgment vs the Phase 0 PROVISIONAL MLX
  row, per-component headroom quantified (D7; staleness rule points the
  final head-to-head at Phase 6).
- Pre-committed gates walked ✓ = microbench fraction (≥30.69 GB/s @
  M=8), prefill floor (≥135 tok/s, as amended at the veto close),
  decode regression (≥24.0 tok/s), Tier-M/E + KV-contents correctness
  suites on the tiled path.
- DECISIONS.md entries for every gate outcome, the before/after result,
  the judgment, and anything else decided/measured (standing
  discipline).
