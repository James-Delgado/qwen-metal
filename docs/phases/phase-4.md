# Phase 4 Spec — fused attention + dispatch reduction

Written 2026-09-05 (SPEC-P4), from Phase 3 results logged in DECISIONS.md.
Context sources: PLAN.md v2 (phase table row 4, invariants 1/4/5), the
2026-08-20 eng review record Part 4 (SPEC-P4 obligation: the
dispatch-overhead target consumes the wall−GPU delta metric, OV#10) and
Part 1 Issue 1 (the KV split that deliberately deferred fused SDPA +
norm/RoPE folding + dispatch reduction to THIS phase), docs/phases/
phase-2.md (D5: one command buffer per token; the kernel set being fused)
and phase-3.md (D4/D7: the quant kernels and microbench this phase builds
on), DECISIONS.md 2026-09-04/05 entries (P3-5/P3-6/P3-7/P3-EXEC: the
measured Phase 4 inputs). Numeric gates for this phase are committed in
DECISIONS.md ("Phase 4 gates pre-committed", 2026-09-05) — this file
explains them; the DECISIONS entry is the binding record.

---

## Purpose

Phase 3 left the engine in a precisely measured state (all numbers from
DECISIONS.md 2026-09-05 P3-7/P3-EXEC):

- Decode warm-burst window median **20.61 tok/s** (range 20.47–20.88) —
  ~48.5 ms/token, ≈49% of the 43.84 GB/s roofline, 70% of the committed
  29.4 target.
- The same matvec kernels, weights-only, hit **35.29 GB/s (≈80% of
  roofline)** in the D7 microbench — one token's packed weights
  (967,753,728 B) stream in ≈27.4 ms at that rate.
- The gap is **≈17 ms/token of non-matvec time**: naive unfused attention
  over the fp16 cache (3 dispatches/layer, one-thread-per-output),
  standalone norm/RoPE/SwiGLU/residual elementwise kernels, and
  **591-dispatch overhead at 1.9–2.0 ms/token** (wall−GPU, stable across
  every session since P2-5).

Phase 4 attacks exactly that slice — the work deliberately deferred here
by the eng review's Issue 1 split: a fused GQA-correct SDPA kernel,
RMSNorm/RoPE folding into neighbors, and dispatch reduction, measured by
the wall−GPU delta (OV#10). **The end-to-end decode-vs-roofline judgment
lives in this phase** (PLAN phase table): after the fused rows land, the
29.4 tok/s success metric is judged against a measured per-stage
decomposition and the verdict — met, or the residual gap explained
component-by-component — is recorded in DECISIONS.md.

Weights, quantization, the packed schema, the KV cache layout, and the
oracle chain are all untouched: this is a kernel-structure phase, not a
format phase.

## Scope

**In:** per-stage GPU time attribution (measure before fusing — the
PLAN.md "do not guess at bottlenecks" rule made concrete), the fused SDPA
decode kernel, the elementwise folding set (QK-norm/RoPE/KV-append
cluster, residual folds, SwiGLU fold, norm folds and/or matvec
concatenation as needed to meet the dispatch gate), a selectable
naive-vs-fused kernel path for the on-device before/after row, per-token
latency-variance statistics, Tier-M/E + free-run re-verification on the
fused path, Mac PROVISIONAL rows, and the on-device Phase 4 rows
(James) including the decode-vs-roofline judgment inputs.

**Out (unchanged non-goals + deferred):** batched/tiled prefill GEMM
(Phase 5 — prefill continues to run sequentially per-token and inherits
the fused kernels 1:1), any KV-cache format change (stays fp16,
preallocated, head-major — a quantized KV cache would be new scope),
sampling beyond greedy, energy rounds (Phase 6), any change to the q4g64
schema or packing recipe, speculative decoding (post-Phase-6 stretch).
The CPU reference and CPU-quant oracle stay frozen — Phase 4 changes GPU
kernel structure only; every oracle diff continues to compare against the
same CPU-quant reference.

## Design decisions

### D1. Attribution before fusion (measure first)

The ≈17 ms non-matvec figure is a subtraction, not a breakdown. Before
any kernel is fused, the engine gains a **diagnostic attribution mode**
that reports per-kernel-class GPU time for a decode token at a recorded
cache depth: quant-matvec, attention (scores/softmax/PV — later the fused
SDPA), norm+elementwise (RMSNorm/RoPE/SwiGLU/residual/append), head/tail
(embedding, final norm, lm_head), plus the wall−GPU overhead already
measured. Constraints:

- Attribution numbers come from GPU timestamps (hard rule 7 dual timing
  recorded throughout); implementation may use per-class command-buffer
  splits or MTLCounterSampleBuffer stage sampling — task's choice — but
  the **production one-command-buffer-per-token path is never altered**;
  diagnostic mode is opt-in and never produces benchmark rows.
- The "before" attribution (naive kernels) is recorded on Mac
  (PROVISIONAL) during P4-1 and on-device (via the app's diagnostics
  export) during P4-5 — the on-device before/after breakdown is what the
  roofline decomposition (D6) consumes.
- A sanity test pins that class times sum to ≈ the whole-token GPU time
  (small gap tolerated for timestamp granularity; the test bounds it).

### D2. Fused GQA SDPA decode kernel

One dispatch per layer replaces the three-kernel
scores → softmax → PV chain: for the single query position, compute
QK^T over cache[0..p], softmax, and the PV product in one kernel using
**online (streaming) softmax in fp32** — running max + running
denominator + rescaled accumulator, so no scores/probs buffer is ever
materialized (the Phase 2 256 KB probs buffer disappears). GQA mapping
(16 query heads → 8 KV heads) lives inside the kernel. Parallelization
(threadgroup per head, simdgroup reductions over positions) is the
task's choice — but hard rule 3 binds: the kernel's correctness tests
pass BEFORE any optimization iteration, and every iteration re-passes
them. fp16 cache reads, fp16 output store, fp32 arithmetic throughout —
the same boundary semantics as the naive chain, so the Phase 2/3
divergence analysis (fp16 rounding + reduction order) still covers it.

The naive attention kernels are not deleted this phase (D4).

### D3. Folding set (RMSNorm/RoPE folding + dispatch reduction)

Current per-layer dispatch count is 21 (P2-4 structure, unchanged
through Phase 3): input norm, 3 QKV matvecs, 2 QK-norms, 2 RoPE, 2
KV-appends, 3 attention, o-proj, residual, post-norm, gate, up, SwiGLU,
down, residual (591 = 21×28 + embedding + final norm + lm_head).
Mandated fold outcomes (each preserving fp16-boundary/fp32-accumulate
semantics):

- **QK-norm/RoPE/append cluster:** the 6 post-QKV elementwise dispatches
  (q-norm, k-norm, rope-q, rope-k, append-K, append-V) consolidate to
  ≤2. Note the k-side cache write becomes a computed value rather than a
  copy — its exactness gate is replaced by the mapped tolerance (D5).
- **Residual adds:** eliminated as standalone dispatches (folded into
  the producing matvec's store).
- **SwiGLU:** eliminated as a standalone dispatch (folded into the
  down-matvec's load, or into a gate/up producer).
- **Attention:** 3 → 1 (D2).
- **Norm folds and/or matvec concatenation** (qkv as one dispatch over
  the concatenated 4096-row output; gate+up as one over 12288 rows —
  the quant matvec kernel already binds per-range triplet offsets) are
  the remaining levers, applied as D1's attribution says they pay, until
  the dispatch gate (D6) is met. Structural floor for reference: the
  full set lands ≈8/layer ⇒ ≈227/token; the mandated folds alone land
  ≈11/layer ⇒ ≈311/token — the ≤300 gate deliberately forces at least
  one consolidation beyond the minimum.

### D4. Kernel-path selection (before/after measurability)

The fused path becomes the **default**; the Phase 3 naive path stays
selectable behind an engine-level flag (model-init option) surfaced as a
CLI flag and an app toggle. Reason: the only honest before/after decode
comparison is naive-vs-fused **interleaved in one session on one build**
(D8) — cross-session comparisons are exactly what P3-7's measured
thermal drift invalidates. Both paths pass Tier-E in CI-shape tests
while both exist.

Scope clarification: the fused path targets the **packed (q4g64)
pipeline** — the performance path. The bf16 backend stays permanently on
the naive structure (it is the Phase 2 correctness artifact and keeps
its suites; the fold set would otherwise require bf16 variants of every
fused matvec for no benchmark benefit). Consequently the naive kernels
are never deleted — after the P4-5 before/after row lands, only the
toggle on the packed pipeline is up for removal (follow-up KP-1).

### D5. Correctness gates: constants reused, mapping rule for fused spans

No new tolerance constants. Every fused kernel is diffed against the
CPU-quant oracle computing the SAME span, at the **loosest Phase 2/3
constant among the modules the span absorbs** (equivalently: the
outermost species), floor 2⁻¹¹ throughout:

| Fused span contains | Gate (abs Δ vs CPU-quant oracle) |
|---|---|
| matvec / elementwise only (residual, SwiGLU folds) | Tier K: max(2⁻⁹·M, 2⁻¹¹) |
| a norm (norm+matvec, qk-norm/rope/append cluster) | max(2⁻⁸·M, 2⁻¹¹) |
| attention (fused SDPA) | max(2⁻⁷·M, 2⁻¹¹) |

Surfaces that remain pure copies/lookups stay EXACT (embedding gather
fp16 bitwise; v-side append if it remains a copy). Tier-M module-output
slices that survive fusion (attention out, MLP out, block out,
full-stack slices) keep their constants verbatim; internal boundaries
that fusion erases are no longer sliceable and their standalone tests
retire with the naive path — the fused-span kernel tests above replace
them at equal-or-outer species. Tier-E (teacher-forced 250-step logit
suite, fingerprints, top-64, tie-aware top-1 at ε_tie = 2⁻⁴·M64) reruns
verbatim on the fused path vs live CPU-quant. Free-running divergence
(128 steps × 5 prompts, fused vs CPU-quant) stays REPORTED, not gated —
the 2026-08-23 rationale stands. Naive-vs-fused full-pipeline outputs
are NOT required to match bitwise (reduction order differs); both gate
against the same oracle.

### D6. Performance gates (pre-committed; on-device, decided at P4-5)

Binding record and derivations: DECISIONS.md 2026-09-05 gates entry.
All are tripwire floors (the 2026-08-26 principle: gates catch bugs and
non-delivery; aspirations are reported), decided under the D8 protocol:

- **Dispatch gate:** dispatches/token ≤ **300**, MEASURED by
  DispatchCounter (not derived). ~2× reduction from 591; structurally
  demands the full D3 fold set plus at least one consolidation.
- **Overhead gate (OV#10):** median per-token wall−GPU delta ≤
  **1.2 ms** on-device (from measured 1.9–2.0 ms @ 591 ≈ 3.3 µs/
  dispatch; ≤300 dispatches ⇒ ~1.0 ms expected, 1.2 allows per-dispatch
  variance).
- **Decode floor:** warm-burst canonical-window median ≥ **24.0 tok/s**
  on-device (halving the measured ≈17 ms non-matvec slice ⇒ ~40 ms/token
  = 25.0 tok/s; floor set below at 24.0 — ~4% headroom vs the ~2%
  observed warm-burst spread).
- **The 29.4 tok/s success metric is JUDGED here, not gated:** the P4-5
  rows plus D1's on-device attribution feed a mandatory roofline
  decomposition recorded in DECISIONS.md at P4-EXEC — either the target
  is met, or every ms/token of the residual gap is attributed to a
  measured component (matvec stream rate, attention/KV time, remaining
  elementwise, dispatch overhead) with no unexplained slack. Rationale:
  29.4 is the project's absolute success metric (PLAN: "measuring,
  explaining, and narrowing the gap is" the requirement), already
  committed and unmovable; making it a phase gate would convert an
  aspiration into a tripwire, which the 2026-08-26 veto entry explicitly
  rejected as a gate philosophy.

### D7. Decode latency variance (PLAN exit criterion; reported, not gated)

Per-token wall-clock distribution over the canonical window: p50 / p95 /
p99 / max, plus a stall count (tokens > 2× window median — the
page-fault/preemption signature detector; P3-7 saw zero on the packed
working set). Computed engine-side from the existing TokenStepRecord
stream, exported in BenchmarkReport rows and the CLI per-token block.
Reported for every Phase 4 row, Mac and device. Not gated: variance on a
shared consumer device reflects OS scheduling as much as our code; the
number exists to be judged in the Phase 6 writeup.

### D8. Protocol addendum: bookend rule (from P3-7's session-scale drift)

P3-7 measured mode-independent session-scale thermal drift (first-gen
sustained windows 20.85 → 17.14 tok/s across ~35 min) that dominates
residency-scale effects. For ALL Phase 4+ device rows, extending the
Phase 3 D8 protocol (which stays in force: ≥3 same-session repeats,
median + range, interleaved A/B for directional claims, detached
launches with validation OFF recorded):

- Any session making a directional A-vs-B claim **starts and ends with
  the same configuration** (bookends). The bookend delta estimates
  within-session drift.
- A directional claim now requires BOTH non-overlapping interleaved
  ranges (existing rule) AND an effect size larger than the measured
  bookend drift; otherwise record "unresolved (drift-dominated)".
- The Phase 4 before/after row is naive-vs-fused via the D4 toggle,
  interleaved in one session on one build — never a cross-session
  comparison against the P3-7 rows (those remain the recorded "before"
  reference but the claim-grade comparison is in-session).

## Memory budget

Unchanged from Phase 3: packed weights ~0.97 GB (mmap, the P3-7 default)
+ KV 448 MiB + activations/logits < 10 MB ⇒ ≈1.5 GB. Fusion only
removes buffers (the materialized scores/probs buffers go with D2); no
new persistent allocations are permitted. The wired-copy diagnostic
toggle remains available but the Phase 4 rows run mmap per the P3-7
residency decision.

## Enumerated edge-case tests (land with the code, hard rule 3 / METHODOLOGY 3)

1. **Fused SDPA p=0:** single-position attention — online-softmax weight
   is exactly 1.0; output fp16 == the V row bitwise (the P2-3 exactness
   carries to the fused kernel).
2. **GQA mapping pattern test:** small-dims exact construction (P2-3
   headDim=4 precedent) proving each query head reads its correct KV
   head through the fused path.
3. **Cache-boundary indexing:** p at the last valid slot (4095) runs
   in-bounds (no OOB read past the preallocated cache); context-limit
   append still throws pre-dispatch with the cache untouched.
4. **Online-softmax adversarial orderings:** score max at first
   position, at last position, large-negative tails, and all-equal ties
   — fused output within the D5 attention gate vs the CPU-quant oracle
   (catches running-max rescale bugs, the classic online-softmax
   failure).
5. **Window-depth accumulation:** fused SDPA at p ≈ canonical-window
   depth vs oracle within gate (long-loop accumulator drift check).
6. **Fused QK-norm/RoPE/append cluster:** cache contents after the
   fused cluster vs fp16(CPU-quant fp32 k/v) within the norm-species
   gate; RoPE position indexing verified at positions {0, 1, large}
   (replaces the retired standalone-kernel tests at the mapped species;
   v-side stays exact if it remains a copy).
7. **Residual/SwiGLU folds:** fused-store outputs vs the unfused
   kernel-chain reference within Tier K on odd synthetic dims AND real
   dims (proves the fold changed structure, not arithmetic).
8. **Dispatch-count pins:** tiny-model exact-value tests pin the new
   per-layer/head-tail counts (P2-5 precedent); real-dims count MEASURED
   ≤ 300 asserted in the instrumentation test.
9. **Kernel-path toggle:** naive and fused paths both load, both pass a
   shared smoke + Tier-E-shape spot check; toggle plumbing (CLI flag,
   app control) selects the right encoder set (DispatchCounter
   distinguishes them).
10. **Edge behavior unchanged on the fused path:** empty prompt, >4K
    context, Metal-unavailable, missing-model errors identical to Phase
    3 behavior.
11. **Attribution sanity (D1):** class times are nonnegative, sum ≈
    whole-token GPU time within a pinned bound, wall ≥ GPU per class
    run; diagnostic mode leaves production timing untouched (same
    dispatch count, one command buffer).
12. **Variance statistics (D7):** p50/p95/p99/max + stall count computed
    correctly on hand-built TokenStepRecord sequences (incl. ties and
    the all-equal degenerate case).

## Instrumentation & benchmark deliverables

- Existing per-token dual timing + DispatchCounter ride unchanged; the
  dispatch count drops are measured, never derived (P2-5 rule).
- D1 attribution mode: CLI subcommand/flag + app diagnostics export
  (engine-side report text, exportText precedent) — "before" (naive) and
  "after" (fused) breakdowns, Mac PROVISIONAL + on-device.
- D7 variance stats in BenchmarkReport + CLI per-token block.
- Mac PROVISIONAL rows in benchmarks/results.md: naive-vs-fused decode
  sanity + attribution breakdown (never gated, Phase 2 precedent: Mac
  fractions do not predict device fractions).
- **On-device rows (James, P4-5, all under D8 + bookend protocol):**
  interleaved naive-vs-fused warm-burst decode row (the before/after
  claim), fused sustained row, overhead row (wall−GPU @ new dispatch
  count), attribution breakdown export, latency-variance fields, decode
  gate + overhead gate + dispatch gate verdicts.
- DECISIONS.md entries: gate outcomes, the before/after result, and at
  P4-EXEC the **decode-vs-roofline judgment** (D6) with the full
  decomposition.

## Task breakdown (seeded in docs/PRIORITIES.yaml)

| Task | Deliverable | Depends on |
|---|---|---|
| P4-1 | Attribution harness (per-class GPU time, diagnostic mode) + latency-variance stats + CLI/app surfacing + Mac "before" breakdown row | — |
| P4-2 | Fused GQA SDPA kernel (online softmax) + edge tests 1–5 + pipeline wiring behind the D4 toggle + affected Tier suites re-passed | P4-1 |
| P4-3 | Folding set (QK-norm/RoPE/append cluster, residual/SwiGLU folds, norm folds and/or matvec concatenation) to ≤300 dispatches + edge tests 6–8 + Tier suites re-passed | P4-2 |
| P4-4 | Fused-path default + full Tier-M/E re-verification + free-run report + Mac "after" rows + app toggle/diagnostics finalization | P4-3 |
| P4-5 (james) | On-device rows: decode floor / overhead / dispatch gates, interleaved naive-vs-fused before/after with bookends, sustained row, attribution + variance exports | P4-4 |

## Exit criteria (PLAN.md phase table, walked)

- Fused GQA-correct SDPA kernel ✓ = P4-2 lands with its layered tests at
  the reused attention constant, re-passed through every optimization
  iteration.
- RMSNorm/RoPE folding ✓ = P4-3's fold set (cluster consolidation +
  standalone RoPE/SwiGLU/residual dispatches eliminated).
- Dispatches-per-token reduced, measured via the wall−GPU delta ✓ =
  dispatch gate ≤300 (DispatchCounter) + overhead gate ≤1.2 ms/token
  on-device (P4-5).
- End-to-end decode tok/s vs roofline judged here ✓ = decode floor
  ≥24.0 tok/s + the D6 judgment (29.4 met, or the gap decomposed against
  the D1 on-device attribution) recorded in DECISIONS.md at P4-EXEC.
- Decode latency variance measured ✓ = D7 stats on every Phase 4 row.
- DECISIONS.md entries for every gate outcome, the before/after result,
  the judgment, and anything else decided/measured (standing
  discipline).
