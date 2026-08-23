# Phase 2 Spec — Naive Metal port + minimal KV cache, on-device

Written 2026-08-23 (SPEC-P2), from Phase 1 results logged in DECISIONS.md.
Context sources: PLAN.md v2 (exit-criteria table, invariants), the 2026-08-20 eng
review record Part 4 (SPEC-P2 obligations: OV#5/#9/#11/#12), phase-0-1.md (the
fixture/oracle machinery this phase reuses), DECISIONS.md 2026-08-22/23 entries
(measured bandwidth 43.84 GB/s, MLX 39.2 tok/s, target 29.4 tok/s, EOS-1 stop-set
note). Numeric gates for this phase are committed in DECISIONS.md
("Phase 2 gates pre-committed", 2026-08-23) — this file explains them; the
DECISIONS entry is the binding record.

---

## Purpose

Port the validated CPU reference forward pass to Metal kernels, naively, and run
it on the pinned iPhone. Two things become true for the first time:

1. **Decode is incremental.** A preallocated KV cache (hard rule 4) with per-step
   append and a naive unfused attention kernel over the cache — so the on-device
   build is memory-bound like every competitor engine, and the "before" benchmark
   row is meaningful (eng review Issue 1).
2. **The engine runs on the phone.** A thin iOS shell (`QwenMetalApp/`, planned in
   CLAUDE.md) wraps the engine for James's device runs.

Everything is deliberately naive: correctness and instrumentation are the
deliverables. Kernel speed is Phases 3–5's job; per hard rule 3, no optimization
touches any kernel before its correctness test passes.

## Scope

**In:** GPU weight residency (mmap no-copy + wired-copy variant), the naive
decode kernel set, preallocated KV cache + naive attention, GPU pipeline wiring +
the fp16 gate suite, per-token dual timing + dispatch counting, engine-owned stop
set (EOS-1 follow-up), CLI `--backend gpu`, the iOS shell, the on-device "before"
row, and the mmap-vs-wired sustained-stability comparison.

**Out (unchanged non-goals + deferred):** quantization and packed layouts
(Phase 3), fused/optimized kernels and dispatch reduction (Phase 4), batched
prefill GEMM (Phase 5), sampling beyond greedy, any KV-cache management beyond
append-until-full, energy rounds (Phase 6), simulator support. The CPU reference
is frozen except where a task below explicitly says otherwise — it is the oracle.

## Design decisions

### D1. Weights on GPU: raw bf16 checkpoint bits, one mmap-backed no-copy buffer

The consolidated checkpoint (`models/qwen3-1.7b-70d244cc.safetensors`, bf16) is
mmapped (invariant 5) and the whole mapping is wrapped in a single
`MTLBuffer` via `makeBuffer(bytesNoCopy:)` (mmap base is page-aligned; length
rounded up to page size). Kernels receive per-tensor **byte offsets** from the
existing `SafetensorsFile` index — no per-tensor buffers, no copies, no
conversion at load.

Kernels read weights as `ushort` and upcast bf16→fp32 in registers with an
integer shift + bitcast (`as_type<float>(uint(w) << 16)`) — exact by
construction, identical to the CPU reference's upcast, and free of any MSL
`bfloat` type dependency. Consequences:

- Weight bits are **identical** on both sides of every diff test. No gate below
  carries a weight-rounding term; all fp16 divergence comes from activation
  rounding and reduction order.
- The **wired-copy residency variant** (OV#9) is the same bytes `memcpy`'d into a
  dirty heap-backed `MTLBuffer` — it must produce bit-identical outputs (an exact
  `==` test), so the mmap-vs-wired benchmark isolates residency behavior only.
- No dequantized weight tensor is ever materialized (hard rule 1 — trivially
  satisfied in this phase; the upcast lives in registers).

### D2. Precision policy

| Surface | Format |
|---|---|
| Weights (DRAM) | bf16, raw checkpoint bits (D1) |
| Activations between kernels | fp16 |
| Accumulation inside kernels | fp32, always |
| Softmax (max-subtract, exp, normalize) | fp32; scores buffer fp32 |
| Logits (lm_head output) | fp32 (also what the gate suite and argmax consume) |
| KV cache | fp16 |

This is the "fp16 port" of the PLAN.md phase table: the residual stream is
rounded to fp16 at module boundaries; arithmetic never accumulates in fp16.

### D3. KV cache layout (hard rule 4: preallocated at load)

One buffer, allocated at model load, never grown:
`[layers=28][K|V][kv_heads=8][max_ctx=4096][head_dim=128]` fp16, head-major so
each head's positions are contiguous for the attention kernel's streaming read.
Total = 28·2·8·4096·128·2 B = **448 MiB**. Append at position `p` writes 8×128
fp16 values per layer per K/V (scattered by head — negligible at decode rates).
Position `p == 4096` is a clean stop (context-limit error), never an OOB write.

### D4. Kernel inventory (all naive, one-thread-per-output-element species)

embedding-lookup (gather + bf16 upcast + fp16 store), rmsnorm (parameterized:
input/post-attention/final + per-head QK-norm), matvec (bf16 weights × fp16
input, fp32 accumulate — serves QKV/o/gate/up/down/lm_head; lm_head reads the
tied embedding matrix transposed, computed without materializing a transpose),
rope (half-split rotation, fp32 angles matching the CPU table), kv-append,
attn-scores (q·K[0..p]ᵀ/√d per (kv-head, position), GQA: each KV head serves 2 Q
heads), softmax (fp32, max-subtracted), attn-pv (scores·V[0..p]), swiglu
(silu(gate)·up elementwise), residual-add, and the final argmax stays **CPU-side**
(fp32 logits read back — 608 KB/token on unified memory; reuses `DecodeLoop`'s
first-index tie-break so GPU and CPU decode share one argmax implementation).

Kernels compile from source strings at runtime, following the established
`Metal/` module pattern (MetalContext + P0B kernels).

### D5. Dispatch model and timing

One command buffer per decoded token: encode all ~500 dispatches (≈18/layer × 28
+ head/tail), commit, wait. Per-token record: GPU timestamps
(`gpuStartTime`/`gpuEndTime`), wall clock around commit→completion, and the
**dispatch count** — wall−GPU is the dispatch-overhead metric Phase 4 consumes
(hard rule 7 / OV#10). No `MTLHeap`s, no argument buffers, no encoder tricks:
that is Phase 4 material, and the naive number is the point of the "before" row.

### D6. Prefill: sequential

Prompt tokens feed through the decode path one at a time (KV append per token;
logits computed only at the last prompt position). Prefill tok/s will be near
decode tok/s — recorded honestly in the "before" row; batched GEMM prefill is
Phase 5 by design (eng review Issue 1 split).

### D7. Stop-set assembly moves into the engine (EOS-1 follow-up)

The stop set (config.json ∪ tokenizer ∪ generation_config.json, today
{151645, 151643}) currently assembles in the CLI. Phase 2 gives it an engine
home — `ModelDirectory` grows a loader that returns the resolved stop set
alongside config/tokenizer/generation-config — so CLI and iOS app consume one
implementation and free-running stop behavior is engine-defined (it feeds the
argmax-agreement machinery below). CLI behavior is unchanged; existing EOS-1
tests move/extend with the code.

### D8. iOS shell: `QwenMetalApp/`

New top-level thin SwiftUI target (planned in CLAUDE.md; engine logic stays in
the package — none in the app). Screens: generate (prompt → text, backend
fixed to GPU) and benchmark (runs the pinned protocol: burst decode-essay,
sustained 5-min regenerate loop, residency-mode toggle mmap/wired; displays and
exports the row fields incl. per-token timing summary and dispatch count).
Increased Memory Limit entitlement. Agents build it; James signs, deploys, and
runs — never agents (standing rule).

## Memory budget (Phase 2 is the project's high-water mark)

Weights bf16 3.44 GB (file-backed under mmap) + KV 448 MiB + activations/logits
< 10 MB + app overhead ⇒ ≈ 4.0 GB. Inside the iPhone 15 Pro (8 GB) Increased
Memory Limit envelope, but with the least headroom of any phase — Phase 3 drops
weights to ~0.97 GB. This is exactly the regime where mmap eviction/bimodality
(OV#9) is most likely to appear, which is what the residency comparison is for.
phys_footprint is the only memory metric (protocol pin); expect the mmap mode to
under-report weights relative to wired mode — annotate rows accordingly (the
P0A-1 llama.cpp 307 MB artifact, now on our side of the table).

## Correctness gates (committed in DECISIONS.md — summary here)

All diffs are GPU-vs-CPU-reference (or vs the P1 fixtures, which the CPU
reference matches at ≤1e-3). M = max|ref| over the compared slice; per-step
M64 = max|ref top-64 value| at that step. u16 = 2⁻¹¹ (fp16 unit roundoff).
Full derivations live in the DECISIONS entry; gates never loosen (hard rule 6).

| Tier | Surface | Gate |
|---|---|---|
| K (kernel) | every kernel, synthetic unit-scale inputs, before any optimization (hard rule 3) | abs Δ ≤ max(2⁻⁹·M, 2⁻¹¹) |
| K (exact) | embedding-lookup; kv-append; residency-mode B vs A; bf16 upcast paths | exact (== / bit-identical) |
| M (module) | layer0_pre_attn_norm_output and block-internal slices, isolated (fed reference inputs) | abs Δ ≤ max(2⁻⁸·M, 2⁻¹¹) |
| M (module) | layer0_attn_output vs fixture slice, isolated | abs Δ ≤ max(2⁻⁷·M, 2⁻¹¹) |
| M (exact) | embeddings_output vs fixture | GPU fp16 == fp16(ref fp32) bitwise |
| E (end-to-end) | last_layer_output, final_norm_output (full 28-layer stack) | abs Δ ≤ max(2⁻⁵·M, 2⁻¹¹) |
| E | teacher-forced full-vocab logit checkpoints (steps {0,1,24,49} × 5 prompts) | abs Δ ≤ 2⁻⁵·M_step |
| E | per-step fingerprints, all 250 steps (float64, manifest protocol) | lse/mean ≤ 2⁻⁵·M64; std ≤ 2⁻⁴·M64 |
| E | per-step top-64 at reference indices, all 250 steps | abs Δ ≤ 2⁻⁵·M64 |
| E | **top-1 agreement, N = 250 teacher-forced steps** (OV#5/#11) | exact top-1 where recorded margin ≥ 2⁻⁴·M64 (176 of 250 steps in today's fixtures); below that, our top-1 ∈ {ref top-1, ref top-2} — no step is unasserted |
| — | free-running greedy divergence (128 steps × 5 prompts, GPU vs CPU reference) | **reported, not gated**: first-divergence index + both texts recorded in DECISIONS.md |

Why free-running is a report, not a gate: with per-logit deviation legitimately
up to ~2⁻⁵·M, any near-tie step can flip and permanently fork the trajectory —
a numeric gate would be either vacuous or a false-failure generator (the same
argument that made the P1 argmax gate tie-aware, compounded by self-feeding).
The teacher-forced form is the strongest agreement statement that composes
across steps; the free-run report is the honest observability on top.

The teacher-forced machinery is P1-5's `LogitMatchSuiteTests` pattern pointed at
the GPU pipeline — same fixtures, same manifest protocol, new tolerances.

## Enumerated edge-case tests (land with the code, hard rule 3 / METHODOLOGY 3)

1. KV append writes exactly the (layer, head, position) slot — synthetic
   distinguishable-pattern test, read back and checked exhaustively at small dims.
2. GQA mapping: KV head h serves Q heads {2h, 2h+1} — pattern test that fails on
   any off-by-one or repeat-interleave mistake.
3. Position indexing: RoPE at cache position p uses angle(p) — decode step at
   p > 0 diffed against CPU full-recompute (position bugs are what fixture steps
   24/49 exist to catch; this is the targeted unit form).
4. Empty cache (p = 0) decode == single-token attention (softmax over one score).
5. Context limit: 4096th append succeeds; the next decode step stops cleanly
   with the existing context-limit error — no OOB write (bounds-asserted).
6. Residency mode B (wired copy) output bit-identical to mode A (mmap) — exact.
7. Misaligned/odd tensor byte-offsets in the no-copy buffer path (the safetensors
   format allows 2-byte alignment) — upcast stays exact (IO-1 precedent).
8. Stop-set: engine-owned assembly reproduces {151645, 151643} for the pinned
   directory; EOS-1's DecodeLoop regression re-passes against the GPU backend.
9. CLI `--backend gpu`: empty prompt, >4K prompt — same errors as CPU backend.
10. Metal unavailable (no device) → clear error, not a crash (CLI + tests skip
    cleanly on machines without Metal, mirroring the checkpoint-absent skips).

## Instrumentation & benchmark deliverables

- **Per-token dual timing + dispatch count** (D5), aggregated: median GPU
  ms/token, median wall ms/token, wall−GPU delta, dispatches/token. Printed by
  the CLI and displayed/exported by the app.
- **Windowed decode rate**: the engine measures the canonical window natively
  (wall time from generated token 128 to 512 ⇒ tok/s) in addition to the overall
  rate — stronger than the app-reported overall rates the Phase 0 baselines had
  to use; both numbers go in every row.
- **Mac dev-loop sanity row** (M2 Pro, PROVISIONAL, results.md): catches gross
  regressions before device time is spent; never comparative.
- **On-device "before" row (James, per parity pins OV#12):** burst + sustained
  (5-min regenerate loop), cold + warm, greedy, pinned prompts
  (decode-essay / prefill-summarize roles per benchmarks/prompts/README),
  per-engine token counts, phys_footprint via Xcode gauge, Metal API validation
  OFF and recorded (P0A-1 addendum), device/iOS/battery-health annotations,
  PROVISIONAL marker. Template: benchmarks/phase0-runbook.md row format.
- **mmap vs wired-copy sustained stability (James, OV#9):** the sustained loop
  run once per residency mode, same session; record per-generation tok/s
  sequence (bimodality is the signal, not just the mean), phys_footprint, and
  any fault-stall observations. Close with a DECISIONS.md entry choosing the
  default residency mode Phase 3+ ships with.

## Task breakdown (seeded in docs/PRIORITIES.yaml)

| Task | Deliverable | Depends on |
|---|---|---|
| P2-1 | GPU weight residency: no-copy mmap buffer + offsets + wired-copy variant + exactness tests | — |
| P2-2 | Naive kernel set (matvec, rmsnorm, rope, swiglu, residual, embedding) + Tier-K tests | — |
| P2-3 | KV cache + kv-append + naive attention kernels + cache/GQA/position edge tests | P2-2 |
| P2-4 | GPU pipeline wiring + Tier-M/E suites + free-run report + engine stop set + CLI `--backend gpu` | P2-1, P2-3 |
| P2-5 | Per-token dual timing + dispatch count + windowed rate + Mac sanity row | P2-4 |
| P2-6 | QwenMetalApp thin shell (bench protocol runner, residency toggle, entitlement) | P2-4 |
| P2-7 (james) | On-device "before" row + mmap-vs-wired comparison + DECISIONS entries | P2-5, P2-6 |

## Exit criteria (PLAN.md phase table, walked)

- Preallocated K/V buffers, append per step, naive unfused attention over the
  cache; decode incremental from the first on-device build ✓ (P2-3/P2-4).
- Pre-committed fp16 gate passes vs CPU reference: all Tier K/M/E gates green
  ✓ (P2-4; gates in DECISIONS.md before any P2 test exists).
- "Before" benchmark row recorded per parity pins ✓ (P2-7).
- mmap vs wired-copy sustained-stability comparison recorded, default residency
  mode decided in DECISIONS.md ✓ (P2-7).
- DECISIONS.md session entries for anything decided/measured along the way
  (standing discipline), incl. the free-running divergence report.
