# Phase 3 Spec — 4-bit quantization + fused dequant-matvec

Written 2026-08-25 (SPEC-P3), from Phase 2 results logged in DECISIONS.md.
Context sources: PLAN.md v2 (exit-criteria table, invariants 1/2/4), the
2026-08-20 eng review record Part 1 Issue 2 (the layered quant oracle — the
backbone of this phase) and Part 4 (SPEC-P3 obligations: OV#4/#11/#12),
docs/phases/phase-2.md (the kernel/gate machinery this phase extends),
DECISIONS.md 2026-08-22/25 entries (measured bandwidth 43.84 GB/s, target
29.4 tok/s, Phase 2 "before" range 6.7–8.6 tok/s, ~1.4× device-state
variance, residency decision). Numeric gates for this phase are committed in
DECISIONS.md ("Phase 3 gates pre-committed", 2026-08-25) — this file explains
them; the DECISIONS entry is the binding record.

---

## Purpose

Replace the bf16 weights with our own packed 4-bit grouped-affine format and
give every weight-consuming kernel a fused dequant path. Two things become
true for the first time:

1. **Bytes/token drops ~3.6×** (2 B → 0.5625 B per weight element), moving
   the packed decode roofline to ~45.2 tok/s on the measured 43.84 GB/s
   (DECISIONS 2026-08-25 P2-7). The 29.4 target becomes reachable in
   principle; Phases 4–5 close the efficiency gap.
2. **The oracle chain forks and stays intact** (PLAN invariant 4): a
   CPU-quant reference (the existing CPU model running dequantized packed
   weights) becomes the GPU's oracle, and is itself quality-gated against
   mlx-lm 4-bit — so quantization error and kernel bugs can never be
   conflated (eng review Issue 2).

Weights never exist dequantized in DRAM (hard rule 1): dequant happens in
registers inside the consuming kernel, and quantization covers ALL weight
matrices including embeddings and the tied lm_head (invariant 2 / OV#4).

## Scope

**In:** the packed-layout schema (pinned by this spec), the offline Swift
packer + packed artifact, the CPU-quant reference, the quality gate vs
mlx-lm 4-bit (named metrics + named perplexity slice), fused
dequant-matvec + embedding-gather kernels with the layered oracle
(exact tile → matvec tolerance → adversarial fixtures), GPU-quant pipeline
wiring + Tier-M/E suites, the standalone bandwidth microbenchmark with a
pre-committed fraction gate, the on-device memory/decode rows under the new
repeats/interleaving protocol, and the packed-weights mmap-vs-wired re-test.

**Out (unchanged non-goals + deferred):** any second quant format (no
K-quants/i-quants), fused attention / norm folding / dispatch reduction
(Phase 4), batched prefill GEMM (Phase 5), sampling beyond greedy, energy
rounds (Phase 6). The Phase 1 CPU reference and its fixtures stay frozen —
the CPU-quant reference reuses the same module code, never modifies it.
Attention (KV cache, scores/softmax/PV) is untouched this phase: the KV
cache stays fp16 and its kernels are Phase 2's.

## Design decisions

### D1. Packed layout schema (PINNED — changes require a DECISIONS.md entry)

4-bit grouped affine, group size 64 along the input (reduction) dimension,
w ≈ scale·q + bias (PLAN.md format pin). Every pinned weight-matrix in-dim
(2048, 6144) and the embedding row length (2048) divide by 64; a non-multiple
in-dim is rejected loudly by packer and loader.

Each source matrix `W [out, in]` becomes three tensors in one
safetensors-format file (parsed by our existing parser):

| Tensor | dtype/shape | Content |
|---|---|---|
| `{name}.q` | u32 `[out, in/8]` | 8 consecutive 4-bit codes per u32; element j of the group of 8 occupies bits [4j, 4j+4) (low nibble first) |
| `{name}.scales` | fp16 `[out, in/64]` | per-group scale |
| `{name}.biases` | fp16 `[out, in/64]` | per-group bias |

q is unsigned 0..15. 1-D norm vectors (per-layer norms, per-head q/k norms,
final norm) are not matrices — they pass through as raw bf16 tensors, and the
existing bit-shift upcast path consumes them (invariant 2 covers matrices;
this matches the MLX recipe, which quantizes linears + embeddings only).
The tied embedding is stored ONCE as its packed triplet; lm_head reads it as
`[out, in]` directly (Phase 2 established no transpose is ever materialized).

File details: written by our packer (D2) — data section starts 8-byte
aligned, and every `.q` tensor's byte offset is validated ≡ 0 (mod 4) at
pack AND load time so typed u32 loads are legal (all-even dims make this
structurally true; it is asserted, not assumed). `__metadata__` carries
source repo + revision (70d244cc…), packer version, group size, and format
tag; loaders verify revision like SharedCheckpoint does.

### D2. Packing recipe + offline Swift packer

> **AMENDED — current selection is SNAP-SCALE (2026-08-31 QR-3, decided by
> James; binding record: DECISIONS.md "QR-3 DECIDED" entry):** s₀ = range/15;
> edge = the dominant-|.| endpoint (min if |min| ≥ |max| else max);
> q₀ = max(round(|edge|/s₀), 1); s = |edge|/q₀; scale = fp16(s) first;
> bias = fp16(k·scale) with k = −q₀ (min dominant) or q₀ − 15 (max dominant).
> Zero sits on the stored grid for every zero-straddling group; the step stays
> ~range/15; the non-dominant endpoint may clip by ≤ ~one step.
> History: the original min/max selection below was first superseded by the
> A1 covering rule (2026-08-30 QR-1 — zero-aligned but step-inflating in
> asymmetric groups; measured 3.5% out of band on KL), then by snap-scale.
> Everything else in this section — fp16-round-before-code-selection, rounding
> pin, degenerate group, non-finite abort, schema D1 — stands unchanged.

Original (superseded) selection, per group of 64 fp32 values (exact bf16
upcast of the pinned checkpoint):
`scale = (max − min) / 15`, `bias = min`, both rounded to fp16 FIRST; then
`q = clamp(round((w − bias_fp16) / scale_fp16), 0, 15)` — codes are chosen
against the values as stored, so dequant is exactly reproducible from the
file with no hidden fp64/fp32 packer state. Degenerate group (max == min):
scale = 0, all q = 0, w = bias. Non-finite input weights abort the packer
(the pinned checkpoint is finite; anything else is corruption).

The packer is engine code (Swift, `qwen-metal-cli pack`), not tools/ Python:
it must share the parser, the exact upcast, and the dequant arithmetic with
the loaders it feeds, and its tests ride the XCTest suite. Output artifact:
`models/qwen3-1.7b-70d244cc-q4g64.safetensors` (~0.97 GB, local-only,
sha256 recorded in DECISIONS.md when produced). tools/ Python remains the
oracle side only (D6 dump).

### D3. CPU-quant reference (the Phase 3+ oracle; PLAN invariant 4 carve-out)

The existing CPU model code path, fed fp32 weights materialized by
dequantizing the packed file (`fma(float(q), float(scale), float(bias))` per
element). This materialization is explicitly the PLAN invariant 4 design
("CPU-quant reference — same code path, dequantized packed weights"): it is
a macOS test oracle, never device code, so hard rule 1 (register-only
dequant) continues to bind the engine's GPU path and the iOS app unqualified.
No CPU module changes — only a new weight-loading front end.

Dequant determinism note (why exact gates below are achievable): q·scale is
EXACT in fp32 (a ≤4-bit integer times an 11-bit fp16 significand needs ≤15
significand bits), so `q*scale + bias` equals `fma(q, scale, bias)`
bit-for-bit regardless of fma contraction, on both CPU and GPU. One correctly
rounded operation total ⇒ both sides produce identical fp32 dequant values.

### D4. Fused GPU kernels (dequant in registers, hard rule 1)

- **dequant-matvec** (replaces the bf16 matvec for QKV/o/gate/up/down and
  lm_head): reads `.q` as u32, unpacks nibbles, dequants in registers per D3's
  arithmetic, multiplies fp16 activations, accumulates fp32; fp16-store and
  fp32-store (logits) variants as in Phase 2. Unlike Phase 2's naive
  one-thread-per-output pin, THIS kernel may be optimized (simdgroup
  reductions, vectorized loads) — it is the phase's named deliverable and the
  microbench (D7) grades it — but only AFTER its correctness tests pass
  (hard rule 3), and every optimization iteration re-passes them.
- **dequant-tile dump** (test-support kernel): writes the register-dequanted
  fp32 values of a tile straight to a buffer — the exact-match layer of the
  oracle (Issue 2 layer 1), where nibble-order/group-boundary/scale-lookup
  bugs live.
- **embedding-gather-dequant**: row gather + register dequant + fp16 store
  (replaces Phase 2's bf16 embedding-lookup).

Attention/norm/swiglu/residual kernels are unchanged (their operands are
activations or unquantized vectors).

### D5. Correctness gates: layered oracle, Phase 2 constants reused

Full table in the DECISIONS entry; structure here. The oracle for every
GPU-vs-CPU diff is now the CPU-quant reference; both sides consume
bit-identical dequant values (D3 note), so the Phase 2 divergence analysis
(fp16 activation rounding + reduction order only) transfers unchanged and
NO new tolerance constants are introduced for tiers:

| Layer (Issue 2) | Gate |
|---|---|
| 1. Dequant tile | EXACT — GPU fp32 tile dump == CPU dequant of the same packed bytes, bitwise; embedding-gather fp16 store == fp16(exact fp32), bitwise |
| 2. Fused matvec | Phase 2 Tier K unchanged: abs Δ ≤ max(2⁻⁹·M, 2⁻¹¹) vs CPU-quant oracle (BLAS.sgemm over dequant weights, hard rule 8) |
| 3. Adversarial packing fixtures | exact/structural (see edge-case list) |
| 4. Quality gate vs mlx-lm 4-bit | D6 — the only new numeric gates of the phase besides D7 |
| Full model Tier M | Phase 2 constants unchanged (2⁻⁸ / 2⁻⁷ module slices, embeddings exact) — oracle = CPU-quant activations at the same slice points, computed live |
| Full model Tier E | Phase 2 constants unchanged (2⁻⁵ stack slices; teacher-forced 5×50-step logit suite at 2⁻⁵·M_step / 2⁻⁵·M64 / fingerprints; tie-aware top-1 at ε_tie = 2⁻⁴·M64 with margins from CPU-quant logits) — computed live, not from dumped fixtures |
| Free-running divergence | REPORTED, not gated (Phase 2 rationale stands): 128 steps × 5 prompts, GPU-quant vs CPU-quant, first-divergence index + texts in DECISIONS.md |

The Phase 1 HF-fp32 fixtures do NOT gate the quantized model per-logit —
quantization error legitimately exceeds those tolerances; that seam is
exactly what D6 grades instead (the Issue 2 point).

### D6. Quality gate vs mlx-lm 4-bit (OV#12: named metrics, named slice)

Grades the PACKING RECIPE (applied to the CPU-quant reference; the GPU
inherits via Tier E). "In band" means: our quantization loses about as much
as the ecosystem's 4-bit loses, measured identically. Comparator: the
PIN-1-verified mlx-community/Qwen3-1.7B-4bit @ 3b1b1768 (provenance from HF
file history, DECISIONS 2026-08-21; the dump re-verifies the revision
programmatically at run time). All three metrics are computed teacher-forced
on the REFERENCE fp32 argmax token sequences (identical prefixes for every
engine — the P1-5 premise), fp32 logits, float64 softmax/statistics:

1. **Top-1 agreement** vs the fp32 argmax, over the 250 fixture steps
   (5 prompts × 50).
2. **Mean full-vocab KL** — KL(P_fp32 ‖ P_engine) in nats, mean over the
   250 steps.
3. **Perplexity slice (the OV#12 named slice):** WikiText-2, config
   `wikitext-2-raw-v1`, TEST split, pinned dataset revision (recorded in
   tools/pins.py when P3-3 lands): standard document concatenation
   (protocol documented in the dump script), first 4096 tokens under the
   pinned HF tokenizer, one full context window; ppl = exp(mean NLL) over
   positions 1..4095.

Band-setter numbers (mlx-lm 4-bit's own agreement/KL/Δppl vs the same fp32
reference) are measured by the extended tools/ dump and recorded in
DECISIONS.md BEFORE any metric of ours is computed. The gate formulas are
pre-committed now (constants in the DECISIONS entry, flagged
judgment-derived): agreement within 4 percentage points below mlx's; KL ≤
1.5× mlx's; Δppl ≤ 1.5× mlx's Δppl + 0.01. Out-of-band ⇒ suspect the
PACKER, and layer 1 of the oracle says whether the kernel is implicated
(Issue 2's diagnosis rule). The 2026-08-22 mlx secondary dump (argmax-level)
is NOT reused for the band — it predates these definitions.

### D7. Standalone dequant-matvec bandwidth microbenchmark (OV#11)

The phase's kernel-quality judgment, independent of whole-model behavior
(eng review Issue 1): one command buffer running exactly one token's worth
of REAL packed-weight matvecs (28 layers × {q,k,v,o,gate,up,down} + lm_head
= 197 dispatches; no attention/norm/elementwise, no KV — weights-only by
construction, which is what "fixed short context" isolates).

- **Metric:** aggregate weight-stream rate = total packed bytes read
  (q + scales + biases ≈ 0.967 GB) ÷ summed GPU time; per-shape rates
  reported alongside (small per-layer matvecs will individually
  underperform; the aggregate is the roofline-relevant number).
- **Gate (pre-committed, on-device only):** aggregate ≥ **0.70 ×
  43.84 GB/s = 30.7 GB/s** on the pinned iPhone, best of the pinned repeats
  protocol (D8). Mac runs are dev-loop sanity, PROVISIONAL, never gated.
- Dual timing per hard rule 7 (GPU + wall; the wall−GPU delta at 197
  dispatches feeds the Phase 4 baseline). BW-1's caveat stands: the triad
  denominator likely understates read-mostly achievable bandwidth ~5–10%,
  so the fraction is conservative in the strict direction.

### D8. Device-row protocol addendum: repeats + interleaving (from P2-7)

P2-7 measured ~1.4× run-to-run device-state variance at identical settings.
Pinned for ALL Phase 3+ device rows (extends the benchmark protocol; PLAN
parity pins unchanged):

- Any single-configuration row: ≥3 repeats in one session; report median
  AND full range — never a single number.
- Any A-vs-B comparison (residency, before/after): repeats INTERLEAVED
  A,B,A,B,A,B (≥3 per side) in one session; a directional claim requires
  the two ranges to not overlap, otherwise record "unresolved at n=3".
- Detached launches mandatory (validation OFF, recorded) — the P2-7
  attached-session lesson.

Under this protocol P3-7 re-runs mmap vs wired-copy on the ~0.97 GB packed
weights (sustained loop per mode) and closes the residency question left
open by P2-7 with a DECISIONS.md entry; mmap remains the default unless
that entry says otherwise.

## Memory budget

Packed weights ~0.97 GB (file-backed mmap) + KV 448 MiB + activations/logits
< 10 MB ⇒ ≈ 1.5 GB resident — the PIN-1 derived number, now measured for
real. The "~4× memory drop" exit criterion is operationalized honestly as
the derived weight-byte ratio 2 B / 0.5625 B ≈ **3.6×** (the plan's "~4×"
is the nibble ratio before group overhead): verified by (a) packed file size
~0.97 GB vs 3.44 GB and (b) on-device phys_footprint rows (mmap mode
under-reports file-backed weights — annotate per the Phase 2 precedent;
the wired-copy row is the honest total-resident check).

## Enumerated edge-case tests (land with the code, hard rule 3 / METHODOLOGY 3)

1. Nibble order pin: hand-built packed bytes with a known asymmetric code
   pattern → CPU and GPU dequant match hand-computed values exactly (a
   swapped-nibble implementation is a hard value mismatch).
2. Group-boundary indexing: adjacent groups with distinct scale/bias;
   elements 63/64/65 dequant against the correct group (classic off-by-one
   site, Issue 2 layer 1).
3. Degenerate group (max == min ⇒ scale 0): dequant == bias exactly; packer
   emits q = 0.
4. Extreme scales: fp16 max-normal / min-normal / subnormal scale and bias
   patterns dequant exactly (no overflow/flush surprise); negative-heavy
   groups round-trip.
5. Round-trip: `|w − dequant(pack(w))| ≤ scale/2 + fp16(scale,bias)
   rounding effects` per element (structural sanity of the recipe), and
   pack(w) is deterministic across runs (byte-identical artifact).
6. Non-multiple-of-64 in-dim → packer AND loader reject loudly.
7. `.q` offset alignment: loader rejects a packed file whose u32 tensor
   offset is not 4-byte aligned (doctored-file test); packer output asserts
   alignment.
8. Provenance: packed `__metadata__` revision verified at load (mismatch =
   loud error); quality-gate dump verifies the mlx comparator revision.
9. Wrong-format loads: packed file into the bf16 loader and bf16 checkpoint
   into the packed loader both fail with clear errors naming the expected
   format (extends P2-4's badWeightDtype species).
10. Tied lm_head: packed embedding triplet consumed as [out, in] by the
    fp32-store matvec — no transpose, no second copy (structural, P2-2
    precedent).
11. GPU-quant pipeline: CLI/app edge behavior unchanged (empty prompt, >4K
    prompt, no-Metal, missing-model errors) on the packed path.

## Instrumentation & benchmark deliverables

- Existing per-token dual timing + dispatch count ride unchanged
  (dispatches/token drops as bf16 kernels are replaced 1:1 — the measured
  DispatchCounter reports it automatically).
- Microbench (D7): CLI subcommand + app runner, aggregate + per-shape GB/s,
  Mac PROVISIONAL row + on-device gated row (P3-7).
- **On-device rows (James, P3-7, all under D8 protocol):** packed decode
  row (burst + sustained, canonical window — progress check against 29.4;
  P2-7 projected ~22–32 tok/s at observed efficiency), phys_footprint rows
  (mmap AND wired for the memory-drop criterion), microbench row,
  mmap-vs-wired interleaved sustained comparison + residency close-out
  DECISIONS entry.
- Free-run divergence report (D5) in the P3-5 DECISIONS entry.

## Task breakdown (seeded in docs/PRIORITIES.yaml)

| Task | Deliverable | Depends on |
|---|---|---|
| P3-1 | Packed layout + Swift packer (`pack` subcommand) + adversarial/edge tests + packed artifact (sha256 → DECISIONS) | — |
| P3-2 | CPU-quant reference (packed → fp32 materialized oracle through the existing CPU model) + parity/smoke tests | P3-1 |
| P3-3 | Quality gate: tools dump extension (mlx full logits, ppl slice), band-setter measurement → DECISIONS, Swift metric computation, CPU-quant in-band | P3-2 |
| P3-4 | Fused GPU kernels (dequant-tile, dequant-matvec, embedding-gather) + exact-tile and Tier-K tests | P3-1 |
| P3-5 | GPU-quant pipeline wiring + Tier-M/E suites vs CPU-quant + free-run report + CLI/app packed-model plumbing | P3-2, P3-4 |
| P3-6 | Bandwidth microbench (CLI + app) + Mac sanity row | P3-4 |
| P3-7 (james) | On-device rows: microbench gate, memory-drop, decode row, packed mmap-vs-wired interleaved re-test + residency close-out | P3-3, P3-5, P3-6 |

## Exit criteria (PLAN.md phase table, walked)

- ~4× memory drop vs Phase 2 ✓ = packed artifact ~0.97 GB vs 3.44 GB
  (derived 3.6× stated honestly) + on-device phys_footprint rows (P3-1,
  P3-7).
- Standalone dequant-matvec microbench ≥ the pre-committed fraction
  (0.70 × 43.84 GB/s) on-device ✓ (P3-6 builds, P3-7 gates).
- Layered oracle passes ✓: exact dequant-tile match (P3-4); matvec at the
  reused Tier-K gate vs CPU-quant (P3-4); adversarial packing fixtures
  (P3-1); quality gate vs mlx-lm 4-bit in-band per the pre-committed
  formulas (P3-3).
- Tier-M/E suites green vs CPU-quant at the reused Phase 2 constants;
  free-run divergence reported (P3-5).
- DECISIONS.md entries for the packed artifact sha256, band-setter numbers,
  every gate outcome, the residency close-out, and anything else
  decided/measured (standing discipline).
