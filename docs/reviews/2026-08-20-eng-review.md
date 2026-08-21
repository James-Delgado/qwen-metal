# Engineering Review Record — 2026-08-20 (plan solidification)

Full deliberation record of the /plan-eng-review that produced PLAN.md v2,
docs/PRD-phase-0.md, and docs/phases/phase-0-1.md. DECISIONS.md carries the
operative one-paragraph versions; THIS file preserves the reasoning so future
phase-spec sessions (SPEC-P2..P6 in docs/PRIORITIES.yaml) don't lose context.

Reviewed inputs: the three Downloads drafts (PLAN.md, CLAUDE.md, phase-0-1-spec.md).
Outcome: 20 findings (8 first-pass + 12 outside-voice), all resolved and folded.

---

## Part 1 — First-pass findings and resolutions

### D1. safetensors parser: build vs. borrow
jkrukowski/swift-safetensors exists (zero-dep Swift reader). REJECTED in favor of
the ~150-line from-scratch parser: the library's zero-copy path returns Core ML
types (MLTensor/MLMultiArray) that fight a raw-pointer engine, and invariant 5
(mmap, file-backed pages under iOS memory accounting) is easier to guarantee with
our own mmap. The format is a JSON length-prefixed header + raw blobs.

### Issue 1 → KV-cache split (James's hybrid, superseding both offered options)
Problem: Phase 3's original exit criterion ("decode near roofline, bandwidth-limited")
was structurally unreachable — KV cache didn't arrive until Phase 4, so Phases 2-3
recomputed full attention per token (compute-bound, an algorithm no competitor
engine uses; the Phase 2 "before" row would be meaningless).
Resolution: split the KV-cache concept in two.
- Phase 2 gets the MINIMAL cache: preallocated K/V buffers, append per step, naive
  unfused attention kernel reading the cache. Bookkeeping + one simple kernel;
  decode is incremental and memory-bound from the first on-device build.
- Phase 4 keeps the hard parts: fused GQA SDPA, RMSNorm/RoPE folding into
  neighbors, dispatch reduction.
- Phase 3 gains a standalone fused dequant-matvec microbenchmark: achieved GB/s vs
  measured bandwidth at fixed short context — kernel quality judged independently
  of whole-model behavior.
Side effect: CLAUDE.md hard rule "preallocate KV at load" becomes true from Phase 2.

### Issue 2 → Layered quant oracle (James's design)
Problem: at Phase 3 the GPU runs packed 4-bit weights while the CPU reference runs
fp16 — quantization error swamps kernel-bug tolerances; the oracle chain silently
breaks exactly at the trickiest kernel.
Resolution, layered by where bugs actually live:
1. Unpack/dequant layer — EXACT match required. Dequantizing a weight tile
   (scale·q + bias per element) is deterministic, no reduction. A test kernel dumps
   a dequantized tile; must match CPU dequant of the same packed bytes bit-for-bit.
   This is where the classic bugs live: bit-shift errors, nibble order,
   group-boundary indexing, scale/bias lookup.
2. Matvec layer — tight tolerance (~1e-3 relative per output element) vs the
   CPU-quant reference. Same packed weights, same dequant math, fp32 accumulation
   both sides; the ONLY difference is reduction order (GPU tree reduction vs CPU
   loop). Bit-exactness here is a false goal — fp addition isn't associative.
3. Adversarial packing fixtures: group sizes not dividing rows evenly (if layout
   permits), extreme scales, all-zero groups, negative-heavy groups. Packing bugs
   hide at boundaries.
4. Quality gate with defined metrics: same fixture prompts; per-step logits from
   mlx-lm running the same checkpoint at the same nominal format; top-1 agreement
   rate + mean KL divergence + a fixed-slice perplexity scalar. Grading is
   "our quantization quality is in the same band as MLX-4bit-vs-fp16's own
   agreement rate," NOT byte-match (our packed layout is ours; rounding choices
   differ). If our number is way off-band, the PACKING SCRIPT is the suspect, not
   the kernel — and layer 1 tells you which.
Resulting oracle chain (stated in PLAN.md invariant 4): CPU ref ← HF fp32 (P1) →
CPU-quant ref ← quality gate vs mlx-lm 4-bit → GPU dequant tile (exact) → GPU
matvec (tolerance) → full model. Every arrow is a test that exists.

### Issue 3 → Battery-delta energy protocol (James's methodology)
Problem: "Instruments Energy Log ÷ tokens" is unitless, not energy.
Resolution: headline number = battery-delta joules, SUSTAINED ONLY.
Sizing math that justifies every parameter: iOS reports SoC in 1% steps; 1% of a
~13 Wh battery ≈ 130 J. Burn ≥8-10% SoC per run so ±1% read quantization is
≤~12% error before averaging; at 4-8 W GPU draw that's 20-40 min — which doubles
as the sustained-thermal run. Idle baseline (same duration/screen/brightness,
airplane mode, no inference) measured separately, subtracted; table value =
(run − idle) J ÷ tokens. Same SoC band (80→70%; SoC reporting least linear at
extremes), battery health % scales rated → effective capacity, ≥3 repeats with
mean ± spread, and a 3-9 W implied-power sanity anchor (outside → run is broken).
Energy Log kept per-run as "energy impact (relative, unitless)" — cheap,
fine-grained in time (shows prefill vs decode), cross-checks the ranking.
Writeup gets a measurement-limitations paragraph (no power-rail access on iOS →
battery-delta with stated error bars, sustained-only) — credibility feature.

### Issue 4 → Slim logit fixtures (James's refinements on top of the slim option)
Problem: naive per-step full-vocab fixtures = 152k vocab × 50 steps × 5 prompts
≈ 150 MB of committed binaries.
Resolution (~13 MB plain git, no LFS):
- Full-vocab checkpoints at steps {0, 1, mid, last} — early steps catch
  embedding/norm/lm-head drift at onset; mid/last catch POSITION-DEPENDENT bugs
  (RoPE indexing at larger positions, cache-boundary errors once Phase 2's cached
  path diffs against this same fixture set).
- Per-step scalar fingerprints (logsumexp, mean, std of the full logit vector,
  ~12 B/step) close the tail gap: the only miss-class for checkpoint sampling is a
  bug perturbing low-rank tail logits on non-checkpoint steps; broad tail drift
  moves these scalars. Every step is checked at some resolution.
- Per-step top-64 values + indices for ranking-drift visibility.
- TIE-AWARE argmax: the reference dump records the top-1-vs-top-2 margin per step;
  steps below a small epsilon are exempt from exact top-1 (assert our top-1 ∈
  reference top-2 instead). Kills the near-tie-flip false failure — a correct
  implementation with different summation order can flip a near-tie and burn a
  debugging session on a non-bug. Everything else: exact, all 50 steps.
- Regeneration = forensics: checked-in generator script (pinned deps + model
  revision) regenerates the full 150 MB oracle locally on demand.

### Issue 5 → Per-module activation fixtures are PRIMARY
The draft made module-level oracles conditional ("verify ... if logits drift").
First-run end-to-end mismatch is near-certain; conditional oracles mean blind
bisection across 6 module types — repeated in every phase that diffs against the
reference (P2 port, P3 quant, P4 fusion). Now: the dump script also emits
embeddings out, layer-0 pre/post-attention, layer-0 post-MLP, last-layer out,
final-norm out for fixture prompt #1; every module gets an XCTest against its
slice BEFORE pipeline wiring. Same fixtures serve Phases 2-5 kernel diffs.

### Issue 6 → HF fp32 reference, 1e-3, escape hatch deleted (James's refinements)
Problem: draft allowed an fp16-computed reference (mlx-lm) with 1e-2 tolerance and
a "loosen with logged justification" hatch. fp16 reference noise sits exactly where
real bugs live (wrong RoPE theta, off epsilon, transposed rare-head weight → 1e-3
to 1e-2 drift). The hatch is the rot vector: each loosening looks reasonable, the
sum passes everything.
Resolution: reference = HF transformers, CPU, torch_dtype=float32. Key precision
argument: the checkpoint stays bf16 and that's FINE — both sides load identical
bits and upcast exactly (bf16→fp32 appends zero mantissa bits; fp16→fp32 exact).
Only divergence source: summation order, ≈1e-4 absolute on logits of magnitude
~20. So tolerance = 1e-3 with an order of magnitude of headroom, and the hatch is
DELETED — pressure to loosen is itself the bug signal. HF over mlx-lm because it's
an independent implementation (we parse config.json ourselves; an HF config quirk
can't replicate into both sides) and CPU fp32 has no TF32-style silent downcast.
mlx-lm → secondary loose ecosystem check; returns as Phase 3 quality-gate
comparator. Interaction caught: fixtures must be stored fp32 — fp16 spacing at
logit magnitude 16-32 is ~0.016, coarser than the tolerance; fp16 storage would
fail the suite on storage noise alone. Fixture set ~7 MB → ~13 MB.

### Issue 7 → 11 enumerated edge-case tests
Unenumerated tests don't get written. Parser ×5 (bf16 bytes, truncated, malformed
header, sharded reject, OOB/overlapping offsets), config ×2 (missing key, tied
flag both ways), decode stops ×2 (+seeded temperature if built), tokenizer-id
equivalence vs Python dump, timing sanity, CLI (empty prompt, >4K context).

### Issue 8 → Accelerate sgemm in the CPU reference (James's guardrails)
Arithmetic: full recompute over 50 steps ≈ 10^13 FLOPs/prompt; scalar Swift at
0.5-2 GFLOPS → hours per prompt, overnight per suite → the oracle stops being run
and the whole trust chain dangles from a test nobody executes. Accelerate sgemm
(drives AMX, hundreds of GFLOPS) → minutes. Trust argument: the reference's
credibility was never "every instruction is ours" but "every piece of MODEL LOGIC
is simple enough to audit" — sgemm contains zero model logic. Guardrails:
(a) ONE wrapper, validated by a triple-loop test on random odd-shaped/non-square/
non-power-of-two matrices (the naive loop lives only in that test, ~15 lines);
the external HF oracle independently polices transpose/leading-dim bugs — a
botched sgemm call can't self-validate. (b) ALL matmul-shaped work goes through
the wrapper, including per-head QK^T and PV (they dominate at longer contexts;
leaving them naive reinstates the overnight problem). Elementwise stays scalar.
Side benefit: the Phase 3 CPU-quant reference reuses this engine → quality gate
and adversarial fixture runs inherit the speedup.

---

## Part 2 — Outside-voice findings (verbatim; all 12 accepted via D11.0)

1. **The headline success metric is undefined.** "Close 70–80% of the decode-speed
gap to MLX" — gap measured from what starting point? If from the Phase 2 naive fp16
build, quantization alone (4× fewer bytes/token in a bandwidth-bound regime) closes
most of the gap mechanically, and the target is nearly un-failable. If it means
"reach 70–80% of MLX's absolute decode tok/s," say that. As written, the project's
single success criterion can be declared met or missed by choice of denominator.
Fix: define it as an absolute number — e.g. decode ≥ 0.75 × measured MLX tok/s at a
specified context depth — before Phase 2.

2. **The roofline denominator is never measured, and the plan's own arithmetic
contradicts itself.** PLAN.md asserts 50–70 GB/s bandwidth → "ceiling ≈ 50–60
tok/s," then separately calls MLX "near the physical roofline (~61 tok/s)" — MLX
would exceed the stated ceiling. Meanwhile Phase 0's toy kernels run on macOS only;
nothing in any phase measures achievable DRAM bandwidth on the actual iPhone. The
entire Phase 6 roofline analysis rests on a number that is only ever assumed, and
is currently self-inconsistent. Fix: add an on-device streaming-read/triad
bandwidth microbenchmark to Phase 0 and use the measured figure everywhere.

3. **The sustained benchmark contradicts the fixed 4K context.** ≥5 min continuous
generation at ~40 tok/s is ~12k tokens; the engine caps at 4K with no cache
management. What happens at context fill is undefined — and whatever it is (reset,
sliding restart, stop), all three engines must do the identical thing or the
sustained/thermal/energy comparisons are meaningless. Related: bytes-per-token
grows with KV depth, so "decode tok/s" is context-dependent and the protocol never
fixes the context depth at which it's measured. Fix: define a regenerate-loop
protocol and a canonical measurement window (e.g. tokens 128–512) for all engines.

4. **Quantization coverage of embeddings and lm_head is unspecified.** Qwen
2.5-1.5B has a ~151k vocab; the (tied) embedding matrix is ~15% of all parameters,
and the output-head matvec reads the whole thing every token. Unquantized fp16
head ≈ 0.46 GB/token of extra read — it alone would blow the roofline.
mlx-community checkpoints have their own recipe for this. The plan's quant spec
covers only the packed layout for "weights" generically. Fix: decide explicitly
(quantize head + embeddings, matching MLX's recipe) and account for it in the
bytes/token model.

5. **Phase 2 has no defined correctness gate for fp16-GPU-vs-fp32-CPU.** The plan
is rigorous about CPU-vs-Python tolerance (1e-3) and Phase 3 quant tolerances, but
Phase 2's exit is "generates correct text (diff vs CPU reference)." fp16 GPU
kernels will not meet the fp32 fixture tolerances, and greedy trajectories will
legitimately diverge after enough steps. This seam — the first place numerical
drift is *expected* — is the one place with no threshold. Fix: define per-module
fp16 relative tolerances against the activation fixtures plus a
top-1-agreement-over-N-steps gate, before Phase 2 starts.

6. **Model pinning is sequenced too late, and "Qwen 2.5 or 3" is an architecture
fork, not a naming detail.** Phase 0 baselines must use the pinned checkpoint, but
the plan defers pinning to "when Phase 1 begins." Worse: Qwen2.5 has QKV attention
biases; Qwen3 drops them and adds per-head Q/K RMSNorm. PLAN.md's module list
("RMSNorm, RoPE, GQA, SwiGLU") matches *neither* exactly — attention bias and
QK-norm both absent. That omission will surface as an unexplained logit mismatch
in Phase 1. Fix: pin repo+revision before Phase 0a and correct the module list for
the chosen family.

7. **Phase 0 baseline numbers will be stale by Phase 6.** Months of iOS updates,
battery-health drift, and ambient-temperature change between the Phase 0 rows and
your engine's final numbers invalidate the head-to-head. Fix: treat Phase 0 rows
as provisional target-setting only; Phase 6 exit criterion must include re-running
all three engines in one session on one OS build.

8. **The battery-delta energy protocol's operational cost went unexamined, and it
can't cover burst runs.** 8–10% SoC × ≥3 repeats × 3 engines, each constrained to
the 80→70% band, means roughly a full discharge cycle plus recharge-and-cool waits
per benchmark round — days of manual phone-babysitting, repeated at Phase 6. And a
burst run cannot burn 8% SoC, so "energy per token" is undefined for burst even
though the benchmark table implies it for both modes. Fix: scope energy to
sustained-only explicitly; budget the calendar time; consider dropping energy for
llama.cpp if the harness cost doesn't pay for itself.

9. **mmap'd weights on iOS are treated purely as an accounting win; the eviction
downside is ignored.** File-backed pages stay clean and evictable — under memory
pressure iOS purges them and decode re-faults weights from NAND mid-generation,
producing bimodal sustained tok/s. This is exactly why llama.cpp ships mlock,
which iOS won't grant at this size. Fix: benchmark mmap vs. dirty-heap-copy
variants for sustained stability, not just peak-footprint.

10. **The Phase 0 timing utility is specified to hide the cost Phase 4 exists to
fix.** "GPU start/end timestamps, not wall clock" systematically excludes CPU-side
encode/commit/scheduling overhead — which, at ~hundreds of dispatches per decoded
token, is likely the dominant cost of the naive Phase 2 port. Kernels will look
fast while tok/s is bad, with no instrumentation explaining the difference. Fix:
record both GPU time and wall-clock per token from day one; the delta *is* the
dispatch-overhead metric Phase 4's exit criterion needs.

11. **Un-failable exit criteria.** Phase 3: "within a *defined fraction* of
roofline" — defined by whom, when? Specs written just-in-time by the same person
grading them means the number can be set after seeing results. Fix: commit the
fraction (and Phase 2/4 numeric gates) in DECISIONS.md before each phase starts.

12. **Cross-engine generation parity is unpinned.** Nothing fixes sampler settings
(LLMEval defaults ≠ greedy), chat-template application, or the fact that GGUF and
HF tokenizers produce different token counts for the same prompt string — which
directly skews prefill tok/s. Also, "peak memory via Instruments Allocations"
won't capture Metal resource memory; use phys_footprint/Xcode gauge. Fix: protocol
pins sampler=greedy, exact prompt strings, per-engine token counts reported, and
one footprint metric. Similarly, the Phase 3 perplexity gate names no eval
corpus/slice and assumes the mlx-community 4-bit checkpoint derives from your
pinned base revision — verify, or run `mlx_lm.convert` yourself from the pinned
fp16.

All 12 folded into PLAN.md v2 / PRD-phase-0.md / phase-0-1.md. No cross-model
tension with the first-pass findings — purely additive.

---

## Part 3 — Coverage, failure modes, parallelization (as reviewed)

Phase 0+1 planned-codepath coverage before amendments: 4/17 paths had planned
tests (24%), 13 gaps — all closed by Issues 5-7 resolutions. Final failure-mode
audit: ZERO critical gaps (nothing simultaneously untested + unhandled + silent).
Softest spot: oracle drift if fixtures are regenerated with unpinned
transformers/model versions → resolved by pinning deps + revision in the regen
script (tools/).

Parallelization lanes (Phase 0+1): Lane A (Metal harness) ∥ Lane B (parser/config)
∥ Lane C (Python fixture scripts) → Lane D (modules + decode + CLI + tokenizer)
after B and C. No shared-module conflicts.

---

## Part 4 — Context sources for future phase-spec sessions

When writing SPEC-P2..P6 (see docs/PRIORITIES.yaml), read:

| Source | What it holds |
|---|---|
| PLAN.md v2 | invariants, protocol pins, phase exit criteria (operative truth) |
| DECISIONS.md | dated decisions + the OPEN list; numeric gates land here pre-phase |
| this file | full reasoning behind every decision; OV findings verbatim |
| docs/phases/phase-0-1.md | fixture/oracle machinery Phases 2-5 reuse |
| benchmarks/results.md | measured numbers (bandwidth, baselines, targets) |
| TODOS.md | deferred items (CI) that may unblock/land per phase |

Phase-specific spec obligations recorded during review:
- SPEC-P2 must pre-commit: per-module fp16 tolerances vs activation fixtures,
  top-1-agreement-over-N-steps gate (OV#5, #11); include mmap-vs-wired-copy
  sustained-stability bench (OV#9); "before" row per protocol pins (OV#12).
- SPEC-P3 must pre-commit: microbench bandwidth fraction (OV#11); name the
  perplexity eval slice (OV#12); verify mlx-community 4-bit provenance or convert
  from pinned fp16 (OV#12); embeddings/lm_head quantization per MLX recipe (OV#4).
- SPEC-P4: dispatch-overhead target consumes the wall−GPU delta metric (OV#10).
- SPEC-P6: same-session/same-OS re-runs of all engines (OV#7); energy calendar
  budget + possible llama.cpp energy drop decision (OV#8); measurement-limitations
  section (Issue 3 resolution).
