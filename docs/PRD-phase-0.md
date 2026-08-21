# PRD — Phase 0: Baselines and Toy Kernels

Status: SOLIDIFIED (eng-reviewed 2026-08-20, 20 findings folded)
Owner: James (manual device work) + agent (macOS harness)
Exit gate: numeric criteria below, committed here BEFORE work starts (no post-hoc grading)

## Problem

The project's entire success metric — decode ≥ 0.75× MLX's measured decode tok/s — and its
entire analytical frame — roofline = measured bandwidth ÷ bytes/token — depend on numbers
nobody has measured yet. Phase 0 exists to replace every assumed number with a measured one,
and to stand up the smallest possible Metal dev loop so later phases start from a working
dispatch/test/profile workflow instead of a blank screen.

## Why now

Every later phase grades itself against Phase 0 outputs. Written specs are forbidden from
inventing numbers (PLAN.md invariant); this is the phase that makes that rule satisfiable.

## Deliverables

### 0a. Baseline benchmarks (manual — James, on the physical iPhone)

1. **Pin the model first.** Exact HF repo + revision for the Qwen checkpoint goes in
   DECISIONS.md *before* any baseline run. This is a fork decision, not a naming detail:
   Qwen 2.5 has QKV attention biases; Qwen 3 drops them and adds per-head Q/K RMSNorm.
   The pinned family fixes the Phase 1 module list.
2. **MLX baseline.** mlx-swift-examples LLMEval with the pinned model's mlx-community
   4-bit variant. Verify the 4-bit checkpoint derives from the pinned base revision, or
   run `mlx_lm.convert` from the pinned fp16 yourself. Record exact repo in DECISIONS.md.
3. **llama.cpp baseline.** Closest-equivalent GGUF quant (Q4_K_M acceptable; note the
   format difference in the results table).
4. **Metrics per engine** (protocol in PLAN.md — parity pins are mandatory):
   - Prefill tok/s, decode tok/s (canonical window: generated tokens 128–512),
     peak memory (phys_footprint via Xcode gauge — NOT Instruments Allocations,
     which misses Metal resource memory), energy/token (sustained runs only,
     battery-delta protocol).
   - Burst + sustained (sustained = regenerate-loop: on 4K context fill, reset and
     regenerate from the same prompt; identical loop for all engines).
   - Cold start and warm start recorded separately.
5. **Energy method dry-run.** One full battery-delta measurement cycle (≥8–10% SoC burn,
   idle-baseline subtraction, 80→70% band, 3–9 W plausibility anchor) to validate the
   protocol before it's load-bearing. Pin the operational method in DECISIONS.md.
   Calendar note: a full energy round is roughly a discharge cycle per engine — budget
   real days for it, and again at Phase 6.
6. **On-device bandwidth microbenchmark** (agent builds it, James runs it): a Metal
   streaming-read/triad kernel measuring achievable DRAM GB/s on the iPhone. This number
   is the roofline denominator for the entire project. The current PLAN.md figures
   (50–70 GB/s assumed, "~61 tok/s MLX") are provisional and internally inconsistent —
   the measured figure replaces them everywhere. Run it on the Mac too (dev-loop sanity).

### 0b. Toy Metal kernels (agent — macOS CLI target)

1. Device/queue/library setup in the shared engine package.
2. **Timing utility records BOTH GPU timestamps and wall-clock per dispatch batch.**
   The (wall − GPU) delta is the CPU encode/commit/scheduling overhead metric that
   Phase 4's dispatch-reduction exit criterion consumes. GPU-only timing is forbidden —
   it hides exactly the cost Phase 4 exists to fix.
3. Kernel 1: saxpy (establishes dispatch → readback → XCTest loop).
4. Kernel 2: naive fp16 matmul (one thread per output element).
5. Kernel 3: streaming-read/triad bandwidth microbench (deliverable 0a.6 above).
6. XCTest: saxpy + matmul diffed against CPU; timing-utility sanity tests
   (nonzero duration, start ≤ end, wall ≥ GPU).

## Acceptance criteria (all must hold)

- [ ] Model repo + revision pinned in DECISIONS.md, dated, before first baseline row.
- [ ] Baseline table in benchmarks/results.md: 2 engines × {prefill, decode, memory} ×
      {burst, sustained} × {cold, warm}, with device, iOS version, battery health,
      starting temperature per row. Energy: ≥1 validated sustained battery-delta row
      per engine.
- [ ] Measured iPhone DRAM bandwidth (GB/s) recorded in DECISIONS.md with the kernel
      used; PLAN.md roofline numbers updated to derive from it.
- [ ] Absolute success target computed and committed in DECISIONS.md:
      target = 0.75 × (MLX measured decode tok/s at canonical window).
- [ ] saxpy + matmul + triad kernels pass XCTest on macOS; timing utility reports
      GPU time AND wall time.
- [ ] Baseline rows explicitly marked PROVISIONAL (Phase 6 re-runs all engines in one
      session on one OS build for the publishable head-to-head).

## Out of scope for Phase 0

Any model inference in our engine (Phase 1), any on-device deployment of our code
beyond the bandwidth microbench, tokenizers, safetensors parsing, CI (captured in
TODOS.md), any kernel optimization.

## Risks

| Risk | Mitigation |
|---|---|
| Battery-delta noise swamps signal | Dry-run in 0a.5 validates before load-bearing; ≥3 repeats, error bars |
| llama.cpp iOS runner friction | Q4_K_M via llama.cpp example app; a thin wrapper is acceptable; timebox to a day |
| Bandwidth microbench measures cache, not DRAM | Working set ≫ SLC size (≥1 GB streamed); report best sustained, not peak |
| mlx-community quant ≠ pinned base | Verify provenance or convert from pinned fp16 (OV#12) |

## Dependencies

None. Phase 0a and 0b can run in parallel (James on device, agent on Mac).
Phase 1 may start once the model is pinned (its spec depends on the family fork);
Phase 1 exit requires Phase 0's fixture workflow only, not the baseline table.
