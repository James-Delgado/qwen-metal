# METHODOLOGY.md — binding standards for qwen-metal

Short by design. The detailed versions live in PLAN.md (invariants), CLAUDE.md
(hard rules), and the phase specs. This file is the checklist an agent re-reads at
post-task review. Deviating from any rule here requires a DECISIONS.md entry BEFORE
the deviation lands.

## Correctness

1. **The oracle chain is sacred** (PLAN.md invariant 4). No GPU result is ever
   compared against an oracle that legitimately differs from it. Every new kernel
   gets its diff test before any optimization.
2. **Numeric gates are pre-committed.** Tolerances, roofline fractions, and
   agreement rates for a phase are written to DECISIONS.md before the phase starts.
   Gates are never set or adjusted after seeing results. Tolerances never loosen.
3. **Tests land with the code** — including the enumerated edge-case tests for host
   plumbing, not just kernels. A deliverable without its tests is not done.

## Measurement

4. **No invented numbers.** Roofline denominators, success targets, and baseline
   rows come from measurements recorded in DECISIONS.md / benchmarks/results.md,
   never from assumptions. If a spec needs a number that hasn't been measured,
   measuring it becomes a task.
5. **Benchmark protocol pins are invariant** (PLAN.md): greedy sampling, pinned
   prompts, per-engine token counts, phys_footprint, canonical decode window
   (tokens 128–512), regenerate-loop sustained protocol, battery-delta energy
   (sustained only). Comparative rows that violate a pin are invalid.
6. **Timing is dual** (GPU timestamps + wall clock) everywhere, always.

## Record-keeping

7. **DECISIONS.md is append-only** and gets a dated entry from any session that
   decided or measured anything. benchmarks/results.md rows are never overwritten.
8. **Scope discipline:** nothing from the PLAN.md non-goals list gets built, ever,
   without a recorded decision. Follow-ups go to docs/PRIORITIES.yaml, not into
   the current diff.
