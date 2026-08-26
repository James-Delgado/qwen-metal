# CLAUDE.md — qwen-metal

> **Entry point for any agent or contributor. Read in this order:**
> 1. [`docs/AGENT_OPERATION.md`](docs/AGENT_OPERATION.md) — the standard operating
>    procedure (the per-task workflow). **This project's SOP overrides global defaults.**
> 2. [`docs/PRIORITIES.yaml`](docs/PRIORITIES.yaml) — the task backlog. The next
>    action is the **lowest-`rank` task with `status: ready`**.
> 3. [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — binding standards for this project.
> 4. `PLAN.md` — goal, scope cuts, invariants, phase exit criteria (referenced below).

## Tier

- **full** — task backlog + tests-with-code + one-commit-per-task, plus pre-committed
  gates, backlog drift test, trial/decision ledger (DECISIONS.md), per-task post-task
  review, and session logs. Chosen because this is a research-grade benchmark project
  whose claims must survive scrutiny.

## Project status

Phase 0 (baselines + toy kernels) — not started. Plan solidified 2026-08-20 after full
eng review (20 findings folded). No code yet; docs only. Blocking decision: pin the
model (Qwen 2.5 vs Qwen 3 — architecture fork) in DECISIONS.md before Phase 0a.
See PLAN.md phase table for the roadmap.

## Codebase map

```
PLAN.md                     project charter: goal, non-goals, invariants, phases
CLAUDE.md                   this file — standing instructions + SOP entry point
DECISIONS.md                append-only decision/measurement ledger
TODOS.md                    deferred work with context (currently: CI)
Package.swift               engine package manifest: QwenMetalEngine library +
                            qwen-metal-cli executable (target QwenMetalCLI)
Sources/
  QwenMetalEngine/          engine core (shared library — all engine logic here)
  QwenMetalCLI/             macOS CLI entry point (thin; no engine logic)
docs/
  AGENT_OPERATION.md        per-task SOP (authoritative for this repo)
  PRIORITIES.yaml           ranked task backlog (drift-tested)
  METHODOLOGY.md            binding standards
  PRD-phase-0.md            Phase 0 deliverables + acceptance criteria
  phases/phase-0-1.md       engineering spec for Phases 0b and 1
benchmarks/                 results.md + pinned prompt set (created when first row lands)
tests/                      test_priorities.py (backlog drift test) +
                            QwenMetalEngineTests/ (XCTest; explicit path in manifest —
                            repo test root is lowercase on a case-insensitive FS)
tools/                      Python fixture/reference-dump scripts (Phase 1, pinned deps)
```

Planned (Phase 2): `QwenMetalApp/` — thin iOS SwiftUI shell.

## Environment & running

- Swift toolchain via Xcode (macOS): `xcode-select` points at the release
  `/Applications/Xcode.app` (26.6+). Engine package tests: `swift test`. CLI:
  `swift run qwen-metal-cli ...`. An Xcode beta also exists at
  `/Applications/Xcode-beta.app` — never use it for benchmark rows.
- iOS target: never built/run by agents — James deploys manually in Xcode.
- Python tooling (`tools/`): `python3`, deps pinned in `tools/requirements.txt`.
- Backlog drift test: `.venv/bin/python -m pytest tests/test_priorities.py -q`
  (one-time setup: `python3 -m venv .venv && .venv/bin/pip install pyyaml pytest`;
  call venv binaries directly, never `source activate`).

## Skill routing

When the user's request matches a capability below, invoke it as your FIRST action.

| Request type | Invoke |
|---|---|
| Bugs / "why is this broken" | `investigate` (gstack) or `superpowers:systematic-debugging` |
| **Code review of a diff** | **`/code-review`** — *not* bare `review` (that's gstack's) |
| Deep code/MLE audit → backlog | `/code-audit` |
| Architecture / plan review | `plan-eng-review` (gstack) |
| Ship / open a PR | `ship` (gstack) or `/pr` |
| Update docs after shipping | `document-release` (gstack) |

## Conventions that bind agents

- Tests land **with** the code, not after.
- For substantive diffs, run `/code-review` (or `ecc:swift-reviewer`) before commit.
- Capture discovered follow-ups in `docs/PRIORITIES.yaml`; never drop them.
- Don't modify pinned invariants (tolerances, gates, packed-layout schema, benchmark
  protocol pins) without a decision recorded in DECISIONS.md.

---

# Standing instructions (project-specific)

Read PLAN.md before doing anything. It defines goal, scope cuts, invariants, and
phase exit criteria. The current phase's spec is in docs/phases/. If no spec exists
for the phase being requested, stop and say so — specs are written just-in-time from
the previous phase's results, not invented mid-session.

## Project shape

- Language: Swift for host code, Metal Shading Language (.metal) for kernels.
  C++ interop is allowed where it clearly helps (e.g., file parsing), but prefer Swift.
- Two build targets:
  - `qwen-metal-cli` (macOS command-line): the dev workhorse. All unit tests,
    correctness diffs, and first-pass profiling run here.
  - `QwenMetalApp` (iOS): thin SwiftUI shell around the engine for on-device runs and
    official benchmarks. Requires Increased Memory Limit entitlement.
- The engine core is a shared Swift package used by both targets. No engine logic in
  the app target.

## Hard rules (from PLAN.md invariants — do not violate)

1. Never materialize dequantized weights to a memory buffer. Dequant happens in
   registers inside the consuming kernel. Quantization covers ALL weight matrices,
   including embeddings and the lm_head.
2. Never add scope from the non-goals list (no extra quant formats, samplers,
   architectures, MoE, batching, simulator support, third-party safetensors libs)
   even if it seems easy or the session would benefit. Flag it as a suggestion in
   DECISIONS.md instead.
3. Every GPU kernel gets a correctness test that diffs against the CPU reference
   before any optimization work on it begins. Host plumbing (parser, config,
   tokenizer adapter, decode loop) gets its enumerated edge-case tests in the same
   change as the code — the TDD rule binds ALL code, not just kernels.
4. Preallocate the KV cache at model load (exists from Phase 2 on). No dynamic growth.
5. Weights load via mmap, not heap reads. (Phase 2 additionally benchmarks a
   wired-copy variant for sustained stability — see PLAN.md invariant 3.)
6. Numeric gates (tolerances, roofline fractions, agreement rates) for a phase are
   committed in DECISIONS.md BEFORE that phase starts. Never set a passing bar after
   seeing results. Never loosen a tolerance — pressure to loosen is a bug signal.
7. All timing records BOTH GPU timestamps and wall clock. GPU-only timing is
   forbidden; the wall−GPU delta is the dispatch-overhead metric.
8. In the CPU reference, all matmul-shaped work (QKV/MLP/lm_head projections AND
   per-head QK^T / PV products) goes through the single validated Accelerate sgemm
   wrapper. Elementwise logic (RMSNorm, RoPE, softmax, SwiGLU, embedding lookup)
   stays hand-rolled, readable Swift.

## Dev loop

- Default to building and testing the macOS CLI target. Do not attempt to run the
  iOS target — device deployment, signing, and all on-phone benchmarking are done
  manually by James in Xcode. Prepare the code and the benchmark harness; he runs it.
- Unit tests: XCTest in the engine package. Kernel tests compare GPU output to CPU
  reference within the tolerance defined in the current phase spec.
- When perf work begins (Phase 3+), do not guess at bottlenecks. Add counters/timers,
  and structure kernels so Instruments' GPU counters can attribute time. If a change
  is speculative, mark it clearly and keep it in a separate commit.

## Session discipline

- At the end of any session that made a decision (chose a layout, pinned a model repo,
  changed a kernel strategy, got a benchmark number), append a dated entry to
  DECISIONS.md: what was decided/measured, and why.
- Benchmark numbers go in benchmarks/results.md as dated table rows with device,
  iOS version, battery health, burst/sustained, cold/warm annotations. Never
  overwrite old rows. Phase 0 rows carry a PROVISIONAL marker (see PLAN.md
  baseline staleness rule).
- Commits are small and phase-scoped. Commit messages reference the phase
  (e.g., "phase3: pack weights into grouped 4-bit layout").
- If something in a phase spec conflicts with reality (API changed, measurement
  contradicts an assumption), stop, log it in DECISIONS.md, and surface it rather
  than silently working around it.

## Testing

- Framework: XCTest, engine package. Run via `swift test` in the engine package.
- Reference oracles: HF transformers fp32 (primary), mlx-lm (secondary/ecosystem,
  and Phase 3 quality gate). Regeneration scripts live in tools/ with pinned
  Python dependency versions and the pinned model revision — regenerating fixtures
  with unpinned versions is a bug.

## Architecture document upkeep

docs/architecture.pdf is a rendered snapshot of PLAN.md + DECISIONS.md + the
phase specs, produced by docs/generator/ (see its README for the procedure).
At the end of each phase — or when a DECISIONS.md entry changes an invariant,
exit criterion, or number the PDF displays — regenerate it: update the generator
scripts' content from the new entries, bump the version/date, rerun both scripts,
and commit scripts + figures + PDF together. Never edit the PDF's claims without
a matching DECISIONS.md basis, and never let it drift silently: if it is stale
relative to DECISIONS.md, say so rather than presenting it as current.

The newcomer-facing surface rides the same trigger (decided by James,
2026-08-26): every phase-exit close-out (*-EXEC milestone) also refreshes
README.md (Status section, stale phrasing, doc tables) and any other doc a
curious engineer would check when stumbling on the repo — the repo must never
present an exited phase as current.
