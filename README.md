# qwen-metal

A from-scratch, single-model LLM inference engine in Swift + Metal for iPhone —
Qwen ~1.5–2B, 4-bit quantized — benchmarked head-to-head against MLX Swift and
llama.cpp on the same physical device.

Deliberately the "nanoGPT of Metal inference": one model family, one quantization
format, batch size 1, no abstraction layers — so every level of the stack that
mature engines hide (weight layout, kernel dispatch, quantized matvec, attention
over a cache) is implemented, measured, and understood by hand. The deliverable is
a working on-device engine **plus** a rigorous benchmark writeup: prefill tok/s,
decode tok/s, peak memory, and energy/token for all three engines, with a roofline
analysis explaining every gap.

**Success metric:** decode ≥ 0.75 × MLX's decode tok/s, measured same-session on
the same device at the canonical window (generated tokens 128–512). Decode is
memory-bandwidth-bound; the roofline denominator is measured on-device, never
assumed.

## Status

**Phases 0–3 exited; Phase 4 (fused attention + dispatch reduction) is
next** (as of 2026-09-05). The engine decodes Qwen3-1.7B end-to-end from its
own packed 4-bit format (q4g64, ~0.97 GB) with dequantization fused into every
weight-consuming kernel — every pre-committed correctness gate across all four
phases has held unmodified on its first run, the packing recipe measures at KL
parity with mlx-lm's 4-bit (quality gate in-band), and the GPU free-running
trajectory is token-identical to its CPU oracle on all fixture prompts.
Measured so far (iPhone 15 Pro, PROVISIONAL rows): DRAM bandwidth 43.84 GB/s;
MLX baseline 39.2 tok/s decode (committed target: 29.4 = 0.75×); packed decode
**20.6 tok/s** warm-burst (~3.0× the Phase 2 bf16 "before" row, 70% of target)
at 1.43 GB resident (wired; 538 MB mmap phys_footprint); the weights-only
dequant-matvec microbench sustains 35.3 GB/s (80% of roofline, clearing its
pre-committed gate). The remaining gap is non-matvec time — Phase 4's target.
Ledger: `DECISIONS.md`; rows: `benchmarks/results.md`.

## Documents

| Doc | What it is |
|---|---|
| [`docs/architecture.pdf`](docs/architecture.pdf) | Rendered system-design document (Figures 1–7: context, dataflow, memory budget, fused dequant, roofline, oracle chain, roadmap) |
| [`PLAN.md`](PLAN.md) | Project charter: goal, non-goals, invariants, benchmark protocol, phase exit criteria |
| [`CLAUDE.md`](CLAUDE.md) | Agent/contributor entry point: read order, hard rules, dev loop |
| [`DECISIONS.md`](DECISIONS.md) | Append-only decision + measurement ledger (authoritative when docs disagree) |
| [`docs/PRIORITIES.yaml`](docs/PRIORITIES.yaml) | Ranked task backlog spanning all 7 phases (drift-tested) |
| [`docs/PRD-phase-0.md`](docs/PRD-phase-0.md) | Phase 0 deliverables + acceptance criteria |
| [`docs/phases/`](docs/phases/) | Per-phase engineering specs, written just-in-time |
| [`docs/reviews/`](docs/reviews/) | Full engineering-review records (all findings + reasoning) |

## Layout

- **Engine** (Phase 0b+): shared Swift package; `qwen-metal-cli` macOS target is
  the dev workhorse, `QwenMetalApp` (iOS) is a thin shell for on-device benchmarks.
- **`tools/`** (Phase 1+): Python reference-dump scripts (HF transformers fp32
  oracle), pinned deps.
- **`benchmarks/`**: results tables + pinned prompt set. Rows are never overwritten.
- **`docs/generator/`**: matplotlib + reportlab scripts that produce
  `docs/architecture.pdf` — a rendered snapshot of the planning docs, regenerated
  at phase boundaries (never edited independently).

## Development

- Engine tests: `swift test` (XCTest; oracle suites need the local-only
  consolidated checkpoint under `models/` and skip cleanly without it).
- Backlog drift test: `.venv/bin/python -m pytest tests/test_priorities.py -q`
- Design doc rebuild: see [`docs/generator/README.md`](docs/generator/README.md).
- All official benchmark numbers come from a physical iPhone; the iOS target is
  deployed manually via Xcode. Correctness is enforced by an oracle chain — no GPU
  result is ever diffed against a reference that legitimately differs from it.
