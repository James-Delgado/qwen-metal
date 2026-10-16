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

**Phases 0–6 exited (as of 2026-10-16); next is the optimization-campaign
charter (CAMP-1, James).** The engine runs Qwen3-1.7B end-to-end from its own
packed 4-bit format (q4g64, ~0.97 GB) with dequantization fused into every
weight-consuming kernel: a fused decode path (GQA SDPA with online softmax,
split-K; norm/RoPE/append and SwiGLU/residual folds; GPU argmax) at 200
dispatches/token, and a batched prefill path (512-position chunks through a
tiled q4g64 dequant-GEMM on `simdgroup_matrix` and a query-tiled causal
attention kernel). Every pre-committed correctness gate has held unmodified on
its first run; the GPU free-running trajectory is token-identical to its CPU
oracle on all fixture prompts, on both prefill paths, and at full 4K cache
depth the fp16 activation path produced no non-finite value (LD-1, 2026-10-16).
The engine is frozen at the `phase6-build` tag.

**The Phase 6 head-to-head** — [`docs/writeup.md`](docs/writeup.md), the
document the project is defined by. All three engines in one session on one
iOS build (26.6.1), interleaved, one metric definition per metric, iPhone 15
Pro, detached launches, Metal API validation OFF (speed session 2026-09-27;
energy round 2026-10-12 to 10-15; every table regenerated from the archived
exports and drift-tested):

| | qwen-metal | MLX LLMEval | llama.cpp |
|---|---|---|---|
| decode tok/s, warm burst (canonical window 128–512) | **31.16** | 43.40 | 34.80 |
| prefill tok/s (852 tokens, first-token span) | **232.63** | 387.60 | 440.71 |
| sustained plateau, 5-min regenerate loop | 22.83 | 34.94 | 27.41 |
| phys_footprint (Xcode gauge) | 574.9 MB mmap / 1.46 GB wired | 1.07 GB | 560.9 MB |
| net J/token (sustained, battery-delta, n=4) | 0.198 | 0.143 | 0.174 |

**Success metric: NOT MET** — 31.16 ÷ 43.40 = 0.718 vs ≥ 0.75. The committed
29.4 tok/s constant (0.75 × MLX's PROVISIONAL Phase 0 39.2) is exceeded; the
same-session MLX window rate is 43.40, ≈11% above the basis it was planned
from (validation ON, the app's overall rate, template feeding, an older OS —
the causes are named per engine in the writeup). By the pre-committed rule the
verdict is recorded with its anatomy and does not block the exit: on our
counted byte basis MLX decodes at 1.00× the measured 43.84 GB/s roofline and
we decode at 0.72× — the weight-streaming matvec runs at ≈80% of roofline and
≈1.4 ms/token of submission overhead sits outside the GPU. Prefill is 60% /
53% with one owner, the dequant-GEMM's 0.78 TFLOPS compute plateau on the A17
Pro (our attention kernel reaches 0.92 on the same silicon). Two gates are on
record FAILED with their anatomy (Phase 4 overhead 1.40 vs ≤1.2 ms; Phase 5
GEMM M=8 19.54 vs ≥30.69 GB/s); their remedies (PIPE-1, P5-2C) and the GEMM
lever (GE-1) are seeded for the campaign. The energy row's dominant limitation
is a waived start-of-charge anchor; the writeup's §7 lists seventeen such
limitations. Ledger: `DECISIONS.md`; rows: `benchmarks/results.md`; raw
exports: `benchmarks/phase6/`.

## Documents

| Doc | What it is |
|---|---|
| [`docs/writeup.md`](docs/writeup.md) | The Phase 6 benchmark writeup: headline table, success-metric verdict, sustained-thermal chart, energy with both error bars, roofline from measured denominators, per-component headroom, measurement limitations, honest gaps, reproducibility — every table embedded verbatim from `tools/phase6_analyze.py` and drift-tested |
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
- **`benchmarks/`**: results tables + pinned prompt set + runbooks + the Phase 6
  raw-export archive (`benchmarks/phase6/`). Rows are never overwritten.
- **`docs/generator/`**: matplotlib + reportlab scripts that produce
  `docs/architecture.pdf` — a rendered snapshot of the planning docs, regenerated
  at phase boundaries (never edited independently).

## Development

- Engine tests: `swift test` (XCTest; oracle suites need the local-only
  consolidated checkpoint under `models/` and skip cleanly without it).
- Backlog drift test: `.venv/bin/python -m pytest tests/test_priorities.py -q`
- Writeup drift test: `.venv/bin/python -m pytest tests/test_phase6_analyze.py tests/test_writeup.py -q`;
  regenerate its tables and figures with
  `.venv/bin/python tools/phase6_analyze.py benchmarks/phase6 --out docs/writeup`.
- Design doc rebuild: see [`docs/generator/README.md`](docs/generator/README.md).
- All official benchmark numbers come from a physical iPhone; the iOS target is
  deployed manually via Xcode. Correctness is enforced by an oracle chain — no GPU
  result is ever diffed against a reference that legitimately differs from it.
