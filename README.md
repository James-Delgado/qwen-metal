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

**Phases 0–5 exited; Phase 6 (benchmark writeup) is next** (as of
2026-09-23). The engine runs Qwen3-1.7B end-to-end from its own packed
4-bit format (q4g64, ~0.97 GB) with dequantization fused into every
weight-consuming kernel: a fused decode path (GQA SDPA with online softmax,
split-K; norm/RoPE/append and SwiGLU/residual folds; GPU argmax) at 200
dispatches/token, and — new in Phase 5 — a batched prefill path: the prompt
is processed in 512-position chunks through a tiled q4g64 dequant-GEMM
(threadgroup tiles + `simdgroup_matrix`, fp32 accumulate) and a query-tiled
causal attention kernel, streaming the weights once per chunk instead of
once per token. Every pre-committed correctness gate has held unmodified on
its first run; the GPU free-running trajectory is token-identical to its CPU
oracle on all fixture prompts, on both prefill paths. Measured (iPhone 15
Pro, detached): **decode 31.67 tok/s warm-burst** (Phase 4; **the committed
29.4 target = 0.75 × MLX's measured 39.2 is exceeded**; unchanged through
Phase 5 at 31.05 / 30.90) and **prefill 172.23 tok/s** at the Phase 5 exit
rows (852-token prompt; floor ≥135 PASS; ≈4.7× the sequential path
in-session, claim-grade) — **240.86 tok/s** on the engine as it stands
after the in-phase query-tiled attention kernel — at 573 MB mmap
phys_footprint. Against MLX's PROVISIONAL ≈370 tok/s prefill that is 47%
at exit / 65% at close-out, with the remaining gap attributed to one
component: the dequant-GEMM's 0.78 TFLOPS compute plateau on the A17 Pro
(the attention kernel reaches 0.92 on the same silicon), now 93% of the
prefill span. Two gates are on record FAILED with their full anatomy:
Phase 4's per-token wall−GPU overhead (1.40 ms vs ≤1.2; ≈62% OS/driver
latency around an idle GPU) and Phase 5's GEMM microbench at M=8 (19.54
vs ≥30.69 GB/s; the A17 Pro is already compute-bound at M=8, so the
gate's bandwidth premise does not hold there). Their approved remedies
(pipelined decode; an M=8-exact matrix-unit kernel) and the GEMM
efficiency lever are seeded for the post-Phase-6 optimization campaign.
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
