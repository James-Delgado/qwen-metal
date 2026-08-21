# qwen-metal: A minimal from-scratch LLM inference engine for iPhone

Revision: v2 — solidified 2026-08-20 after full eng review (8 first-pass + 12 outside-voice
findings folded). Supersedes the Downloads draft.

## Goal

Build a from-scratch, single-model LLM inference engine in Swift + Metal that runs
Qwen (1.5B–2B class, 4-bit quantized) on a physical iPhone, benchmarked head-to-head
against MLX Swift and llama.cpp on the same device. Success = a working engine plus a
quantified writeup: prefill tok/s, decode tok/s, peak memory, and energy/token for all
three engines, with a roofline analysis explaining the gaps.

**Success metric (absolute, no movable denominator):** our decode tok/s ≥ 0.75 × MLX's
measured decode tok/s, both measured in the same session on the same device at the
canonical measurement window (generated tokens 128–512). The concrete target number is
computed and committed in DECISIONS.md at Phase 0 exit. Matching MLX is not required;
measuring, explaining, and narrowing the gap is.

## Non-goals (scope cuts — do NOT build these)

- No architecture registry: exactly one model family (pinned in DECISIONS.md before
  Phase 0a — the Qwen 2.5 vs Qwen 3 choice is an architecture fork, see Target model).
- No MoE, no VLM/multimodal, no batching (batch size is always 1).
- One quantization format only: 4-bit grouped affine (MLX-style). No K-quants, no i-quants.
- Fixed max context (4K), preallocated KV cache. No dynamic cache management.
  (Sustained benchmarks handle context fill via the regenerate-loop protocol below.)
- Greedy + temperature sampling only. No top-p/top-k/beam/etc.
- No tokenizer implementation: borrow swift-transformers (or a minimal BPE port).
- No ANE/Core ML build target (ANE is closed to custom kernels). Core ML appears only
  as an optional benchmark comparison column.
- No iOS Simulator support: MLX-class Metal features don't run there, and simulator
  perf numbers are meaningless. Physical device + macOS only.
- No third-party safetensors library: hand-written single-file parser (~150 lines,
  mmap-based; sharded checkpoints are rejected loudly). Decided over swift-safetensors
  for raw pointer/mmap control.

## Core technical constraints (invariants — every phase respects these)

1. **Decode is memory-bandwidth-bound, and the roofline denominator is MEASURED, not
   assumed.** Roofline: decode tok/s ≈ measured DRAM bandwidth ÷ bytes read per token.
   Phase 0 measures achievable bandwidth on the physical iPhone with a streaming
   triad kernel; that figure (recorded in DECISIONS.md) replaces every assumed number.
   The bytes-per-token model accounts for ALL per-token reads: quantized weights
   including lm_head and embeddings, KV cache at the measurement depth, activations.
   Every design decision is evaluated by its effect on bytes-per-token first, FLOPs second.
2. **Weights stay 4-bit in DRAM, always — including embeddings and the lm_head.** The
   output-head matvec reads a ~151k×hidden matrix every token; left at fp16 it alone
   blows the roofline. Quantization coverage matches the MLX recipe for the pinned
   checkpoint. Dequantization happens in registers inside the consuming kernel (fused
   dequant-matmul). Never materialize a dequantized weight tensor to memory.
3. **iOS memory ceiling is hard.** The OS jetsams the app well before 8 GB. Use the
   Increased Memory Limit entitlement; mmap weights (file-backed pages are cheaper
   under iOS memory accounting than dirty heap pages) — but mmap'd pages are also
   evictable under pressure, which can produce bimodal sustained tok/s as weights
   re-fault from NAND. Phase 2 benchmarks mmap vs wired-copy variants for sustained
   stability, not just peak footprint. Budget: weights + preallocated fp16 GQA-sized
   KV cache + activations, with headroom.
4. **Correctness is diff-tested, not eyeballed — via an explicit oracle chain.**
   CPU reference ← validated against HF transformers fp32 logits (Phase 1)
   → CPU-quant reference (same code path, dequantized packed weights) ← quality-gated
   against mlx-lm 4-bit → GPU dequant-tile test (exact match) → GPU matvec (tolerance)
   → full model. Every arrow is a test that exists; no GPU result is ever compared
   against an oracle that legitimately differs from it. Numeric gates for each phase
   (tolerances, roofline fractions, agreement rates) are committed in DECISIONS.md
   BEFORE that phase starts — never set after seeing results. The Phase 2 fp16-GPU
   gate (per-module fp16 tolerances vs activation fixtures + top-1 agreement over N
   steps) is defined before Phase 2 begins.
5. **Dev loop: macOS first, iPhone for truth — and timing is dual from day one.**
   Identical Metal API on Mac; develop, unit-test, and first-pass profile there. All
   official benchmark numbers come from the physical iPhone only. Every timing
   measurement records BOTH GPU timestamps and wall clock; the delta is the CPU
   encode/commit/scheduling overhead that Phase 4's dispatch-reduction work targets.

## Target model & formats

- Model: Qwen 2.5 or Qwen 3 Instruct, 1.5B–2B dense variant. **Pin the exact HF repo
  and revision in DECISIONS.md BEFORE Phase 0a** (baselines must run the pinned model).
  The family choice fixes the module list:
  - Qwen 2.5: QKV projections carry biases.
  - Qwen 3: no QKV biases; adds per-head Q/K RMSNorm.
- Modules to implement: token embeddings (tied output head if the checkpoint ties
  them), RMSNorm, RoPE, GQA attention (+ family-specific: QKV bias OR QK-norm),
  SwiGLU MLP, final norm + logits.
- Weight input format: safetensors, single-file, parsed by our own mmap parser
  (Phase 1). Sharded checkpoints rejected with a clear error.
- Quantized format: our own packed layout, 4-bit grouped affine — group size 64,
  per-group fp16 scale and bias, w ≈ scale·q + bias. Covers all linear layers plus
  embeddings/lm_head (invariant 2). Packing done offline on macOS.

## Benchmark protocol (applies to every engine, every phase)

Same physical iPhone for everything. Record device model, iOS version, battery level
(>50%), battery health %, and starting temperature (rest to ambient between runs).

**Cross-engine parity pins (mandatory for any comparative row):**
- Sampler: greedy, temperature 0, for every engine (LLMEval defaults must be overridden).
- Exact prompt strings pinned in benchmarks/prompts/; chat template application
  documented per engine.
- Per-engine token counts reported (GGUF and HF tokenizers tokenize the same string
  differently — prefill tok/s is meaningless without the count).
- Memory metric: phys_footprint (Xcode memory gauge), one metric everywhere.
  Instruments Allocations misses Metal resource memory — not used for headline rows.

| Metric | How measured |
|---|---|
| Prefill tok/s | prompt tokens (per-engine count) ÷ prefill wall time |
| Decode tok/s | generated tokens ÷ decode wall time, canonical window = tokens 128–512 |
| Peak memory | phys_footprint via Xcode memory gauge |
| Energy per token | SUSTAINED RUNS ONLY: battery-delta protocol below |

**Burst vs sustained.** Burst = single short generation from rest. Sustained = ≥5 min
continuous generation via the regenerate-loop protocol: when the 4K context fills, reset
and regenerate from the same prompt; every engine runs the identical loop (a fixed
context cap with no cache management makes some reset policy unavoidable — pinning one
policy for all engines is what keeps thermal/energy rows comparable). Decode tok/s is
always reported at the canonical window so KV-depth-dependent bytes/token doesn't skew
comparisons. Report cold start (first launch, weights from disk) and warm separately.

**Energy protocol (sustained only — burst energy is undefined and not reported):**
energy = ΔSoC × battery-health-scaled capacity, converted to joules.
- ≥8–10% SoC burn per measured run (≈20–40 min at 4–8 W; doubles as the thermal run).
- Idle baseline (same duration/screen/brightness, airplane mode, no inference) measured
  separately and subtracted. Table value = (run − idle) joules ÷ tokens.
- Controlled: airplane mode, fixed minimum brightness, background refresh off, same
  starting SoC band (80%→70%), same starting temperature, unplugged. Battery health %
  recorded and used to scale rated → effective capacity.
- ≥3 repeats per engine; report mean ± spread. Implied average watts must land in
  3–9 W or the run is invalid.
- Instruments Energy Log is kept per-run as "energy impact (relative, unitless)" —
  a cross-check only; it never wears the label "energy per token."
- Calendar cost is real (≈ a discharge cycle per engine per round). Budgeted in
  Phase 0 and Phase 6; energy rows for llama.cpp may be dropped at Phase 6 if the
  harness cost outweighs the comparison value (decide in DECISIONS.md, not silently).

**Baseline staleness rule:** Phase 0 rows are PROVISIONAL (target-setting only). The
publishable head-to-head requires Phase 6 to re-run all three engines in one session,
on one OS build, interleaved. Never compare rows recorded months apart.

## Phases and exit criteria

Detailed specs live in docs/phases/, written just-in-time. Do not begin a phase
without its spec; do not write a phase's spec until the previous phase's results are
logged in DECISIONS.md. Each phase's numeric gates are committed in DECISIONS.md
before the phase starts (invariant 4).

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | Baselines + toy kernels + measured bandwidth | Per docs/PRD-phase-0.md acceptance list: pinned model; provisional MLX & llama.cpp rows; measured on-device DRAM GB/s; absolute success target committed; saxpy/matmul/triad kernels + dual-timing utility passing on macOS |
| 1 | CPU reference engine (macOS) | Greedy decode produces text; per-step logits match HF fp32 reference within max abs Δ ≤ 1e-3 on the fixture set; per-module activation tests pass |
| 2 | Naive Metal port + minimal KV cache, on-device | Preallocated K/V buffers, append per step, naive unfused attention over the cache (incremental decode from the first on-device build); passes the pre-committed fp16 gate (per-module tolerances + top-1 agreement over N steps) vs CPU reference; "before" benchmark row recorded; mmap vs wired-copy sustained-stability comparison recorded |
| 3 | 4-bit quant + fused dequant-matvec | ~4× memory drop vs Phase 2; standalone dequant-matvec microbenchmark achieves the pre-committed fraction of measured DRAM bandwidth at fixed short context; layered oracle passes (exact dequant-tile match; matvec ≤ ~1e-3 rel vs CPU-quant; adversarial packing fixtures; quality gate vs mlx-lm 4-bit in-band) |
| 4 | Fused attention + optimization | Fused GQA-correct SDPA kernel; RMSNorm/RoPE folding; dispatches-per-token reduced (measured via the wall−GPU timing delta); end-to-end decode tok/s vs roofline judged here; decode latency variance measured |
| 5 | Tiled prefill GEMM | Prefill tok/s benchmarked separately vs MLX; uses threadgroup memory + simdgroup_matrix |
| 6 | Benchmark writeup | Full table across engines re-run same-session/same-OS; sustained-thermal chart; energy/token (sustained, with error bars); roofline analysis from measured bandwidth; measurement-limitations section (no power-rail access → battery-delta with stated error bars; SoC quantization; provisional-vs-final baseline handling); honest gaps |

Stretch (only after Phase 6): speculative decoding with Qwen 0.5B draft model;
upstream a kernel improvement to MLX or ExecuTorch's MPS backend.

## Reference material

- llama.cpp `ggml-metal.metal` — reference for quantized matvec kernel patterns.
- MLX `quantized.metal` and Steel GEMM kernels — reference for grouped affine layout
  and tiled GEMM.
- Apple: Metal compute fundamentals; Metal debugger + Instruments GPU counters docs.
- mlx-swift-examples LLMEval — baseline app and deployment reference.
- swift-transformers — tokenizer (Qwen-verified, incl. chat templates).
