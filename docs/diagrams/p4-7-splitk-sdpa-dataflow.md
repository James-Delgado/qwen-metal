# Data flow: split-K / two-pass fused SDPA (P4-7)

**Task:** P4-7 — "Split-K / two-pass fused SDPA (occupancy: >16
threadgroups at depth) + edge tests re-passed" (docs/PRIORITIES.yaml,
rank 18.7; Phase 4 iterate round, docs/phases/phase-4.md 2026-09-12
addendum).

**Landed:** 2026-09-12, commit `840adb5`. Binding record: DECISIONS.md
2026-09-12 P4-7 entry. Implementation:
`Sources/QwenMetalEngine/Metal/FusedSDPAKernel.swift` (pass 1
`sdpa_decode_split_f16`, pass 2 `sdpa_decode_reduce_f16`); sole call
site is the folded layer in `Sources/QwenMetalEngine/Metal/GPUModel.swift`
(`encodeFoldedLayer`). Dims below are the pinned model's (16 query
heads, 8 KV heads, headDim 128, GQA 2:1); the kernels are
shape-generic within headDim ≤ 128.

The diagram shows one decode step of one layer.

```
INPUTS (per layer, per token)
  qVec: fp16 [16 heads × 128]        KV cache slab (layer L): fp16
  (post QK-norm + RoPE,              K[kvHead][0…p][128] ─┐ GQA: qHead h reads
   from the cluster kernel)          V[kvHead][0…p][128] ─┘ kvHead = h / 2
        │                                   │
        ▼                                   ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ PASS 1  sdpa_decode_split_f16          128 threadgroups = 16 heads × 8 ║
║                                        chunks (was 16 total in P4-2)   ║
║                                                                        ║
║  positions 0…p, ceil-divided into 8 contiguous chunks:                 ║
║    chunk 0: [0 … c)   chunk 1: [c … 2c)   …   chunk 7: [7c … p+1)      ║
║                                                                        ║
║  one threadgroup = (head h, chunk s), 128 threads = 4 simdgroups       ║
║  ┌──────────────────────────────────────────────────────────────┐      ║
║  │ sg0  j = start+0, +4, +8 …   ┐  each simdgroup: own online   │      ║
║  │ sg1  j = start+1, +5, …      │  softmax (m, l, acc[dims]);   │      ║
║  │ sg2  j = start+2, +6, …      │  score_j = simd_sum(q·K_j)·√d⁻¹│     ║
║  │ sg3  j = start+3, +7, …      ┘  acc += w_j·V_j  (fp32)       │      ║
║  │          │                                                   │      ║
║  │          ▼  threadgroup memory, FIXED sg order 0→3           │      ║
║  │  merge: m_c = max mᵢ;  l_c, acc_c rescaled by e^(mᵢ−m_c)     │      ║
║  └──────────────────────────────────────────────────────────────┘      ║
║   empty chunk (p+1 < 8·c): writes sentinel m = −inf, l = 0             ║
║   p == 0: pass 1 early-outs entirely (no state written)                ║
╚═══════════════════════╤═══════════════════════════════════════════════╝
                        │ one fp32 partial state per (head, chunk),
                        │ slot index  part = head·8 + chunk
                        ▼
        SCRATCH (GPU-private, ~66.5 KB, reused across all 28 layers)
        ┌─────────────────────────────────────────────────┐
        │ partM  [16×8] fp32      running max  m_c        │
        │ partL  [16×8] fp32      denominator  l_c        │
        │ partAcc[16×8][128] fp32 numerator    acc_c      │
        └─────────────────────────────────────────────────┘
                        │  (serial encoder + hazard tracking orders
                        │   pass1-write → pass2-read → next layer)
                        ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ PASS 2  sdpa_decode_reduce_f16                16 threadgroups (1/head) ║
║                                                                        ║
║  every thread merges the head's 8 states in FIXED split order 0→7:     ║
║    mT = max m_c        (skip m_c = −inf sentinels)                     ║
║    lT = Σ e^(m_c−mT)·l_c                                               ║
║    numerator[t] = Σ e^(m_c−mT)·acc_c[t]     (thread t = output dim t)  ║
║    out[h][t] = fp16( numerator[t] / lT )                               ║
║                                                                        ║
║  p == 0 bypass: out[h][·] = V[kvHead][0][·]  raw half copy from cache  ║
║                 (bitwise: −0.0 and NaN payloads survive)               ║
╚═══════════════════════╤═══════════════════════════════════════════════╝
                        ▼
  OUTPUT  attnOut: fp16 [16 heads × 128], head-major
                        │
                        ▼
  o_proj matvec+residual (GPUModel encodeFoldedLayer) — consumes it directly
```

## Reading notes

- **Where the parallelism came from:** the P4-2 kernel was the pass-1
  box with only the (head) axis — 16 threadgroups walking all of
  `0…p` each. Adding the chunk axis multiplies workers by 8 while each
  does ⅛ of the positions; the price is that softmax's global
  normalization must finish later, which is exactly what the scratch
  state + pass 2 do (partial online-softmax states compose exactly
  under the rescale rule shown).
- **Determinism:** both merges are fixed-order (simdgroups 0→3 inside
  pass 1, splits 0→7 in pass 2), so the pair is bitwise deterministic
  across runs — the pipeline's incremental-decode bitwise contract
  rides on this (test-pinned per depth in
  `FusedSDPAKernelTests.testSplitBoundaryDepthSweepMatchesOracle`).
- **The scratch is the only new data at rest**, and it is transient:
  written and consumed inside the same one-command-buffer-per-token,
  never CPU-visible, reused by all 28 layers (the serial encoder's
  hazard tracking serializes write → read → rewrite).
- **Dispatch accounting:** the two boxes are 2 dispatches/layer —
  fused path 171 → 199 dispatches/token, measured (≤300 gate; pins in
  `GPUQuantModelTests` tiny 8/10 and the real-artifact smoke 199).
- Precision is unchanged from the naive chain's boundary semantics:
  fp16 cache/query reads, fp32 arithmetic throughout, fp16 store;
  gates per spec D5 (attention species max(2⁻⁷·M, 2⁻¹¹) vs the
  CPU-quant oracle).

Diagram is documentation only — the binding numbers and gate record
live in DECISIONS.md and benchmarks/results.md (2026-09-12 P4-7 rows).
