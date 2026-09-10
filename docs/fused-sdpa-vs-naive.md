# Fused SDPA vs. the naive attention chain — a reader's guide

> Reference note (written 2026-09-10, after P4-2 landed). Explains the two
> attention implementations in this engine: the naive three-kernel chain
> (`Metal/AttentionKernels.swift`, Phase 2) and the fused single-dispatch
> SDPA kernel (`Metal/FusedSDPAKernel.swift`, Phase 4). Binding specs:
> docs/phases/phase-2.md D4 and phase-4.md D2/D4/D5; gates in DECISIONS.md.
> This note explains — it does not bind anything.

Pinned model dims used throughout: hidden 2048, 28 layers, **16 query
heads**, **8 KV heads** (GQA group size 2), **headDim 128**, cache depth
`p+1` (positions `0…p` occupied at decode step `p`).

---

## 1. The computation

Decode-time attention is *not* a big square matmul. Each step has ONE new
query position, so per query head it is a vector-matrix-vector sandwich
over that head's slice of the KV cache:

```
      q[head]          K[kvHead]ᵀ              scores                probs
     ┌────────┐   ┌─────┬─────┬─────┐      ┌──┬──┬──┬──┐  softmax ┌──┬──┬──┬──┐
     │ 1×128  │ · │ 128 × (p+1)     │  =   │ 1 × (p+1) │  ──────► │ 1 × (p+1) │
     └────────┘   │ (cached K rows) │      └──┴──┴──┴──┘  (÷√128) └──┴──┴──┴──┘
                  └─────┴─────┴─────┘

      probs            V[kvHead]                out
     ┌──┬──┬──┬──┐  ┌────────────────┐     ┌────────┐
     │ 1 × (p+1) │ ·│ (p+1) × 128    │  =  │ 1×128  │
     └──┴──┴──┴──┘  │ (cached V rows)│     └────────┘
                    └────────────────┘
```

×16 query heads, where query heads {2h, 2h+1} share KV head h (GQA).
At window depth (p≈512) the dominant cost is streaming the K and V rows
from memory: ~46 MB/token across all 28 layers.

The softmax in the middle is the structural troublemaker: classically it
needs the row **max** and the row **sum** — global functions of *all*
p+1 scores — before any probability can be produced. That single fact
shapes both implementations.

---

## 2. The naive chain (Phase 2): three kernels, zero communication

Design rule: **one thread per output element, each thread computes its
element start-to-finish, threads never talk to each other.** Trivially
correct, trivially testable — and that was the point (hard rule 3:
correctness before optimization).

### 2.1 `attn_scores_f16` — one thread per score cell

```
grid = (p+1) columns × 16 head-rows          each thread, alone:
         j=0   j=1   j=2  ...  j=p             acc = 0
head  0 [ T ] [ T ] [ T ] ... [ T ]            for d in 0..<128:        ← serial
head  1 [ T ] [ T ] [ T ] ... [ T ]                acc += q[d]·K[j][d]
  ...                                          scores[head][j] = acc/√128
head 15 [ T ] [ T ] [ T ] ... [ T ]
```

Parallel across the score matrix; serial inside each 128-element dot.
Writes a **materialized fp32 scores buffer** ([16 × 4096] = 256 KB).

### 2.2 `softmax_rows_f32` — one thread per cell, redundant row scan

```
each thread (row r, column c), alone:
    rowMax = max over x[r][0..p]      ← re-walks the WHOLE row   (serial)
    rowSum = Σ exp(x[r][j] - rowMax)  ← re-walks the WHOLE row   (serial)
    out[r][c] = exp(x[r][c] - rowMax) / rowSum
```

Every thread in a row recomputes the identical max and sum — (p+1)×
redundant work, deliberately traded for zero inter-thread communication.
Writes a second 256 KB fp32 buffer (`probs`).

### 2.3 `attn_pv_f16` — one thread per output dim

```
grid = 128 dims × 16 heads                   each thread, alone:
         d=0   d=1  ...  d=127                 acc = 0
head  0 [ T ] [ T ] ... [ T ]                  for j in 0...p:          ← serial
  ...                                              acc += probs[j]·V[j][d]
head 15 [ T ] [ T ] ... [ T ]                  out[head][d] = fp16(acc)
```

### 2.4 What blocks what

The global softmax dependency is implemented with **dispatch
boundaries**. Metal's hazard tracking sees that softmax reads the buffer
scores wrote, and PV reads what softmax wrote, and serializes:

```
one layer, naive:

[ attn_scores ]══════╗                                    3 dispatches
   writes 256KB      ║ full pipeline barrier              2 device-memory
                [ softmax ]══════╗                          round trips
                   writes 256KB  ║ full pipeline barrier  ~3.3 µs encoder
                            [ attn_pv ]                     cost apiece
```

Each barrier drains the whole GPU and each intermediate makes a round
trip through device memory. ×28 layers ×every token.

---

## 3. The fused kernel (Phase 4): dissolve the dependency, then parallelize

### 3.1 Online softmax — the enabling identity

Keep three running values while streaming scores one at a time:
`m` (max so far), `l` (denominator so far), `acc` (PV numerator so far,
kept consistent with the current `m`). One new score `s` with value row
`v` updates the state:

```
mNew = max(m, s)
corr = exp(m − mNew)          ← rescales history if the max moved
w    = exp(s − mNew)
l    = l·corr + w
acc  = acc·corr + w·v         ← PV consumes the weight IMMEDIATELY
m    = mNew
final output = acc / l
```

Worked example, scores `[2, 5, 3]`, values `[v₀, v₁, v₂]`:

```
step   s    m     corr        l                     acc
 1     2    2      —          1                     v₀
 2     5    5    e⁻³        e⁻³ + 1               e⁻³·v₀ + v₁
 3     3    5     1         e⁻³ + 1 + e⁻²         e⁻³·v₀ + v₁ + e⁻²·v₂

acc/l = (e⁻³v₀ + e⁰v₁ + e⁻²v₂)/(e⁻³+e⁰+e⁻²)  ≡ classic softmax·V  ✓
```

Consequences: **no scores buffer, no probs buffer, no waiting** — the
softmax's "global" dependency became a streaming, *associative* update.
Associative means two partial states over disjoint position sets can be
merged afterwards with the same rescale rule — which is what makes the
parallel split below legal.

### 3.2 The execution shape — `sdpa_decode_f16`

```
dispatch ─── 16 threadgroups, one per QUERY head (fully independent;
   │         GQA mapping kvHead = head/2 lives here)
   └── threadgroup: 128 threads = 4 simdgroups × 32 lanes
         │
         ├─ sg0 walks positions j = 0, 4,  8, …   ┐ four INDEPENDENT
         ├─ sg1 walks positions j = 1, 5,  9, …   │ online-softmax
         ├─ sg2 walks positions j = 2, 6, 10, …   │ states (m, l, acc)
         └─ sg3 walks positions j = 3, 7, 11, …   ┘ over ¼ of the row each
                    │
                    └─ within a simdgroup, per position j:
                       lane ln owns dims {ln, ln+32, ln+64, ln+96}
                       partial = Σ q[d]·K[j][d] over its 4 dims
                       s = simd_sum(partial)·scale     ← register-level
                             (all 32 lanes get s; no memory, no barrier)
                       …then the online update; each lane keeps acc
                       for its own 4 dims
```

Per-position dataflow inside one simdgroup, visualized:

```
        K[j] row (128 fp16)                     V[j] row (128 fp16)
   ┌──┬──┬──┬──┬─ ─ ─┬──┐                  ┌──┬──┬──┬──┬─ ─ ─┬──┐
   │d0│d1│d2│…        │ │                  │  │  │  │  │      │ │
   └┬─┴┬─┴┬─┴─────────┴┬┘                  └┬─┴┬─┴┬─┴─────────┴┬┘
    ln0 ln1 ln2 … (strided across 32 lanes) ln0 ln1 …
      \  |  /                                  |
     partial products                          |
        \ | /                                  ▼
      simd_sum ──► s ──► (m,l,corr,w) ──► acc[lane dims] += w·V dims
      (hardware       broadcast to
       reduction)     every lane
```

### 3.3 The merge — the only synchronization point

After the loop, four `(m, l, acc)` states exist per head. They merge
through a few KB of **threadgroup memory** with ONE barrier:

```
sg0: (m₀,l₀,acc₀) ┐
sg1: (m₁,l₁,acc₁) │ write to threadgroup mem ── barrier ──►
sg2: (m₂,l₂,acc₂) │   mTotal = max(m₀…m₃)              fixed order ⇒
sg3: (m₃,l₃,acc₃) ┘   out[d] = Σ_g e^(m_g−mTotal)·acc_g[d]   bitwise
                              ÷ Σ_g e^(m_g−mTotal)·l_g       deterministic
```

Note the ownership shuffle: during the loop, lane `ln` of each simdgroup
accumulated dims {ln, ln+32, …}; at the write-out, thread `tid` (0–127)
owns output dim `tid` and sums the four simdgroups' contributions for it.
The threadgroup-memory pass is what permits the remapping.

Special case pinned by test: at **p = 0** the single weight is exactly
1.0, so the kernel copies the V row verbatim — exact for every bit
pattern (±0, subnormals, ±inf, NaN payloads). The accumulate form would
flip −0.0 to +0.0 (`0 + 1.0·(−0.0) = +0.0` in IEEE); see the DECISIONS
2026-09-09 P4-2 entry and follow-up NK-1.

### 3.4 Side-by-side

| | naive chain | fused SDPA |
|---|---|---|
| dispatches / layer (attention) | 3 | 1 |
| intermediate buffers | scores + probs, 256 KB each, fp32, device memory | none (registers only) |
| synchronization | 2 full pipeline barriers through device memory | 1 threadgroup barrier through on-chip memory; `simd_sum` in registers |
| softmax | classic two-pass, recomputed redundantly per thread | online single-pass, 4 parallel streams merged once |
| K/V traffic | K read by scores, V read by PV (separate kernels) | K and V streamed once, in the same loop |
| parallel width per head | (p+1)×16 threads (scores) but serial 128-dot each | 128 threads; dot parallel across lanes, positions across simdgroups |
| inter-thread communication | none (by design) | constant (simd_sum) + one merge |
| correctness gate | Tier K max(2⁻⁹·M, 2⁻¹¹) per kernel | attention-span max(2⁻⁷·M, 2⁻¹¹) vs the same oracle; p=0 bitwise |
| determinism | trivial (no communication) | fixed stride + fixed merge order ⇒ bitwise across runs (test-pinned) |

Both paths still use the same two `kv_append` dispatches (P4-3's cluster
fold will absorb those). Selection: `GPUModel(packed:…, kernelPath:
.naive/.fused)`; naive is the default until P4-4; the bf16 backend is
permanently naive (spec D4 — it is the Phase 2 correctness artifact).

---

## 4. Why fusion stops where it does — what forces a kernel boundary

Metal has no device-wide barrier *inside* a dispatch: threadgroups
cannot synchronize with each other mid-kernel. So a dispatch boundary is
required exactly when the **complete output of a computation scattered
across many threadgroups** must be **gathered** by the next step — and
none of the three dissolving tricks applies:

1. **Containment** — keep producer and consumer inside one threadgroup
   (fused SDPA: one head's whole computation fits one threadgroup).
2. **Redundant recomputation** — let each consumer recompute a cheap
   global quantity instead of waiting for it (naive softmax's row scan;
   norm folding below).
3. **Associative streaming** — restructure the dependency into a
   running update whose partial states merge (online softmax).

A perhaps-surprising consequence: **norms do NOT force a boundary.**
RMSNorm needs the sum of squares of the whole 2048-dim vector — a global
reduction — but every consuming matvec already reads that entire vector
once per output row anyway. A matvec threadgroup can re-derive the sum
itself (trick 2, one extra pass over 2048 values) and apply the
normalization during its dot products. That is precisely the "norm
folds" lever in phase-4.md D3.

What *does* force boundaries here is the **matvec → matvec handoff**:
the full output vector of one wide projection (scattered across its
threadgroups) is the input of the next. Recomputing a 2048-dot per
consumer is real work (trick 2 explodes), there is no associative
reformulation (trick 3), and the vector doesn't fit one producer
threadgroup's ownership (trick 1). Per layer the irreducible chain is:

```
[norm ⊕ QKV] ─► [qk-norm/RoPE/append cluster] ─► [SDPA] ─► [o_proj ⊕ residual]
     ─► [norm ⊕ gate+up] ─► [swiglu ⊕ down ⊕ residual]
```

(⊕ = foldable into the same dispatch.) That lands at the **~8
dispatches/layer structural floor ⇒ ≈227/token** named in phase-4.md D3
— against 21/layer naive, 19/layer after P4-2, ≤300/token as the P4-3
gate. The true theoretical minimum is 1 kernel (one threadgroup grinding
the whole model serially — a curiosity, not a target: it forfeits the
GPU), and split-K-with-atomics tricks that could squeeze further are
excluded because nondeterministic accumulation order would break the
bitwise-replay contract the oracle suites rely on.

---

## 5. Pointers

- Naive kernels + wrappers: `Sources/QwenMetalEngine/Metal/AttentionKernels.swift`
- Fused kernel + wrapper: `Sources/QwenMetalEngine/Metal/FusedSDPAKernel.swift`
- Path selection: `GPUModel.KernelPath` in `Sources/QwenMetalEngine/Metal/GPUModel.swift`
- Tests: `tests/QwenMetalEngineTests/AttentionKernelTests.swift` (naive),
  `FusedSDPAKernelTests.swift` (fused: p=0 bitwise, GQA mapping,
  p=4095 boundary, adversarial online-softmax orderings, window depth,
  determinism, real-artifact smoke)
- Gates: DECISIONS.md "Phase 4 gates pre-committed" (2026-09-05);
  landing record: DECISIONS.md P4-2 entry (2026-09-09)
