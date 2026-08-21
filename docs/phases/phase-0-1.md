# Phase 0 + Phase 1 Spec (v2 — solidified 2026-08-20)

Supersedes the Downloads draft. Phase 0's product framing and acceptance criteria live
in docs/PRD-phase-0.md; this file carries the engineering detail for 0b and all of
Phase 1. Phase 0a is manual (James, in Xcode, on the phone); 0b and Phase 1 are agent
builds. They can overlap once the model is pinned.

---

## Phase 0 — Baselines and toy kernels

See docs/PRD-phase-0.md for the full deliverable list, acceptance criteria, and risks.
Engineering notes for the agent-built pieces (0b):

- Metal harness in the shared engine package: device/queue/library setup.
- Timing utility: records GPU start/end timestamps AND wall clock per dispatch batch
  (hard rule 7). Sanity tests: nonzero duration, start ≤ end, wall ≥ GPU.
- Kernel 1: saxpy (y = a*x + y) — establishes dispatch + readback + test loop.
- Kernel 2: naive fp16 matmul (one thread per output element).
- Kernel 3: streaming-read/triad bandwidth microbench. Working set ≥1 GB (must defeat
  the system-level cache); reports best sustained GB/s. Runs on macOS for dev-loop
  sanity; James runs it on the iPhone for the roofline denominator.
- XCTest: saxpy + matmul vs CPU; timing sanity.

Exit: PRD acceptance criteria.

---

## Phase 1 — CPU reference engine (macOS)

### Purpose

An obviously-correct implementation of the full Qwen forward pass on CPU. This is the
oracle every GPU kernel diffs against for the rest of the project. Model logic stays
readable; the only performance concession is the BLAS wrapper (below), without which
the test suite takes hours and stops being run.

### Inputs

- Pinned model (repo + revision from DECISIONS.md, pinned at Phase 0a). Download the
  original fp16/bf16 safetensors, not a quantized variant — quantization is Phase 3.
- Module list follows the pinned family: Qwen 2.5 → QKV projection biases;
  Qwen 3 → no biases, per-head Q/K RMSNorm.
- Tokenizer via swift-transformers (Hugging Face).

### Build steps

1. **safetensors parser** (~150 lines, ours — see PLAN.md non-goals). Header is a
   JSON length-prefixed index; tensors are raw little-endian blobs. Support fp16 and
   bf16 source dtypes (bf16 → fp32 upcast is exact). mmap the file; keep tensors as
   views where possible (note: bf16 weights all convert, so expect ~7 GB resident
   for a 1.5B model during Phase 1 — fine on the Mac). Single-file only: a sharded
   checkpoint (index.json present) is rejected with a clear error.
2. **Config load.** Parse config.json for: hidden size, layer count, head counts
   (Q heads vs KV heads — GQA), head dim, intermediate size, RMSNorm eps, RoPE theta,
   vocab size, tied embeddings flag, max position embeddings, and (per family)
   attention-bias / qk-norm flags.
3. **BLAS wrapper (hard rule 8).** One wrapper over Accelerate `cblas_sgemm`. ALL
   matmul-shaped work goes through it: QKV/MLP/lm_head projections and per-head
   QK^T / PV products. Validated by a unit test diffing against a naive triple-loop
   matmul (the naive loop lives ONLY in that test, ~15 lines) on random odd-shaped,
   non-square, non-power-of-two matrices to flush transpose/leading-dimension bugs.
   The external HF fp32 oracle independently polices the wrapper — a botched sgemm
   call cannot self-validate.
4. **Module implementations** (plain Swift, readable over fast, fp32 accumulation):
   - Embedding lookup.
   - RMSNorm: x * rsqrt(mean(x²) + eps) * weight. Mean in fp32.
   - RoPE: rotate Q and K in pairs; match HF's implementation for the pinned family
     exactly (half-split rotation, theta from config).
   - Attention: incremental decode is Phase 2; here, full causal attention over the
     whole sequence each step. GQA: repeat/interleave KV heads to match Q heads.
     Softmax in fp32 with max-subtraction. Family-specific: QKV bias or QK-norm.
   - SwiGLU MLP: down( silu(gate(x)) * up(x) ).
   - Final norm → logits (respect tied embeddings if config says so).
5. **Decode loop.** Greedy argmax; temperature optional. Stop on EOS or max tokens.

### Correctness harness

**Reference generator (primary): HF transformers, CPU, torch_dtype=float32.**
Both sides load identical checkpoint bits and upcast exactly, so the only legitimate
divergence is summation order (~1e-4 absolute on logits of magnitude ~20).
**Tolerance: max |Δlogit| ≤ 1e-3 at full-vocab checkpoints. There is NO loosening
escape hatch — pressure to loosen is a bug signal (hard rule 6).**

mlx-lm is a secondary, loose-tolerance ecosystem sanity check (argmax-level agreement
with what the MLX world produces for this checkpoint); it returns as the quality-gate
comparator in Phase 3.

**Fixture prompt set** (5 prompts, tests/fixtures): short English; longer
multi-sentence; code snippet; non-ASCII text; chat-template formatted.

**Fixture contents, per prompt** (all logit data stored fp32 — fp16's ~0.016 spacing
at logit magnitude would quantize the oracle coarser than the tolerance):
- Full-vocab logits at steps {0, 1, mid, last} of a 50-step greedy generation
  (late checkpoints catch position-dependent RoPE/cache bugs; Phase 2's cached
  attention diffs against this same set).
- Per-step scalar fingerprints: logsumexp, mean, std of the full logit vector
  (~12 B/step) — every step checked at some resolution; broad tail drift moves these.
- Per-step top-64 logit values + indices (ranking-drift visibility between checkpoints).
- Per-step top-1-vs-top-2 margin, and the full argmax token sequence.
- **Tie-aware argmax assertion:** steps where the reference margin is below a small
  epsilon are exempt from exact top-1 match (assert our top-1 ∈ reference top-2
  instead). Everything else: exact match, all 50 steps. This kills the
  near-tie-flip false-failure mode without weakening the test.
- Total ≈ 13 MB, plain git — no LFS.

**Per-module activation fixtures (PRIMARY, not conditional).** For fixture prompt #1,
the generator script also dumps: embeddings output; layer-0 pre-attention (post-norm)
input; layer-0 attention output; layer-0 post-MLP output; last-layer output; final-norm
output. Each Swift module gets an XCTest against its slice BEFORE the full pipeline is
wired. These same fixtures serve as kernel-diff oracles in Phases 2–5.

**Tokenizer equivalence.** The generator script dumps Python tokenizer ids for all 5
prompts; an XCTest asserts swift-transformers produces identical ids. On disagreement:
log it, match Python.

**Regeneration = forensics.** The generator script (tools/, Python) is checked in with
pinned dependency versions (transformers, mlx-lm, torch) and the pinned model revision.
One documented command regenerates the full uncompressed oracle set locally when a
failure needs the complete picture. Regenerating with unpinned versions is a bug.

### Enumerated edge-case tests (all required, same change as the code)

Parser: (1) bf16→fp32 against known hand-built bytes; (2) truncated file → clear
error; (3) malformed JSON header → clear error; (4) sharded checkpoint → loud reject;
(5) out-of-bounds / overlapping tensor offsets → clear error.
Config: (6) missing required key → clear error naming the key; (7) tied-embeddings
flag exercised both ways.
Decode: (8) EOS stop; (9) max-tokens stop; (10) temperature path seeded + tested
IF built.
Integration: (11) tokenizer-id equivalence (above); CLI: empty prompt → usage error;
prompt exceeding 4K context → clear error, not a crash.

### Explicitly out of scope for Phase 1

KV cache (Phase 2), quantization (Phase 3), any Metal code in the forward pass, chat
templating beyond what the fixture prompts need, performance beyond the BLAS wrapper.

### Exit criteria

- CLI: `qwen-metal-cli generate --prompt "..."` produces coherent text.
- Logit-match suite passes on all 5 fixture prompts (checkpoints, fingerprints,
  top-64, tie-aware argmax) at ≤ 1e-3.
- All per-module activation tests pass.
- All enumerated edge-case tests pass.
- DECISIONS.md updated: tokenizer observations, any family-specific implementation
  notes. (Model repo/revision was already pinned at Phase 0a.)
