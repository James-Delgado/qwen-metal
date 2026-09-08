# TODOS.md — deferred work with context

## CI: GitHub Actions macOS arm64 runner for the engine test suite

- **What:** Workflow running the XCTest suite (module tests, parser/config edge tests,
  Metal toy-kernel tests) on a macOS arm64 runner on every push.
- **Why:** The oracle chain only protects the project if it runs without anyone
  remembering to run it. GitHub's M-series macOS runners have working Metal, so even
  GPU-diff tests can run in CI.
- **Pros:** Regressions caught at commit time across all phases; the fast-oracle
  investment (Accelerate wrapper) pays out automatically.
- **Cons:** macOS runner minutes are ~10× Linux pricing. The full logit test needs the
  ~3 GB checkpoint (not in git), so CI needs either weight caching or a weights-free
  subset.
- **Context (as of 2026-08-20):** Repo not yet scaffolded. Likely shape: CI runs
  everything except the full end-to-end logit test — per-module activation fixtures
  ARE checked into git (~13 MB) and cover most of the correctness surface; full
  end-to-end stays a local target. Decided as "capture, revisit after Phase 1" in the
  2026-08-20 eng review (D10).
- **Depends on / blocked by:** Phase 1 test suite existing; repo scaffold (/project-init).

## Post-Phase-6 optimization campaign: speed + quality + memory (three axes)

- **What:** A chartered optimization campaign after Phase 6 that pushes all
  three product-relevant axes as far as they can go: decode tok/s up, output
  quality up, memory footprint down. North star recorded in DECISIONS.md
  2026-09-07 (decided by James): the learnings feed real iPhone local-LLM
  products/apps.
- **Axes and candidate levers:**
  - **Speed:** the measured leftovers from the P4-EXEC decode decomposition
    (dequant-matvec kernel internals — still naive as of Phase 3 exit —
    deeper folds, unexercised P4-3 levers), quantized KV cache (also a
    memory lever), speculative decoding with a 0.5B draft (the PLAN stretch
    item).
  - **Quality:** the quant-quality section below (beat mlx-4bit; GPTQ-style
    error-compensated rounding first).
  - **Memory:** quantized KV cache (fp16 448 MiB today), further footprint
    work surfaced by the Phase 6 rows.
- **Why not now:** PLAN v2's charter is satisficing-plus-explaining with
  unmovable pre-committed targets; mid-phase goal drift was considered and
  rejected (OV#1 lesson). Phase 6's head-to-head + the P4-EXEC decomposition
  produce the *measured* target menu a maximization campaign needs —
  optimizing against measured gaps beats optimizing speculatively.
- **Depends on / blocked by:** Phase 6 exit; a scope decision + spec + gates
  recorded in DECISIONS.md per the just-in-time pattern (each axis re-enters
  scope only through a recorded charter, per the non-goals discipline).

## Quant-quality optimization phase: beat mlx-4bit, not just match it

> 2026-09-07: folded into the post-Phase-6 optimization campaign above as
> its quality axis (north-star entry in DECISIONS.md). Details below stand.

- **What:** A future optimization phase implementing quality mechanisms from
  the DECISIONS.md 2026-08-29 suggestion entry — GPTQ-style
  error-compensated rounding first (q4g64 file format unchanged; only the
  packer's code selection), then possibly AWQ-style scaling / codebooks /
  g32 (each a schema re-pin).
- **Why:** Decided by James 2026-08-30 (QR-1 entry): A1 zero-point
  alignment targets parity with mlx-4bit; the explicit later goal is to
  perform BETTER than mlx. The quality band (P3-3 machinery: band.json,
  ref-logits artifact, QuantQualityGateTests) is the ready-made
  measurement harness for any such attempt.
- **Cons/context:** Still on the PLAN.md non-goals list until a phase is
  chartered — needs a scope decision in DECISIONS.md, and GPTQ-style
  selection breaks the "nearest-to-source" round-trip bound + determinism
  pins (re-derivation required; see the 2026-08-29 suggestion entry).
- **Depends on / blocked by:** Phase 3 exit (packed pipeline + band green);
  realistically slots after Phase 6 or as part of a chartered follow-on.
