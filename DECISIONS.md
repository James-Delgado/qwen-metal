# DECISIONS.md — dated record of decisions and measurements

Append-only. Every session that decides or measures something adds a dated entry:
what was decided/measured, and why. See CLAUDE.md session discipline.

---

## 2026-08-20 — Plan solidified via full engineering review (20 findings folded)

Source: /plan-eng-review over the Downloads drafts; 8 first-pass findings + 12
outside-voice (cross-model) findings, all accepted. Load-bearing decisions:

1. **safetensors parser: hand-written, not swift-safetensors.** ~150 lines, mmap-based,
   single-file only. Raw pointer/mmap control matters for iOS memory accounting;
   the library's zero-copy path returns Core ML types the engine doesn't use.
2. **KV cache split.** Minimal preallocated cache (append per step, naive unfused
   attention over the cache) lands in Phase 2 so decode is incremental from the first
   on-device build. Phase 4 keeps the hard parts: fused GQA SDPA, norm/RoPE folding,
   dispatch reduction. Phase 3 gains a standalone fused dequant-matvec microbenchmark
   (achieved GB/s vs measured bandwidth, fixed short context).
3. **Layered quant oracle (Phase 3).** Exact match required at the dequant-tile layer
   (deterministic; where nibble-order/group-boundary bugs live). Matvec layer: ~1e-3
   relative tolerance vs CPU-quant reference, fp32 accumulation both sides (reduction
   order is the only difference). Adversarial packing fixtures: uneven group
   boundaries, extreme scales, all-zero groups, negative-heavy groups. Quality gate:
   top-1 agreement + mean KL vs mlx-lm 4-bit on fixture prompts + fixed-slice
   perplexity — graded "same band as MLX-4bit-vs-fp16," not exact match.
4. **Energy metric: battery-delta joules, sustained runs only.** ΔSoC ×
   health-scaled capacity; ≥8–10% SoC burn/run; idle-baseline subtraction; 80→70%
   band; ≥3 repeats, mean ± spread; 3–9 W plausibility anchor. Instruments Energy
   Log demoted to labeled unitless indicator. Burst energy undefined, not reported.
5. **Logit fixtures.** fp32 storage (fp16 spacing ~0.016 exceeds tolerance);
   full-vocab checkpoints at steps {0,1,mid,last}; per-step logsumexp/mean/std
   fingerprints; per-step top-64; tie-aware argmax via recorded top-1-vs-top-2
   margins. ~13 MB plain git. Regeneration script checked in, versions pinned.
6. **Reference oracle: HF transformers CPU fp32, tolerance ≤ 1e-3, no loosening
   escape hatch.** mlx-lm demoted to secondary ecosystem check; returns as Phase 3
   quality-gate comparator.
7. **Per-module activation fixtures are primary** — each CPU module unit-tested in
   isolation before pipeline wiring; same fixtures serve Phase 2–5 kernel diffs.
8. **CPU reference uses Accelerate cblas_sgemm** for ALL matmul-shaped work via one
   wrapper (validated against a test-only triple-loop on odd shapes). Scalar suite
   would take 5–15 h and stop being run. Elementwise logic stays hand-rolled.
9. **Outside-voice batch (all 12 accepted):** absolute success metric
   (≥ 0.75 × MLX measured, same session, canonical window 128–512); on-device
   bandwidth microbench is the roofline denominator; regenerate-loop sustained
   protocol; quantize embeddings + lm_head (fp16 head alone blows the roofline);
   Phase 2 fp16 gate pre-committed; model pinned BEFORE Phase 0a (2.5-vs-3 is an
   architecture fork: QKV bias vs QK-norm); Phase 0 baselines PROVISIONAL, Phase 6
   re-runs all engines same-session/same-OS; energy calendar cost budgeted; mmap vs
   wired-copy sustained-stability bench in Phase 2; dual GPU+wall timing from day
   one; numeric gates pre-committed per phase; cross-engine parity pins (greedy,
   pinned prompts, per-engine token counts, phys_footprint).

## OPEN — to be pinned before Phase 0a begins

- [ ] Exact model repo + revision (Qwen 2.5 vs Qwen 3 family fork decides module list).
- [ ] mlx-community 4-bit checkpoint provenance verified against the pinned base
      (or convert from pinned fp16 via mlx_lm.convert).
- [ ] Measured iPhone DRAM bandwidth (GB/s) — from the Phase 0 triad kernel.
- [ ] Absolute decode target = 0.75 × MLX measured decode tok/s (canonical window).
- [ ] Energy method validation result from the Phase 0 dry-run.

## 2026-08-20 — Backlog expanded to full-project DAG; review record archived

- Full deliberation record of the solidification review (all 20 findings verbatim,
  resolutions, coverage/failure-mode audits, per-phase spec obligations) archived
  at docs/reviews/2026-08-20-eng-review.md. DECISIONS.md stays the operative
  summary; the archive is the context source for future SPEC-P2..P6 sessions.
- docs/PRIORITIES.yaml now carries the whole project: concrete tasks for Phase 0
  (PIN-1, SETUP-1, P0B-1..4, P0A-1) and Phase 1 (P1-1..5, whose spec exists), and
  for Phases 2-6 only SPEC-Pn (write the spec, pre-commit gates) + Pn-EXEC
  (milestone placeholder, replaced by real tasks when the spec lands). This makes
  the just-in-time spec rule an enforceable DAG edge instead of convention.
- AUDIT-1 retargeted to depend on P1-5 (audit needs code to exist).

## 2026-08-21 — Architecture PDF generator incorporated (docs/generator/)

- Adopted the browser-agent's system-design PDF generator: make_diagrams.py
  (Figures 1-7) + build_pdf.py -> docs/architecture.pdf. Paths made repo-relative.
- Content drift fixed before first commit (v1.1): absolute success metric wording
  (>=0.75x MLX measured, canonical window), phys_footprint as the sole memory
  metric, "MLX ~61 tok/s" relabeled published/provisional, finding count corrected
  to 20, parity pins + regenerate-loop added to the methodology section.
- Its CLAUDE.md was NOT adopted (it predated the v2 hard rules); only its
  "Architecture document upkeep" section (regeneration triggers + anti-drift
  clause) was merged into ours. Root README.md added.
