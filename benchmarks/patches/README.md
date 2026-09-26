# External-harness patches (provenance of record)

The MLX and llama.cpp comparison rows are produced by the two upstream
example apps, patched for measurement-grade parity. The patches are
`git format-patch` output from the local clones and are the provenance
pins cited by every comparative row (PLAN.md benchmark protocol; Phase 6
spec D9). Upstream revisions and checkpoints never move here — a version
bump is a new PROVISIONAL baseline, not the head-to-head.

| Patch | Applies on | Branch / commits | Adds |
|---|---|---|---|
| `mlx-swift-examples-p0a1.patch` | mlx-swift-examples `378f2449` | `qwen-metal-p0a1`: 47c36a0, 992118b | PIN-1 checkpoint revision, greedy, sustained loop (Phase 0) |
| `mlx-swift-examples-p6.patch` | on top of the p0a1 patch | `qwen-metal-p0a1`: the P6-2 commit | Phase 6 parity bench: per-token timestamps → canonical-window rate, first-token prefill span, per-generation loop timeline, health/SoC fields, operator Stop that reports, rendered-input mode with the 852/84 parity signature, validation OFF scheme pin, text + JSON exports |
| `llama-swiftui-p0a1.patch` | llama.cpp `b9999` (`47c78692`) | `qwen-metal-p0a1`: 97e552a, bceddff | measurement-grade fixes, greedy, parse_special, bundled prompts, sustained loop (Phase 0) |
| `llama-swiftui-p6.patch` | on top of the p0a1 patch | `qwen-metal-p0a1`: the P6-2 commit | Phase 6 parity bench: per-token timestamps (post-sample, backend-synchronized), completion_init→first-sample prefill bracket, generated-token caps + n_ctx 4096 (the pinned 4K reset), per-generation loop timeline, health/SoC fields, operator Stop that reports, text + JSON exports |

## Applying

```bash
cd ~/Projects/mlx-swift-examples && git checkout 378f2449 -b qwen-metal-p0a1
git am /path/to/qwen-metal/benchmarks/patches/mlx-swift-examples-p0a1.patch
git am /path/to/qwen-metal/benchmarks/patches/mlx-swift-examples-p6.patch

cd ~/Projects/llama.cpp && git checkout 47c78692 -b qwen-metal-p0a1
git am /path/to/qwen-metal/benchmarks/patches/llama-swiftui-p0a1.patch
git am /path/to/qwen-metal/benchmarks/patches/llama-swiftui-p6.patch
```

Both clones already carry these commits locally (James's machine); the
files here are the archive. Build verification (agents; never run for
numbers): LLMEval on the macOS destination, llama.swiftui as
`generic/platform=iOS` with signing disabled.

## Using the Phase 6 panels (operator)

Both apps gain a "qwen-metal Phase 6 bench" panel. Metric definitions
are the qwen-metal engine's (phase-6.md D1), so the exports read like
ours:

- **Burst**: one generation of the selected pinned prompt, cap 640 new
  tokens. The export carries `prefill span:` (of record) and
  `decode rate: overall …, canonical window (tokens 128-512) …`.
  decode-essay ⇒ the decode row; prefill-summarize ⇒ the prefill row.
- **Loop / Stop (report)**: the energy / sustained regenerate loop —
  same prompt, reset on EOS or 4K context fill, until Stop. Stop ends the
  in-flight generation at its next token boundary and STILL reports
  (per-generation `gen N:` lines, cumulative tokens, cycle wall, the
  programmatic SoC at start/end). Type the Settings → Battery SoC at the
  stop mark and the battery health AFTER stopping — the export re-renders.
- **Feeding (MLX only)**: `rendered` = the pinned rendered string,
  tokenizer-direct, no template — the export prints the prompt token count
  against the parity target (852 / 84) and says MATCH or MISMATCH; a
  MISMATCH invalidates the mode for that row (fallback `template`, the
  Phase 0 mode, count recorded). llama.cpp always feeds the rendered
  string; its own counts are recorded (92 / 852 on record vs HF 84 / 852).
- **Copy export / Copy JSON / Copy output**: the text export (of record)
  and the JSON timeline (schema `qwen-metal-timeline/1-external`) go to
  `benchmarks/phase6/` per the Phase 6 runbook (P6-3); output is for the
  zero-`<think>` check.
- **Metal API validation**: OFF, scheme-pinned in both apps; detached
  home-screen launches carry no validation layer regardless.
- **Memory**: the in-app lines are cross-checks; the Xcode gauge on an
  attached footprint-only launch stays the value of record.
