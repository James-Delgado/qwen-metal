# Code Audit — qwen-metal

**Date:** 2026-08-23 (UTC) · **Task:** AUDIT-1 · **Scope:** whole repo (Sources/, tests/, tools/, benchmarks/; excl. .venv/.build/models)
**Depth:** full (~5,500 LOC, 46 files) · **Lens:** MLE + general (auto-detected: transformers/mlx oracle toolchain, fixture-based eval chain)

**Method:** architecture map (read-only explorer) → 4 parallel specialist reviewers fed the map
(`ecc:swift-reviewer`, `ecc:silent-failure-hunter`, `ecc:mle-reviewer`, `ecc:python-reviewer`) →
cross-reviewer dedup → adversarial verification (independent skeptics prompted to refute,
default-refuted-if-uncertain; 3 votes on CRITICAL/HIGH, 1 vote on MEDIUM/LOW).
All 4 reviewers landed; 22/22 agents completed. **12 candidates → 5 confirmed, 7 refuted.**
Only confirmed findings appear below; refuted candidates are listed in the appendix for audit honesty.

---

## Design summary (audit input)

Phase 0 (toy Metal kernels + dual-timing harness + baselines) and Phase 1 (CPU reference
engine) are exited. The engine is a Swift package (`QwenMetalEngine`) with a thin CLI:

- **Load path:** `ModelDirectory` → `ModelConfig` (typed config.json, family flags) →
  `SafetensorsFile` (hand-written mmap parser, header bounds/overlap validation, Accelerate
  fp16/bf16→fp32 upcast) → `QwenModel.init` (fp32 weights; tied lm_head).
- **Forward:** `Embedding` → 28× pre-norm blocks (`Attention` with QK-norm→RoPE→causal
  softmax; `SwiGLUMLP`) → final `RMSNorm` → lm_head. All matmul-shaped work through the
  single `BLAS.sgemm` wrapper (hard rule 8); elementwise hand-rolled fp32.
- **Decode:** `TextTokenizer` (swift-transformers 1.3.3 exact) → `DecodeLoop` (greedy,
  first-index tie-break, full re-forward per step — KV cache is Phase 2) → EOS/max/context stops.
- **Oracle chain:** pinned HF fp32 dumps (`tools/dump_reference.py`) → committed fixtures →
  `ActivationFixtureTests` (per-module) + `LogitMatchSuiteTests` (5 prompts × 50 teacher-forced
  steps, pre-committed gates) + `TokenizerEquivalenceTests`.
- **Metal (Phase 0 only):** `MetalContext.timedDispatch` dual timing; saxpy / naive fp16
  matmul / STREAM triad kernels, each with GPU-vs-CPU tests at pre-committed gates.

Design verdict: conventions (pre-committed gates, mmap-only load, single sgemm path,
per-module throwing error enums, scope cuts) are enforced consistently across the tree.
Every confirmed defect below is a gap **inside** the validation layer those conventions
mandate, not a violation of the architecture itself.

---

## Confirmed findings (severity-ordered)

### F1 · HIGH · `Sources/QwenMetalEngine/IO/ModelConfig.swift:176` — `positiveInt()` boundary bug admits 2^63, later traps instead of throwing

**Claim.** The guard `asDouble <= Double(Int.max)` is inclusive of one out-of-range value:
`Double(Int.max)` rounds **up** to 2^63 (9223372036854775808), so a config.json integer of
exactly 2^63 passes every clause; `NSNumber.intValue` then wraps to **`Int.min`** (verified on
this platform — the original reviewer said Int.max; skeptics corrected the mechanism), and the
negative value flows into `QwenModel.init`.

**Verified failure (reproduced end-to-end by 2 of 3 skeptics, upheld 3–0):**
`num_hidden_layers = 9223372036854775808` → `numHiddenLayers = Int.min` → `for layer in
0..<config.numHiddenLayers` traps ("Range requires lowerBound <= upperBound", SIGTRAP) at
`QwenModel.swift:53`; `num_attention_heads = 2^63` passes the `% 8 == 0` divisibility guard
(Int.min % 8 == 0) and traps on `numHeads * headDim` overflow at `QwenModel.swift:60`. Both
reachable from `qwen-metal-cli generate --model-dir` with a doctored config.json — a hard crash
on external input, violating the throw-never-crash convention. (Fields like `vocab_size` /
`max_position_embeddings` are shielded by downstream shape/positivity guards that throw cleanly.)

**Fix.** Replace the boundary test with strict `asDouble < 9223372036854775808.0 /* 2^63 */`
(or validate via `Int(exactly:)` on the NSNumber without a Double round-trip), and add the
missing bounds edge-case tests to `ModelConfigTests` (none of `positiveInt`'s overflow edges
are currently tested). → task **CFG-1**

### F2 · HIGH · `Sources/QwenMetalCLI/GenerateCommand.swift:77` (root: `ModelConfig.swift:119`) — decode stop set misses the model's second real stop token (`generation_config.json` never read)

**Claim.** The pinned checkpoint's `generation_config.json` — which HF `generate()` itself
consults for stopping — lists `eos_token_id: [151645, 151643]`. The engine's stop set is built
only from config.json (scalar 151645) ∪ tokenizer_config.json eos_token (also 151645), so
`<|endoftext|>` (151643) never stops decode. `grep -rn generation_config Sources tools docs`
returns nothing; the DECISIONS.md P1-5 entry cross-checked only tokenizer-vs-config.json.

**Verified failure (upheld 3–0):** on completion-style prompts (3 of the 5 pinned fixture
prompts), if greedy continuation emits 151643 to end a response — common for Qwen3 in raw
completion mode — `DecodeLoop.generate` (DecodeLoop.swift:90) misses it and re-forwards until
max-tokens/context fill, silently appending post-EOS garbage (`skipSpecialTokens: true` hides
the token itself from output, so the miss is invisible). The teacher-forced logit suite
structurally cannot catch free-running stop behavior; no test covers the [151645, 151643] pair.
`ModelConfig.intList` already parses list-form ids — the capability exists, fed from the wrong file.

**Fix.** Read `generation_config.json` when present next to config.json and union its
`eos_token_id` list into the stop set (or record an explicit DECISIONS.md scope decision);
add a `DecodeLoopTests` regression with the real [151645, 151643] pair via the scripted
logits source. → task **EOS-1**

### F3 · MEDIUM · `Sources/QwenMetalEngine/IO/ModelConfig.swift:149` — `double()` accepts non-finite/negative `rms_norm_eps` / `rope_theta`, yielding silent NaN logits

**Claim.** `double(_:_:)` returns `value.doubleValue` with no finiteness/sign check, unlike
every integer field (range-checked `positiveInt`). Verified (upheld 1–0): a negative
`rms_norm_eps` makes the radicand at `RMSNorm.swift:30` negative → `Float.squareRoot()` returns
NaN (no trap); a negative `rope_theta` yields NaN unconditionally via `powf(negative,
fractional)` at `RoPE.swift:29` (theta is the one RoPE.init parameter with no guard). NaN
propagates through every block to the logits; `Argmax.firstIndex`'s strict `>` scan then
deterministically selects index 0 — the CLI runs to completion printing garbage with zero
diagnostics. **Fix.** Guard `> 0 && .isFinite` for both fields, throwing
`ModelConfigError.invalidValue`; add edge-case tests. → task **CFG-1**

### F4 · MEDIUM · `Sources/QwenMetalEngine/IO/ModelConfig.swift:158` — `intList` (eos_token_id) omits the upper-bound check its sibling `positiveInt` performs

**Claim.** `intList`'s per-element guard checks integrality and non-negativity only. Verified
empirically (upheld 1–0): `"eos_token_id": 1e20` passes and `NSNumber.intValue` silently
returns `Int.max`; integer-literal `100000000000000000000` (NSDecimalNumber) returns wraparound
garbage `7766279631452241920` — neither throws. The garbage id flows into the DecodeLoop stop
set via GenerateCommand. Mitigation (severity nuance, not refutation): the tokenizer's own
eosTokenId is unioned in, so EOS stopping usually still works. **Fix.** Apply the same
`<= Double(Int.max)`-style bound (post-F1-fix form) in `intList`; test out-of-range values.
→ task **CFG-1**

### F5 · LOW · `tests/QwenMetalEngineTests/TokenizerEquivalenceTests.swift:23` — tokenizer artifacts have no programmatic content pin (checkpoint has `source_revision`; tokenizer files have prose-only hashes)

**Claim.** `SharedCheckpoint.model()` hard-fails on `source_revision` mismatch, but
`TokenizerEquivalenceTests.setUpWithError` only checks file existence for
tokenizer.json/tokenizer_config.json. The sha256 hashes DECISIONS.md records by hand
(aeb13307…, d5d09f07…) appear nowhere in tests/ or Sources/ (verified 1–0). A regenerated or
hand-edited models/ tokenizer is caught only if ids happen to differ on the 5 short pinned
prompts — no lineage guarantee mirroring the checkpoint's. **Fix.** Verify sha256 of the local
tokenizer files against the DECISIONS.md-recorded hashes in setUp, mirroring the
source_revision check. → task **TOK-1**

---

## Appendix — refuted candidates (killed by adversarial verification)

Recorded so a re-run doesn't resurface them without new evidence:

1. ~~HIGH `tools/dump_reference.py:79` non-atomic fixture writes~~ — refuted 3–0: mechanically true, but sha256s are computed from in-memory bytes and manifest.json is written only at the end of main(), so an interrupted run leaves a stale/absent manifest whose sha256/byte_len mismatch the truncated blob — `test_every_manifest_blob_matches_on_disk` (tests/test_fixtures.py:78-97) is a purpose-built backstop that fails loudly on exactly this case.
2. ~~MEDIUM `DecodeLoop.swift:37` precondition() instead of throw~~ — refuted: both trap paths are unreachable from external input — the only production caller passes `maxContext = min(4096, config.maxPositionEmbeddings)` already validated by `positiveInt()`, and empty logits cannot come from QwenModel (lm_head width is the checkpoint's validated vocab dimension); external inputs are caught at the config/safetensors boundary where the throwing enums live.
3. ~~MEDIUM `SafetensorsFile.swift:267` shard check skipped if directory listing fails~~ — refuted: the only production path reaches SafetensorsFile via `ModelDirectory.init(validating:)`, which runs a *propagating* `contentsOfDirectory` on the same directory immediately before — an unlistable directory fails loudly there, never reaching the `try?` in a failing state; ModelDirectory also rejects sharded layouts outright.
4. ~~MEDIUM `LogitMatchSuiteTests.swift:109` full-vocab diff at only 4 of 50 steps~~ — refuted: the 4-checkpoint protocol IS the pre-committed spec ("max |Δlogit| ≤ 1e-3 at full-vocab checkpoints", phase-0-1.md; checkpoint steps {0,1,24,49} pinned in DECISIONS.md 2026-08-20 with the 12.53 MB fixture budget) — the test implements the gate exactly, and the hypothesized escaping-bug class has no mechanism in this architecture.
5. ~~MEDIUM `tools/dump_reference.py:222` stale blobs not cleared on regeneration~~ — refuted: `test_no_orphan_files_on_disk` (tests/test_fixtures.py:100-113) computes on-disk minus manifest files and fails on any orphan, naming the files; the scenario requires skipping the test suite before commit, which the SOP (tests land with code) forecloses.
6. ~~MEDIUM `tests/test_fixtures.py:61` requirements.txt pin parser fragility~~ — refuted: traced through the assertions, every hypothesized malformed line fails CLOSED and LOUD (mismatch messages print the offending text; unparseable lines hit the "recorded in manifest but not pinned" assertion) — no path silently passes.
7. ~~LOW `SharedCheckpoint.swift:50` static var check-then-act race~~ — refuted: XCTest parallel testing shards classes across separate *processes*, never concurrent threads within one process — within a single xctest process execution is strictly serial by the framework's execution model, so two threads can never race the lazy init.
