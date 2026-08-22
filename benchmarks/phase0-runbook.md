# Phase 0a runbook — on-device baselines (P0A-1, James)

Everything agent-preparable for the manual device session, in run order.
Protocol authority: PLAN.md benchmark protocol + parity pins; prompts:
`benchmarks/prompts/README.md`. All Phase 0 rows are **PROVISIONAL**.
Record every row with: date, device (iPhone 15 Pro), iOS version, battery
health %, starting temperature, engine + exact checkpoint revision, prompt
name + per-engine prompt token count, template feeding mode (raw vs rendered),
cold/warm, burst/sustained, toolchain (Xcode 26.6 build 17F113).

## 1. Triad bandwidth (roofline denominator — do this first)

Follow `benchmarks/device-shell/README.md` (scratch app, decision 1A).
Outputs: sustained GB/s + spread → results.md row + DECISIONS.md OPEN item.

## 2. MLX baseline (LLMEval)

- Cloned 2026-08-22: `~/Projects/mlx-swift-examples` @ `378f2449` (record
  this sha in each MLX row). LLM libraries come from the `mlx-swift-lm` SPM
  dependency — record the `mlx-swift-lm` AND `mlx-swift` versions from
  Package.resolved AS BUILT (Xcode may re-resolve minors on open; the fresh
  clone resolved mlx-swift-lm 3.31.3/`1c05248b`, and a later resolution moved
  mlx-swift 0.31.3→0.31.6 — whatever is in Package.resolved at build time is
  the truth for the row).
- **Exact overrides (verified against the source, 2026-08-22), both in
  `Applications/LLMEval/ViewModels/LLMEvaluator.swift`:**
  1. Line 50 — replace the registry default with the PIN-1 revision-pinned
     checkpoint (`ModelConfiguration(id:revision:)` supports this;
     `LLMRegistry.qwen3_1_7b_4bit` exists but tracks `main`):
     `var modelConfiguration = ModelConfiguration(id: "mlx-community/Qwen3-1.7B-4bit", revision: "3b1b1768f8f8cf8351c712464f906e86c2b8269e")`
  2. Line 54 — greedy: `temperature: 0.6` → `temperature: 0.0`.
  No thinking edit needed: `enableThinking = false` is already the default
  (line 20) and is passed via `additionalContext["enable_thinking"]`
  (line 239) — still verify zero `<think>` content in outputs.
  `maxTokens = 2048` default is fine (≥ 600 needed for decode rows).
- Residual provenance sanity check: confirm the app-downloaded config matches
  the pinned revision (file hashes or config field spot-check).
- Rows: burst decode (canonical window 128–512, `decode-essay`), prefill
  (`prefill-summarize`), sustained (regenerate-loop ≥5 min, `decode-essay`),
  cold + warm variants, phys_footprint from the Xcode memory gauge.
- Verify ZERO `<think>` content in outputs (row-invalidating).
- **Absolute target commit:** after the sustained/burst decode rows, compute
  target = 0.75 × MLX measured decode tok/s (canonical window) and commit it
  to DECISIONS.md (closes an OPEN item).

## 3. llama.cpp baseline

- Checkpoint: locally converted GGUF (decision 2026-08-22):
  `models/qwen3-1.7b-70d244cc-Q4_K_M.gguf`, converted from the pinned base
  checkpoint at pinned llama.cpp `b9999` (`47c78692`) — provenance + sha256 in
  DECISIONS.md. Note "Q4_K_M (k-quant) vs 4-bit grouped affine" format
  difference in the results table (PRD 0a.3).
- Runner: llama.cpp's iOS example app (`examples/llama.swiftui` in the SAME
  pinned clone at `~/Projects/llama.cpp`), built in Release with Increased
  Memory Limit; copy the GGUF into the app bundle/documents per its README.
- Feeding mode: the example app takes bare completion text → use
  `benchmarks/prompts/rendered/*.rendered.txt` verbatim.
- Sampling: greedy (temp 0 / top-k 1) — do NOT adopt the official README's
  presence_penalty 1.5 recommendation (parity pins forbid per-engine sampler
  differences; note in the row that repetition may appear under greedy).
- Same row matrix as MLX. phys_footprint from the Xcode gauge (attach via
  Debug → Attach or run from Xcode).
- Timebox: one day (PRD risk table). If the example app fights, a thin wrapper
  is acceptable; log what was changed.
- **Mac-side sanity check (verified 2026-08-22):** use `llama-completion`, NOT
  `llama-cli` — in the pinned build llama-cli is a chat TUI that re-applies the
  GGUF's embedded chat template on top of the rendered prompt (double
  templating) and ignores `-no-cnv`:
  `~/Projects/llama.cpp/build/bin/llama-completion -m models/qwen3-1.7b-70d244cc-Q4_K_M.gguf --temp 0 -n 60 --no-display-prompt -f benchmarks/prompts/rendered/decode-essay.rendered.txt`
  Verified output: coherent, zero `<think>` tags; Mac M2 Pro reference:
  ~1153 tok/s prompt, ~117 tok/s generation.
- **Known cross-engine variances (report, don't fix):** llama.cpp tokenizes
  the rendered decode-essay prompt to 92 tokens vs HF's 84 (why per-engine
  token counts are mandatory); under Q4_K_M the greedy continuation opens in a
  deliberative register ("Okay, the user wants…") unlike MLX-4bit/fp32 — no
  `<think>` tags, rows remain valid.

## 4. Energy method dry-run (one validated cycle — MLX)

Battery-delta protocol (DECISIONS.md 2026-08-20 item 4), operationalized
2026-08-22 with the LLMEval **Loop** button (toolbar; added for this — shows
cumulative `LOOP N gens | M tokens | Ts wall | first/last t/s`):

**Numbers for this device** (iPhone 15 Pro ≈ 12.6 Wh rated × 85% health ≈
10.7 Wh effective ⇒ 10% SoC ≈ 3.9 kJ; at 4–6 W the 80→70% burn takes
**~12–18 min** — over the ≥8–10% floor by construction).

1. Setup: airplane mode ON, brightness MINIMUM, Background App Refresh off,
   unplugged, rested to ambient, charge to ≥80%, Metal API Validation OFF.
   Run attached to Xcode ONCE first for the validation-off MLX spot-check
   burst (§5 ride-along) + phys_footprint glance, then relaunch DETACHED
   (from the home screen) for the energy cycle itself — the debugger costs
   energy.
2. At exactly 80% SoC (Settings → Battery): load `decode-essay` prompt, tap
   **Loop**, note wall-clock start.
3. At exactly 70% SoC: tap **Stop Loop**. Record: wall time, the LOOP line
   (generations, total tokens, first/last t/s).
4. **Idle baseline**: same screen-on state, same brightness, airplane mode,
   app open but NOT generating, same wall duration; record SoC drop.
5. Math: energy_J = (10% − idle_SoC_drop%) × 10.7 Wh × 36; J/token =
   energy_J ÷ loop total tokens; implied W = energy_J ÷ run seconds must
   land in 3–9 W or the cycle is invalid. One valid cycle validates the
   method (≥3-repeat rounds are Phase 6); record method + result in
   DECISIONS.md (closes the last OPEN item).

## 5. Row templates (append to benchmarks/results.md — never overwrite)

```markdown
### <date> — Phase 0a provisional baselines (iPhone 15 Pro, iOS <ver>, battery health <..>%, start temp <ambient/°C>, Xcode 26.6 17F113)

| engine | checkpoint | prompt (tokens) | mode | cold/warm | prefill tok/s | decode tok/s (128–512) | phys_footprint | notes |
|---|---|---|---|---|---|---|---|---|
| MLX LLMEval @ <sha> | Qwen3-1.7B-4bit @ 3b1b1768 | decode-essay (<n>) | burst | cold | — | <..> | <..> MB | PROVISIONAL; greedy; non-thinking via <mechanism> |
| ... | | | | | | | | |

| engine | sustained decode tok/s | run length | energy J/token (validated?) | implied W | notes |
|---|---|---|---|---|---|
| MLX | <..> | <min> | <..> | <..> | PROVISIONAL; regenerate-loop |

Bandwidth: sustained (median) <..> GB/s, spread <..>–<..>, dispatch overhead <..> ms/iter (triad, pinned protocol, Release).
```

## 6. Close-out (agent can do this part when numbers exist)

- DECISIONS.md: measured GB/s; absolute target (0.75 × MLX); energy method
  verdict. PLAN.md provisional figures updated to derive from measured GB/s.
- PRIORITIES.yaml: P0A-1 → done; SPEC-P2 stays blocked on P1-5 only.
- PRD-phase-0 acceptance checklist walked item by item.
