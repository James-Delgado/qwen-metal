# Phase 6 runbook — the same-session head-to-head round (P6-4 / P6-5, James)

The Phase 0 runbook's successor (docs/phases/phase-6.md D3–D8; validity gates
in DECISIONS.md 2026-09-23 "Phase 6 gates pre-committed", veto window closed
2026-09-24). Two device sessions: the **speed session** (one day, ≈2–2.5 h)
and the **energy round** (two half-days, ≈7 h + a 1 h idle baseline). Every
number the writeup shows is regenerated from the exports you archive here —
you never type a metric into results.md from the phone; you paste the
analyzer's tables (§6). Agents never run any of this.

## 0. Before the first row (once)

- [ ] **Build of record:** check out the `phase6-build` tag; build
      QwenMetalApp in Release from it (Xcode 26.6 — the release Xcode, never
      the beta). The tag's correctness evidence (Tier-M/E, 250-step GPU-quant
      logit suite, free-run ×5) is recorded in DECISIONS.md (P6-3) — confirm
      the entry exists before any row. No engine change after the tag until
      the round closes (a change re-opens the round).
- [ ] **External harnesses:** `~/Projects/mlx-swift-examples` @ branch
      `qwen-metal-p0a1` with the P6-2 commit (patch archive
      `benchmarks/patches/mlx-swift-examples-p6.patch`), `~/Projects/llama.cpp`
      likewise (`llama-swiftui-p6.patch`). Build both in Release; the GGUF
      `models/qwen3-1.7b-70d244cc-Q4_K_M.gguf` in the llama app's documents,
      the pinned MLX checkpoint `mlx-community/Qwen3-1.7B-4bit @ 3b1b1768`
      downloaded. Upstream revisions and checkpoints do NOT move.
- [ ] **Metal API validation OFF** in all three schemes (already pinned;
      every export prints the line — confirm it reads OFF on the first row of
      each app). Detached launches carry no validation layer regardless.
- [ ] **Session facts** (typed once into `benchmarks/phase6/manifest.json`,
      schema in `benchmarks/phase6/README.md`): device, iOS build (Settings →
      General → About; re-check at the end of every session), Xcode build,
      the `phase6-build` commit sha, date.
- [ ] **Round marker strings** (typed identically into all three apps'
      `round` field; an export without it stays PROVISIONAL and is not a
      headline row): `phase6-speed-YYYY-MM-DD` for the speed session,
      `phase6-energy-YYYY-MM-DD` for the energy round (one marker per round
      even if it spans two days).
- [ ] **Battery health** (Settings → Battery → Battery Health, maximum
      capacity %) read at the start of each session and typed into every
      export's `battery health` field. It is NOT state of charge.
- [ ] Prompts: the bundled rendered forms only (never the clipboard).
      Expected prompt token counts — ours/MLX 84 (decode-essay) / 852
      (prefill-summarize); llama.cpp 92 / 852.

## 1. Operator fields (every export)

| field | where | value |
|---|---|---|
| round | all three apps | the round marker string (§0) |
| battery health | all three apps | maximum capacity %, typed (never SoC) |
| cold/warm | all three apps | `cold` = first launch of that app this session (weights from disk); `warm` otherwise |
| SoC at start / SoC at stop | energy mode only | Settings → Battery integer at the band marks; stop value typed AFTER Stop (the export re-renders) |
| iOS build, Xcode, footprints, idle baseline | manifest.json | typed once per session |

Rule: archive a row (Copy export + Copy JSON → files) BEFORE typing fields
for the next one — edits re-render the export currently shown.

## 2. Speed session (P6-4) — one calendar day, one iOS build, interleaved, bookended

Wall budget ≈2–2.5 h. Detached (home-screen) launches for every timed row;
rested to ambient at the start (unplugged, idle ≥ 15 min; no readout
exists). Keep the device state constant through the session (airplane
mode on or off — pick one, record it in the manifest note). Every timed row
is a fresh tap in an already-launched app except the cold rows.

### 2.1 Bandwidth pair (BW-1 handoff — weights-free, do it first)

1. Launch QwenMetalApp detached (no model load needed) → Benchmark →
   microbench → **read-only**, then **triad**. Copy each export to
   `speed/bandwidth-readonly-0.txt` / `speed/bandwidth-triad-0.txt`.
2. Rest ≥ 1 min, repeat twice more (`-1`, `-2`). Three interleaved pairs.
3. Each export must carry `correctness: … within pre-committed gate`; an
   export stamped `DIAGNOSTIC — non-pinned geometry` is never archived.
4. Quit the app fully (app switcher swipe) and rest ≥ 2 min. The weights
   file was never touched, so the next launch is still the cold row.

The analyzer reports the median run of each kernel and the paired
read-only ÷ triad ratio (the writeup §5/§7 number of record; the Mac read
≈0.97).

### 2.2 Cold rows + our opening bookend

5. **Ours, cold:** launch QwenMetalApp detached → load model (q4g64, mmap,
   fused, tiled C=512, query-tiled) → Benchmark → burst, decode-essay →
   Run. cold/warm = `cold`. Archive as `speed/ours-decode-cold-0`.
6. **Ours, bookend open:** same process, burst decode-essay again,
   cold/warm = `warm`. Archive as `speed/ours-bookend-open`.
7. **MLX, cold:** launch LLMEval detached → qwen-metal Phase 6 bench panel
   → feeding `rendered`, prompt decode-essay → Burst. `mlx-decode-cold-0`.
8. **llama.cpp, cold:** launch llama.swiftui detached → load the GGUF →
   P6 panel, decode-essay → P6 Burst. `llama-decode-cold-0`.

### 2.3 Warm bursts in rotation (decode, then prefill)

9. Decode-essay warm bursts in the order **ours → MLX → llama.cpp**, three
   times round (≈30–40 s each; ≥ 30 s pause between; no formal rest — the
   rotation spreads any drift across engines). Archive as
   `<engine>-decode-warm-{1,2,3}`.
10. Prefill-summarize bursts in the same rotation, three times round.
    Archive as `<engine>-prefill-warm-{0,1,2}`. (A prefill row stops at EOS
    below 512 tokens — its `decode rate … canonical window` reads n/a by
    design; the `prefill span:` line is the metric of record.)

### 2.4 Sustained 5-min loops (the short thermal rows)

11. **Ours:** Benchmark → sustained (decode-essay, 5 min, duration-bounded;
    Stop aborts WITHOUT a report — do not stop early). Archive
    `ours-sustained-0.txt` + `.json`.
12. Rest ≥ the loop's duration (≥ 5 min), unplugged.
13. **MLX:** P6 panel → Loop; at 5:00 tap **Stop (report)** → archive
    `mlx-sustained-0.txt` + `.json`. Rest ≥ 5 min.
14. **llama.cpp:** P6 Loop; at 5:00 **P6 Stop (report)** → archive
    `llama-sustained-0.txt` + `.json`. Rest ≥ 5 min.

### 2.5 Closing bookend, attribution, footprint

15. **Ours, bookend close:** warm burst decode-essay → `ours-bookend-close`.
    The bookend delta (close − open, window tok/s) is the within-session
    drift; any cross-engine claim smaller than it is "unresolved
    (drift-dominated)".
16. **Attribution (DIAGNOSTIC, ours, same build/session):** Benchmark →
    attribution → decode, then prefill (needs Prefill = tiled). Archive
    under `speed/attribution/` (ignored by the analyzer; quoted by writeup
    §6 as the re-exported decompositions).
17. **Footprint launches, attached (Xcode gauge = value of record), after
    all timed rows:** for each of ours (mmap), ours (wiredCopy — reload
    after the residency toggle), MLX, llama.cpp: run one decode-essay burst
    attached and read the memory gauge steady during decode. Type the four
    values into `manifest.json` → `sessions.speed.footprint_mb`. In-app
    footprint lines are cross-checks only. State the accounting asymmetry
    in the row notes (mmap'd file-backed weight pages excluded on ours-mmap
    and llama.cpp; ours-wired is the honest resident total).
18. Re-check the iOS build (About). If it changed during the session, the
    session is invalid — restart another day.

### 2.6 Per-row checklist (tick for every archived row)

- [ ] launched detached (bookends, bursts, loops); the export's validation
      line reads OFF.
- [ ] header reads `round: <marker>` (not PROVISIONAL); `cold/warm` and
      `battery health` typed (no `(record manually)` left).
- [ ] **ours:** engine line reads `weights q4g64, residency mmap, kernels
      fused, prefill tiled (C=512), attention queryTiled`; decode rows show a
      canonical-window rate; prefill rows show `prefill span: 852 tokens …`
      with no `[WARM PREFIX …]` tag.
- [ ] **MLX:** prompt line reads `parity target 84: MATCH` (decode) /
      `parity target 852: MATCH` (prefill). **MISMATCH ⇒ the rendered mode
      is invalid for this row**: switch feeding to `template`, redo the
      row, record the count the export prints, and note the fallback (D2) —
      the analyzer labels template-mode rows and the writeup names the delta.
- [ ] **llama.cpp:** prompt line reads `matches the llama.cpp count on
      record (92)` (decode) / `(852)` (prefill). `DIFFERS` ⇒ do not archive
      as a row; investigate the build/prompt bundle first.
- [ ] Copy output → zero `<think>` content (row-invalidating otherwise).
- [ ] Copy export → `speed/<name>.txt`; Copy JSON → `speed/<name>.json`
      (same stem). File names per `benchmarks/phase6/README.md`.

### 2.7 Tripwires (D7, walked from the analyzer output)

Run `tools/phase6_analyze.py` on the archive (§6): ours decode warm-burst
window median ≥ **24.0 tok/s** and prefill warm span median ≥ **135 tok/s**
must read PASS. A FAIL after harness-only changes signals breakage — stop
and record it; no constant moves.

## 3. Energy round (P6-5) — idle baseline + ≥3 cycles per engine, Latin square

Same iOS build as the speed session (recorded; a mid-round OS change
invalidates the incomplete round). Budget ≈45 min per cycle × 9 + ≈1 h idle.

### 3.1 Conditions (every cycle and the idle baseline)

Detached, **airplane mode ON, minimum brightness, Auto-Lock Never,
Background App Refresh off, unplugged**, rested to ambient (§3.3), validation
OFF, the app foregrounded on its bench screen with decode-essay selected.

### 3.2 Idle baseline (once per iOS build; ≥ 4% SoC drop — ≈ 1 h)

1. Conditions as above, QwenMetalApp launched detached, model loaded,
   Benchmark screen showing, **no generation**.
2. Note Settings → Battery SoC and the wall clock; wait until the SoC has
   dropped by **at least 4%** (≈ 1 h at ≈ 0.5 W); note SoC and wall clock.
3. Type into `manifest.json` → `sessions.energy.idle_baseline`:
   `health_percent`, `soc_start`, `soc_end`, `wall_seconds`, `ios_build`.
   Below 4% the analyzer refuses every energy headline — extend the wait.

### 3.3 One cycle

1. Charge to ≥ 81%, unplug, then **rest ≥ the duration of the preceding
   loaded run** (charging counts as load; rest starts when the charger
   comes off). First cycle of a day: rest ≥ 20 min unplugged.
2. Launch the engine's app detached, load the model, select decode-essay,
   type round marker + battery health; leave SoC-at-start empty for now.
3. Watch Settings → Battery. **At exactly 80%** switch to the app and tap
   Run (ours: mode `energy`; MLX/llama: P6 Loop). Note the wall clock.
4. Leave the device alone (≈ 17 min). Near 71%, watch Settings; **at
   exactly 70%** switch to the app and tap Stop (ours: Stop; MLX/llama:
   Stop (report)). The loop ends at the next token boundary and reports.
5. Type SoC at start `80`, SoC at stop `70` (or the value actually read,
   if you were late — the analyzer marks the cycle by the ≥ 8% floor and
   notes any deviation). Copy export → `energy/<engine>-energy-<n>.txt`;
   Copy JSON → `.json`. `<n>` counts cycles per engine from 0, invalid ones
   included (never reuse an index).
6. Check the export: `energy cycle: … ended by operator stop`, the
   `cumulative:` line, the `gen N:` timeline, both `state of charge` lines,
   health typed, round marker in the header, zero `<think>`.

### 3.4 Order (Latin square) and validity

Cycle order across the round: **(ours, MLX, llama.cpp), (MLX, llama.cpp,
ours), (llama.cpp, ours, MLX)** — each engine occupies each position once.
After each cycle, run the analyzer: a cycle is VALID only if the start mark
is 80, ≥ 8% SoC burned, and the implied gross watts land in 3–9 W. An
INVALID cycle is archived (it stays in the per-cycle table) and **repeated**
so each engine ends with ≥ 3 valid cycles. Cut rule (decided 2026-09-24): if
the calendar forces a cut, llama.cpp's cycles are dropped first, and a
partial engine (n < 3) is reported labeled "below the ≥3 pin — not a
comparative energy row", never averaged — the analyzer enforces this.

Instruments Energy Log, if captured, is "energy impact (relative,
unitless)" only — never a J/token.

## 4. Row templates (append to benchmarks/results.md — never overwrite)

Paste the numbers from `docs/writeup/tables.md` (analyzer output). Rows
inside a round carry the round marker instead of PROVISIONAL.

```markdown
## Phase 6 — same-session head-to-head, iPhone 15 Pro (P6-4, James)

### <date> — speed session, round `phase6-speed-<date>` (iOS <build>, Xcode 26.6 (17F113), battery health <n>%, build phase6-build @ <sha>, validation OFF, detached)

Order run: bandwidth pair ×3 → ours cold + bookend → MLX cold → llama.cpp cold → warm decode rotation ×3 → prefill rotation ×3 → 5-min loops (ours, MLX, llama.cpp; rests ≥ 5 min) → bookend close → attribution (DIAGNOSTIC) → footprint launches (attached).

| engine | harness / checkpoint | prompt (tokens) | cold window tok/s | warm window tok/s (n, values) | warm median | notes |
|---|---|---|---|---|---|---|
| qwen-metal q4g64 (mmap, fused, tiled C=512, query-tiled) | phase6-build @ <sha> | decode-essay (84) | <..> | <..> | **<..>** | bookends <open> → <close> (Δ <..>) |
| MLX LLMEval | 378f2449 + p0a1 + p6 / Qwen3-1.7B-4bit @ 3b1b1768 | decode-essay (84, rendered, MATCH) | <..> | <..> | **<..>** | |
| llama.swiftui | b9999 47c78692 + p0a1 + p6 / Q4_K_M GGUF | decode-essay (92) | <..> | <..> | **<..>** | |

| engine | prompt (tokens) | prefill span tok/s, warm (n, values) | warm median | notes |
|---|---|---|---|---|
| qwen-metal | prefill-summarize (852) | <..> | **<..>** | span GPU s <..> |
| MLX LLMEval | prefill-summarize (852, rendered, MATCH) | <..> | **<..>** | first token from the prompt forward (P6-2) |
| llama.swiftui | prefill-summarize (852) | <..> | **<..>** | completion_init bracket |

| engine | 5-min loop: generations | first window | last window | 2nd-half median | notes |
|---|---|---|---|---|---|
| … | | | | | |

| engine | phys_footprint (Xcode gauge) | notes |
|---|---|---|
| qwen-metal mmap / wiredCopy | <..> MB / <..> MB | file-backed pages excluded on mmap; wired = honest resident total |
| MLX | <..> MB | |
| llama.swiftui | <..> MB | n_ctx 4096 (P6-2); mmap accounting asymmetry |

Bandwidth pair (same build, same session): read-only median <..> GB/s, triad median <..> GB/s, paired ratio <..> / <..> / <..>.
Tripwires: decode ≥ 24.0 → PASS/FAIL (<..>); prefill ≥ 135 → PASS/FAIL (<..>). Success metric (PLAN formula): ours <..> ÷ MLX <..> = <..> ⇒ MET / NOT MET (29.4 constant alongside).
Exports: benchmarks/phase6/speed/. DECISIONS.md entry: <date>.

### <date(s)> — energy round, round `phase6-energy-<date>` (iOS <build> = speed session, battery health <n>%)

Idle baseline: <soc_start>→<soc_end>% over <s> s (≥ 4% ✓). Capacity basis: 453.6 J per 1% × health.

| engine | cycle | order pos. | SoC start→end | ΔSoC | cycle wall s | tokens | gross J | implied W | idle ΔSoC (scaled) | net J | net J/token | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| … | | | | | | | | | | | | |

| engine | n valid | mean net J/token | spread (min–max) | systematic ± | implied W (mean) | status |
|---|---|---|---|---|---|---|
| … | | | | | | headline / below the ≥3 pin |

Timelines (the sustained-thermal chart data): benchmarks/phase6/energy/*.json.
```

## 5. Rest and drift discipline (D6)

Rest to ambient between thermally loaded runs for at least the preceding
run's duration, unplugged; charging is load. The speed session's bookend
delta demotes any smaller cross-engine claim to "unresolved
(drift-dominated)". No temperature readout exists — the rest is procedural
and is recorded as such.

## 6. Archival and regeneration (after each session)

1. Files in place per `benchmarks/phase6/README.md`; `manifest.json`
   complete (iOS build re-checked; footprints; idle baseline).
2. `.venv/bin/python tools/phase6_analyze.py benchmarks/phase6 --out docs/writeup`
   → read `docs/writeup/tables.md` §7 (refusals) first: every refused row
   is either fixed at the source (a mis-typed field) or accepted as
   non-headline — never edited in the export.
3. Paste the tables into results.md (§4 templates); append the DECISIONS.md
   entry (validity walk, tripwire verdicts, bookend delta, anything
   decided); commit exports + manifest + results.md + DECISIONS.md together.
4. P6-5's entry additionally records the llama.cpp-energy decision if the
   cut rule was exercised.
