# Phase 6 Spec — benchmark writeup + final same-session measurement round

Written 2026-09-23 (SPEC-P6), from the Phase 5 results logged in
DECISIONS.md. Context sources: PLAN.md v2 (phase table row 6; the
benchmark protocol, energy protocol, and baseline staleness rule; the
success metric); docs/reviews/2026-08-20-eng-review.md Part 4 (the three
SPEC-P6 obligations: same-session/same-OS re-runs of all engines — OV#7;
energy calendar budget + the possible llama.cpp energy drop — OV#8; the
measurement-limitations section — Issue 3); DECISIONS.md 2026-08-22
(Phase 0 baselines, the energy dry-run, the validation-setting and
template-delta findings), 2026-09-05 (capacity-basis correction — the
binding obligation to re-pin the basis from battery health at run time and
record health and charge separately), 2026-09-07 (north star: quantify
remaining headroom per component), 2026-09-14 P4-EXEC and 2026-09-23
P5-EXEC (the decode and prefill decompositions this writeup reuses), and
2026-09-23 (James: the Phase 6 writeup headlines the PF-2B prefill number).
Numeric gates for this phase are committed in DECISIONS.md ("Phase 6 gates
pre-committed", 2026-09-23) — this file explains them; the DECISIONS entry
is the binding record.

---

## Purpose

Every comparative number the project has recorded so far is PROVISIONAL by
construction. The MLX and llama.cpp rows are from 2026-08-22 on iOS 26.5.2
(MLX with Metal API validation ON, app-reported overall decode rates, not
the canonical window); our rows are from 2026-09-04 … 2026-09-23 on iOS
26.6.1, detached, windowed. PLAN.md's staleness rule says what that means:
"Never compare rows recorded months apart. The publishable head-to-head
requires Phase 6 to re-run all three engines in one session, on one OS
build, interleaved." Phase 6 is that round, plus the writeup the project
is defined by ("a working engine plus a quantified writeup: prefill tok/s,
decode tok/s, peak memory, and energy/token for all three engines, with a
roofline analysis explaining the gaps").

What the engine brings into Phase 6 (all on record; nothing in this phase
changes the engine's kernels):

| Metric (iPhone 15 Pro, detached, validation OFF) | Ours | MLX (Phase 0, PROVISIONAL) | llama.cpp (Phase 0, PROVISIONAL) |
|---|---|---|---|
| Decode tok/s, warm burst, decode-essay | **31.67** window median (P4-11; 31.05 at P5-5B, 30.90 at PF-2B — unchanged since Phase 4) | 39.2 overall rate (validation ON; 39.6 OFF spot check) | 32.44 overall rate |
| Prefill tok/s, prefill-summarize (852 HF tokens) | **240.86** warm span median (PF-2B, query-tiled attention — the number of record per James 2026-09-23; 172.23 at the P5-5B gate walk) | ≈370 (862 app-side tokens ÷ 2.328 s TTFT) | ≈452 (852 tokens; TTFT minus one decode step) |
| Sustained decode (5-min regenerate loop) | 31.59 → ≈23.6 plateau (P4-11, −25%) | −9% over 17.6 min detached (40.43 → 36.78) | −45% over 17.6 min detached (32.81 → 18.17) |
| phys_footprint (Xcode gauge) | 573.2 MB mmap (P5-5); 1.43 GB wired = honest resident total (P3-7) | 1.02 GB | 307 MB (mmap accounting artifact — 1.19 GB GGUF file-backed) |
| Energy, net J/token (sustained, battery-delta, n=1 dry run, corrected basis) | not yet measured | ~0.122 | ~0.154 |

The committed success metric (PLAN.md): our decode tok/s ≥ 0.75 × MLX's
measured decode tok/s, **both measured in the same session on the same
device at the canonical window**. The Phase 0 planning constant derived
from it (29.4 = 0.75 × 39.2) is already exceeded (+7.7%, P4-EXEC), but the
metric as written has never been evaluated — the same-session MLX
windowed number does not exist yet. Phase 6 produces it and records the
verdict by the PLAN formula (D7).

Per the 2026-09-07 north-star binding, the writeup does not stop at the
head-to-head: it carries the per-component headroom decompositions (decode
from P4-EXEC, prefill from P5-EXEC), re-measured on the Phase 6 build in
the same session as the external engines, so the post-Phase-6 campaign
(CAMP-1) charters against measured gaps.

## Scope

**In:** (1) harness parity upgrades so all three engines report the SAME
metric definitions (canonical-window decode rate, first-token prefill
span, per-generation sustained timeline, battery health + state of charge
as separate fields, validation setting) — our app/engine (P6-1) and the
two external harnesses (P6-2, archived as patches); (2) the Phase 6
runbook + row templates + the analysis/plot tooling with its tests (P6-3);
(3) the speed session — all engines, one session, one iOS build,
interleaved with bookends: burst decode cold/warm, prefill, 5-min
sustained loops, phys_footprint (P6-4, James); (4) the energy rounds — ≥3
battery-delta cycles per engine in the pinned SoC band + idle baseline,
each cycle doubling as the sustained-thermal run (P6-5, James); (5) the
writeup with its figures, roofline analysis, headroom decompositions,
measurement-limitations section, and honest gaps (P6-6); (6) P6-EXEC: the
exit walk, the success-metric verdict, the llama.cpp-energy decision
recorded, architecture.pdf + README/CLAUDE.md refresh, CAMP-1 unblocked.

**Out (unchanged non-goals + deferred):** any kernel or decode/prefill
pipeline change (the engine is frozen at the Phase 6 build tag for the
round — an engine change after the tag invalidates the round); the
campaign levers (PIPE-1, P5-2C, GE-1 — behind SPEC-P7); any new
quantization format, sampler, or model; energy for burst runs (undefined
by protocol); a Core ML comparison column (optional per PLAN, not taken);
simulator anything. BW-1 (read-only bandwidth variant) and LD-1
(long-depth fp16 report) were pulled ahead of the writeup by James on
2026-09-24 (veto item 8): BW-1 lands before the `phase6-build` tag and
its iPhone row rides the speed session; LD-1 runs on the Mac alongside
the device sessions and feeds the limitations section as a measurement.
Neither gates anything.

## Design decisions

### D1. One metric definition per metric, engine-owned on every engine

The Phase 0 rows mixed definitions (app-reported overall rates vs our
canonical window; TTFT-including-one-decode vs our prefill span). Phase 6
pins ONE definition per metric and gives every harness the instrumentation
to produce it:

| Metric | Definition (all engines) | Where it already exists |
|---|---|---|
| Decode tok/s (of record) | 384 ÷ (t₅₁₂ − t₁₂₈), where tₖ = host wall time at which generated token k is available (completion of token 128 to completion of token 512 — the DecodeInstrumentation definition). Reported per generation; burst rows cite it; sustained rows cite it per generation | ours: `canonicalWindowTokensPerSecond`. MLX/llama.cpp: per-token timestamps to add (P6-2) |
| Prefill tok/s (of record) | per-engine prompt token count ÷ span from the start of prompt processing to the FIRST generated token id being available (no second forward inside the span). Dual-timed where the engine exposes GPU time (ours); wall everywhere | ours: the D1 prefill span (phase-5.md). llama.cpp: `completion_init` = tokenize + one `llama_decode` over the prompt batch + first sample — bracket it. MLX: generation start → first streamed token (the Phase 0 TTFT); P6-2 VERIFIES whether that token comes from the prompt forward itself or from a second forward — the Phase 0 entry assumed a ~26 ms second step — and brackets accordingly, recording which |
| Sustained timeline | per-generation (window tok/s, overall tok/s, tokens, wall) for every generation of a regenerate loop, in order — the OV#9 bimodality signal and the thermal chart's data | ours: `SustainedLoopResult.generations`. MLX/llama.cpp loops log only cumulative first/last — per-generation lines to add (P6-2) |
| Energy J/token | (ΔSoC × J-per-1%-SoC − idle) ÷ tokens over an operator-bounded cycle (D4) | none — loop mode with cumulative counters exists on all three; SoC/health fields + operator-stop-with-report to add (P6-1/P6-2) |
| phys_footprint | Xcode memory gauge, footprint-only attached launch AFTER the timed session (the P4-11 pattern); in-app `task_info` cross-check where the harness has it | ours: MemoryFootprint. MLX/llama.cpp: gauge only |

Our engine's decode and prefill definitions are unchanged from Phase 4/5
— the external harnesses move to ours, not the reverse, so every prior
row of ours stays comparable with the Phase 6 rows.

### D2. Parity pins re-applied, with the two Phase 0 loose ends closed

All PLAN parity pins hold (greedy, pinned prompts, per-engine token
counts, phys_footprint, canonical window, regenerate loop). Two loose
ends the Phase 0 entries left for Phase 6:

- **Feeding mode: the pinned RENDERED prompt string, everywhere.** Our
  engine and llama.cpp already consume `benchmarks/prompts/rendered/`.
  MLX LLMEval applied its own template in Phase 0 (862 vs 852 tokens — a
  default system message). P6-2 adds a rendered-input mode to the LLMEval
  harness: tokenize the rendered string directly (added/special tokens
  parsed, no template application) and hand the tokens to `generate`;
  the parity check is the token count — MLX's count must equal ours
  EXACTLY (both are HF tokenizers over the same string: 852 / 84). If
  the harness resists within the Phase 0 one-day timebox, fallback =
  raw text + LLMEval's template with the count recorded and the delta
  named in the limitations section (the Phase 0 mode). llama.cpp's
  decode-essay count (92 vs HF 84) is a tokenizer difference — reported,
  as pinned.
- **Metal API validation OFF on every row, recorded** (the 2026-08-22
  addendum; the Phase 0 MLX rows ran ON). Scheme-pinned in all three
  harnesses.

### D3. The speed session (P6-4): one day, one OS build, interleaved, bookended

The Phase 3 D8 + Phase 4 bookend protocol, extended to three engines:

- One calendar session on one iOS build (recorded once, checked at the
  end; a mid-session OS update invalidates the session — restart).
  Detached launches (home screen) for every timed row; validation OFF.
  Rested to ambient at the start (procedural; no readout exists).
- **Rotation, not blocks:** warm-burst decode rows run in rotating
  engine order (ours, MLX, llama.cpp; then again ×3) so each engine's
  repeats are spread across the session; the same for prefill rows.
  Each engine's FIRST launch of the session is its cold row (weights
  from disk), taken before its warm repeats.
- **Bookends:** our engine's warm-burst window row opens and closes the
  session; the bookend delta estimates within-session drift and any
  cross-engine claim must exceed it (the Phase 4 rule) — otherwise
  "unresolved (drift-dominated)".
- **Sustained 5-min loops:** one per engine (decode-essay; reset on EOS
  or context fill), interleaved with rests (D6); these are the short
  thermal rows. The chart of record comes from the energy cycles (D4).
- **Memory:** after the timed rows, one attached footprint-only launch
  per engine for the Xcode gauge; ours in BOTH residency modes (mmap =
  the default row; wired = the honest resident total, since llama.cpp's
  307 MB and our mmap figure both exclude clean file-backed weight pages
  — the invariant-3 accounting asymmetry is stated, never hidden).
- **Our engine's tripwires (D7)** are walked on this session's rows.

Approximate wall budget: 3 engines × (1 cold + 3 warm burst) ≈ 12 × ~40 s,
3 × 3 prefill rows, 3 × 5-min loops + rests, 3–4 footprint launches ⇒
≈2–2.5 h including rests.

### D4. The energy rounds (P6-5): protocol operationalized, calendar budgeted

The PLAN energy protocol, with the 2026-09-05 correction applied and the
Phase 0 dry-run findings folded in:

- **Cycle:** detached, airplane mode, minimum brightness, Auto-Lock Never,
  Background App Refresh off, unplugged, rested to ambient, validation
  OFF. At exactly 80% SoC (Settings → Battery is the value of record for
  the band marks; the harness's programmatic reading is the export's
  cross-check field) start the engine's regenerate loop on decode-essay;
  at exactly 70% stop it. Record wall time, generations, total tokens,
  the per-generation timeline, SoC at start/end, battery HEALTH (maximum
  capacity %, Settings → Battery → Battery Health, typed by the operator
  — no public API exposes it) as a SEPARATE field from SoC.
- **Capacity basis (the binding 2026-09-05 obligation):** J per 1% SoC =
  12.6 Wh × 36 × (health % ÷ 100), from the health read at that cycle
  (100% today ⇒ 453.6 J). Never the state of charge.
- **Idle baseline (one per iOS build; D7 gate):** same conditions, app
  foregrounded, no generation, until ≥4% SoC has dropped; scaled
  pro-rata to each cycle's wall time. Net energy = (ΔSoC_run − idle
  ΔSoC scaled) × J-per-1%; table value = net J ÷ tokens.
- **Validity (PLAN pins, walked per cycle):** implied gross average watts
  in 3–9 W; ≥8% SoC actually burned (10% by construction of the band);
  ≥3 valid cycles per engine; mean ± spread reported. A cycle outside
  the anchor is recorded as INVALID with its reading and repeated.
- **Rotation (Latin square):** cycles run in the order (ours, MLX,
  llama.cpp), (MLX, llama.cpp, ours), (llama.cpp, ours, MLX) so each
  engine occupies each position once — removes order/time-of-day bias
  from a round that spans days. Between cycles: recharge to ≥81%, then
  rest (D6). Every cycle records the iOS build; an OS change mid-round
  invalidates the incomplete round.
- **Error bars (stated in the writeup, D8):** statistical = mean ±
  spread over n≥3; systematic = SoC read quantization, ±0.5% per
  reading ⇒ ±1% of a 10% burn ⇒ ±10% on the gross figure, plus the
  idle term's own quantization (bounded ≤ ±3% of net by the ≥4% idle
  rule). Instruments Energy Log, if captured, is reported ONLY as
  "energy impact (relative, unitless)".
- **The sustained-thermal chart:** each cycle is a ≈17-min continuous
  regenerate loop (Phase 0: 1055 s per 10% at ≈4.3 W gross) — the
  PLAN's "doubles as the thermal run". The chart plots per-generation
  window tok/s against elapsed time for all three engines, all cycles
  (repeats faint, median cycle bold), from the D1 timeline export.
- **Calendar budget (OV#8, budgeted honestly):** per cycle ≈17 min run +
  ≈10 min recharge + rest ≈ 45 min ⇒ 9 cycles ≈ 7 h, plus the idle
  baseline (≈60 min: 1% SoC ≈ 15 min at ≈0.5 W, so 4% ≈ 1 h — it can
  overlap a recharge-free rest slot) and the speed session (D3, ≈2.5 h)
  ⇒ **≈10–11 h of device babysitting ≈ 3 James sessions** (one speed
  day + two energy half-days). This is the Phase 6 device cost.
- **llama.cpp energy — recommendation KEEP at n=3 (James decides, per
  PLAN "decide in DECISIONS.md, not silently"):** its cost is 3 cycles
  ≈ 2.5 h; what it buys is the three-way energy row the PLAN goal names
  and the most product-relevant cross-engine finding on record (MLX
  ≈25% more tokens per joule at identical ≈4.3 W; −9% vs −45% thermal).
  Pre-declared fallback if the calendar forces a cut: llama.cpp's
  cycles are the ones dropped; a partial round (n<3) is reported
  labeled "below the ≥3 pin — not a comparative energy row", never
  averaged into the table.

### D5. Our engine in Phase 6: frozen at a tagged build, suites re-run first

The engine core is unchanged since c2c5fd4 (PF-2). P6-1 adds harness
surfaces only (energy-mode loop with operator stop + report, battery
health/SoC fields, timeline + JSON export). The Phase 6 build is tagged
(`phase6-build`) after P6-3 lands; the FULL correctness evidence is
re-run on that tag before any device row and recorded (Tier-M/E, the
250-step logit suite on the GPU-quant path, the free-run report ×5 —
the P5-4 precedent). No gate constant changes (hard rule 6). The app
export header moves from "Phase 5 row export" to "Phase 6 row export" and
drops the PROVISIONAL marker ONLY on rows produced inside the Phase 6
round (a `round` field; everything else stays PROVISIONAL).

### D6. Rest and drift discipline (judgment-derived, flagged)

No temperature readout exists on iOS (procedural rest is the standing
convention). Pinned for Phase 6: rest to ambient between thermally loaded
runs (sustained loops, energy cycles) for **at least the duration of the
preceding run**, unplugged; charging counts as thermal load (rest starts
after the charger comes off). Within the speed session, the bookends
measure whatever drift survives this rule; a bookend delta larger than
the smallest cross-engine effect the writeup claims demotes that claim
to "unresolved (drift-dominated)".

### D7. Gates (pre-committed; binding record in DECISIONS.md; veto window closed 2026-09-24, all items approved unamended)

Phase 6 builds no kernels; its gates are run-VALIDITY criteria plus the
existing constants reused as regression tripwires. **No new performance
constants anywhere:**

- **Same-build/same-OS validity (OV#7 made operational):** every
  comparative row in the writeup's headline table comes from the D3
  speed session (one session, one iOS build, three engines interleaved)
  or the D4 energy round (same iOS build as the speed session). A row
  that violates this is not a headline row — it may appear only in the
  provisional-vs-final section, labeled.
- **Energy-cycle validity:** the PLAN pins walked per cycle (80→70 band;
  ≥8% burned; 3–9 W implied gross; idle subtracted; n≥3 per engine;
  health and SoC recorded separately; capacity basis from health).
  **Idle baseline ≥4% SoC drop** (the one derived number: bounds the
  idle term's quantization to ≤ ±3% of the net figure, below the run's
  own ±10%; Phase 0's 1%/15 min carried ±50%).
- **Our engine's regression tripwires on the speed-session rows:**
  decode warm-burst window median ≥ **24.0 tok/s** (the Phase 4/5
  constant), prefill warm span median ≥ **135 tok/s** (the Phase 5
  constant), both on the tagged build; correctness suites re-run green
  on the tag before any row (D5). Falling below either after
  harness-only changes signals breakage.
- **Success-metric verdict (PLAN formula, no new number):** at P6-EXEC,
  MET iff (our warm-burst window median) ÷ (MLX warm-burst window
  median, same session, D1 definition) ≥ 0.75. The committed 29.4
  constant is reported alongside (never loosens; already exceeded). A
  NOT MET verdict does not block the phase exit — it is recorded with
  its anatomy and becomes the campaign's first headline gap (the phase
  exits on the writeup's completeness, PLAN row 6).
- **Judged, never gated:** every cross-engine comparison (prefill vs
  MLX/llama.cpp, energy, memory, thermal), the roofline fractions, and
  the headroom decompositions.

### D8. The writeup (P6-6): home, sections, figures, data of record

- **Home:** `docs/writeup.md` (Markdown of record) + `docs/writeup/`
  (figures as PNG; a PDF render via the existing generator toolchain is
  optional, never the source). Raw exports of every Phase 6 row are
  archived under `benchmarks/phase6/` (one file per row, named by
  engine/mode/index; the text exports + the JSON timelines) — the
  writeup's tables and figures are regenerated from these by
  `tools/phase6_analyze.py` (P6-3), so no number in the writeup is typed
  by hand. results.md keeps its dated rows as the ledger; the writeup
  cites them.
- **Sections (pinned by the PLAN row-6 exit criterion):**
  1. Headline table — three engines × {prefill tok/s + token count,
     decode tok/s (window), sustained plateau, phys_footprint, net
     J/token ± spread}, cold/warm and burst/sustained separated, all
     from the Phase 6 round.
  2. Success-metric verdict (D7) with the same-session MLX number.
  3. Sustained-thermal chart (D4) + the 5-min loop rows.
  4. Energy per token with both error bars (D4), implied watts, idle
     baseline, capacity basis, per-cycle table.
  5. Roofline analysis from MEASURED denominators: decode against 43.84
     GB/s ÷ bytes-per-token (967,753,728 B weights + KV at the window's
     depth + activations; the P4-EXEC ≈80% stream-rate finding), each
     engine's implied effective bandwidth; prefill against the measured
     GEMM M-sweep plateau (0.78 TFLOPS on the A17 Pro, the P5-2 sweep)
     and the 45.3 tok/s sequential structural ceiling; the compute vs
     bandwidth crossover as measured (below M=8 on the device).
  6. Per-component headroom (north star): the P4-EXEC decode
     decomposition and the P5-EXEC prefill decomposition, re-exported
     on the Phase 6 build in the same session (attribution mode —
     DIAGNOSTIC, never rows), with the named owner of each residual
     (PIPE-1, P5-2C, GE-1, matvec stream rate).
  7. Measurement limitations (Issue 3 — the credibility feature): no
     power-rail access ⇒ battery-delta, sustained only, with the stated
     bars; SoC quantization; battery-health basis (and that every
     pre-2026-09-05 "health" field was SoC); provisional-vs-final —
     a table of Phase 0 vs Phase 6 values per engine with the causes
     named (OS 26.5.2 → current, validation ON → OFF, overall → window
     rate, template feeding, harness patches); the 2R+1W triad
     denominator's read-only bound (BW-1, measured);
     fp16 activation evidence at depth (LD-1, measured); no
     instrumented temperature; per-engine tokenizer counts; the
     phys_footprint accounting asymmetry (file-backed pages) and the
     Phase 0 MLX activeMemory mislabel; the decode-essay generation
     cap and the prefill prompt's EOS behavior; single device, single
     model — no generalization claimed.
  8. Honest gaps: the two gates FAILED on record with their anatomy
     (Phase 4 overhead 1.40 ms vs ≤1.2, ≈62% OS/driver; Phase 5 GEMM
     M=8 19.54 vs ≥30.69 GB/s, compute-bound below M=8), the prefill
     gap's single owner (GE-1), whatever the Phase 6 head-to-head adds,
     and the non-goals deliberately not built.
  9. Reproducibility: pins (model revision, checkpoint sha256s, GGUF
     provenance, harness commits + `benchmarks/patches/*-p6.patch`,
     Xcode/iOS builds), the runbook, the analysis script.
- **Headline numbers:** the Phase 6 same-session rows. Where the writeup
  narrates the Phase 5 state, 240.86 tok/s is the prefill number of
  record and 172.23 the gate-walk basis (James, 2026-09-23); Phase 6's
  rows supersede both.

### D9. Harness provenance and archival

The external harness edits live in the two local clones
(`~/Projects/mlx-swift-examples` branch `qwen-metal-p0a1` @ 992118b,
`~/Projects/llama.cpp` branch `qwen-metal-p0a1` @ bceddff) and are
archived as `benchmarks/patches/mlx-swift-examples-p6.patch` and
`benchmarks/patches/llama-swiftui-p6.patch` (stacked on the P0A-1
patches, `git format-patch` output). Upstream revisions and checkpoints
stay pinned (mlx-swift-examples 378f2449 + Package.resolved as built;
llama.cpp b9999 47c78692; mlx-community/Qwen3-1.7B-4bit @ 3b1b1768; the
locally converted Q4_K_M GGUF, sha256 in DECISIONS.md) — Phase 6 measures
harness-parity fixes, not newer engine versions (a version bump would be
a new PROVISIONAL baseline, not the head-to-head). Agents build-verify
the patched harnesses where the toolchain allows (LLMEval on macOS;
llama.swiftui as `generic/platform=iOS` with signing disabled) and never
run them for numbers.

## Memory budget

Unchanged: packed weights ~0.97 GB (mmap) + KV 448 MiB + prefill scratch
≤ 64 MiB + activations/logits < 10 MB ⇒ ≈1.5 GB honest total (1.43 GB
measured wired). The energy-mode timeline buffer is bounded (one record
per generation, ≤ a few hundred per cycle) — no new persistent
allocation of note.

## Enumerated edge-case tests (host plumbing; land with the code — hard rule 3 / METHODOLOGY 3)

1. **Energy-mode loop (P6-1):** operator stop ends the loop at the next
   token boundary AND produces a report (today Stop aborts without one);
   the truncated final generation is flagged; cumulative tokens/wall
   equal the sum of the per-generation records; refuses to spin on empty
   generations (the SustainedLoop rule carried over).
2. **Battery fields (P6-1):** health and SoC are separate export fields;
   an empty operator health field renders the record-me placeholder,
   never a number; the programmatic SoC reading is labeled a cross-check
   and exported at start and end.
3. **Timeline export (P6-1):** per-generation window tok/s, overall
   tok/s, tokens, wall, stop reason, in order; JSON round-trips; the
   text export's sustained lines cite the same numbers.
4. **Round marker (P6-1):** the export header carries the phase and a
   `round` field; PROVISIONAL renders unless the round is set.
5. **Regression-tripwire and validity arithmetic (P6-3):** the analysis
   script's energy math (J per 1% from health; idle scaling; net
   J/token; implied watts; the 3–9 W verdict; the ≥8% / n≥3 checks) and
   its window-rate parser verified on synthetic exports incl. an
   out-of-anchor cycle, an n=2 engine, a missing idle baseline, and an
   OS-build mismatch — each must REFUSE to produce a headline value.
6. **Figure generation (P6-3):** the thermal chart and energy bars render
   from the synthetic set; the chart uses the D1 timeline, never the
   cumulative line.
7. **Harness parity signature (P6-2, documented check, not XCTest):** the
   MLX rendered-input mode's prompt token count equals ours (852 / 84);
   llama.cpp's counts recorded (852 / 92); a mismatch on the MLX side
   invalidates the mode (fallback per D2).
8. **Unchanged behavior:** burst/sustained/microbench/attribution modes,
   the CLI, and every existing suite pass unmodified on the tagged
   build (D5 evidence).

## Instrumentation & benchmark deliverables

- Our app: energy mode (D4) + battery health/SoC fields + JSON timeline
  export + `round` marker; CLI parity where the CLI already exports.
- External harnesses: per-token timestamps ⇒ window rate; first-token
  prefill span; per-generation loop log lines; SoC fields; rendered-input
  mode (MLX); validation OFF scheme pins. Archived as `*-p6.patch`.
- `benchmarks/phase6-runbook.md`: the run order (D3 rotation, D4 Latin
  square), per-row checklists, row templates, the operator fields, the
  export-archival step into `benchmarks/phase6/`.
- `tools/phase6_analyze.py` (+ tests): exports → tables + figures; every
  writeup number regenerated, none typed.
- Device rows (James): speed session (P6-4), energy round (P6-5) — each
  with a DECISIONS.md entry (validity walk, tripwire verdicts, anything
  decided).
- `docs/writeup.md` + `docs/writeup/` figures (P6-6).
- P6-EXEC: success-metric verdict, llama.cpp-energy decision, exit walk,
  architecture.pdf (v1.9) + README/CLAUDE.md refresh, CAMP-1 unblocked.

## Task breakdown (seeded in docs/PRIORITIES.yaml)

| Task | Deliverable | Depends on |
|---|---|---|
| P6-1 | Engine/app harness for the round: energy-mode loop (operator stop + report), battery health + SoC separate fields, per-generation timeline JSON export, round marker; edge tests 1–4; suites green | — |
| P6-2 | External-harness parity patches (MLX LLMEval + llama.swiftui): window rate, first-token prefill span, per-generation loop log, SoC fields, MLX rendered-input mode with the token-count signature, validation OFF pins; build-verified; archived as `*-p6.patch`; check 7 | — |
| P6-3 | Phase 6 runbook + row templates + `tools/phase6_analyze.py` with tests 5–6; tag `phase6-build`; full correctness evidence re-run on the tag and recorded (D5) | P6-1, P6-2 |
| P6-4 (james) | Speed session: all engines, one session/one OS build, D3 rotation + bookends; burst cold/warm, prefill, 5-min loops, phys_footprint (ours mmap + wired); tripwires walked; exports archived | P6-3 |
| P6-5 (james) | Energy round: idle baseline + ≥3 cycles per engine in the Latin-square order (D4), validity walked per cycle, timelines archived; llama.cpp cut only by recorded decision | P6-4 |
| P6-6 | The writeup: `docs/writeup.md` + figures regenerated from `benchmarks/phase6/`; roofline + headroom analysis; limitations; honest gaps; reproducibility | P6-5 |
| P6-EXEC | Exit walk + success-metric verdict + llama.cpp-energy decision + architecture.pdf/README/CLAUDE.md refresh; unblocks CAMP-1 | P6-6 |

## Exit criteria (PLAN.md phase table, walked)

- Full table across engines re-run same-session/same-OS ✓ = P6-4's rows
  under the D7 validity gate (one session, one iOS build, interleaved,
  bookended), with our tripwires walked.
- Sustained-thermal chart ✓ = the D4 timeline chart from the energy
  cycles (+ the 5-min loop rows), all three engines.
- Energy/token (sustained, with error bars) ✓ = P6-5's ≥3 valid cycles
  per engine on the health-based capacity basis, both error bars stated;
  llama.cpp's row present or its absence decided in DECISIONS.md.
- Roofline analysis from measured bandwidth ✓ = writeup §5 on 43.84 GB/s
  and the measured GEMM plateau, every fraction traceable to a row.
- Measurement-limitations section ✓ = writeup §7 covering the enumerated
  items (D8), incl. provisional-vs-final handling.
- Honest gaps ✓ = writeup §8: the two FAILED gates with anatomy, the
  prefill gap's owner, the Phase 6 verdicts as measured.
- DECISIONS.md entries for the validity walks, the tripwire verdicts,
  the success-metric verdict, the llama.cpp-energy decision, and anything
  else decided/measured (standing discipline).
