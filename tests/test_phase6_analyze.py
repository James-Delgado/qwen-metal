"""Phase 6 analyzer tests (docs/phases/phase-6.md edge tests 5–6; task P6-3).

Synthetic exports only — every builder below reproduces the export line
formats verbatim from the Swift sources (BenchmarkReport / TimelineExport /
BandwidthMicrobench) and the two P6-2 harness patches. Each invalid input
must make the analyzer REFUSE a headline value, never emit a number.

Run: .venv/bin/python -m pytest tests/test_phase6_analyze.py -q
Deps: the repo-root .venv (pytest, matplotlib — see tools/requirements-phase6.txt).
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import phase6_analyze as an  # noqa: E402
import phase6_exports as ex  # noqa: E402

ROUND_SPEED = "phase6-speed-2026-10-01"
ROUND_ENERGY = "phase6-energy-2026-10-04"
IOS = "26.6.1"

# --------------------------------------------------------------------------
# Synthetic export builders (formats copied from the producing code)
# --------------------------------------------------------------------------

HEALTH_TAIL = ("(maximum capacity %, Settings → Battery → Battery Health, "
               "operator-typed — NOT state of charge)")


def _pct(v):
    return "(record manually)" if v is None else f"{v}%"


def ours_header(round_=ROUND_SPEED, ios=IOS, health=100, soc=None, date="2026-10-01"):
    marker = "PROVISIONAL" if round_ is None else f"round: {round_}"
    lines = [
        f"qwen-metal Phase 6 row export ({marker})",
        f"date: {date}",
        f"device: iPhone16,1 (iOS {ios})",
        "round: (none — PROVISIONAL)" if round_ is None else f"round: {round_}",
        f"battery health: {_pct(health)} {HEALTH_TAIL}",
    ]
    if soc is not None:
        start, end, pstart, pend = soc
        lines.append(
            "state of charge (Settings → Battery, operator-read at the band marks — "
            f"value of record): start {_pct(start)}, end {_pct(end)}")
        lines.append(
            "state of charge (programmatic UIDevice.batteryLevel — cross-check only, never "
            f"the value of record): start {pstart:.1f}%, end {pend:.1f}%")
    lines.append(
        "engine: qwen-metal q4g64 fused-dequant GPU — weights q4g64, residency mmap, "
        "kernels fused, prefill tiled (C=512), attention queryTiled")
    return lines


def ours_burst(prompt="decode-essay", tokens=84, generated=640, window=31.67, overall=31.5,
               prefill=(84, 0.412), cold_warm="warm", stop="maxNewTokens", **hdr):
    lines = ours_header(**hdr)
    lines.append(f"prompt: {prompt} ({tokens} prompt tokens)")
    lines.append(f"mode: burst | cold/warm: {cold_warm}")
    lines.append("sampling: greedy | Metal API validation: confirm OFF and record (P0A-1 addendum)")
    lines.append(f"generated: {generated} tokens in 21.3 s (stop: {stop})")
    if prefill is not None:
        ptok, psec = prefill
        lines.append(
            f"prefill span: {ptok} tokens in {psec:.3f} s = {ptok / psec:.2f} tok/s (of record) — "
            f"GPU {psec * 0.95:.3f} s, 790 dispatches (prompt processing only, excl. first decode forward)")
    lines.append("ttft-style (legacy P2-6 field, runner-clocked): 84 prompt tokens, 0.45 s to first token available")
    lines.append("per-token (640 tokens): median GPU 30.27 ms, median wall 31.52 ms, median wall-GPU 1.384 ms, 200 dispatches/token")
    win = f"{window:.2f} tok/s" if window is not None else f"n/a (needs >= 512 generated tokens, got {generated})"
    lines.append(f"decode rate: overall {overall:.2f} tok/s, canonical window (tokens 128-512) {win}")
    lines.append("latency (window): p50 31.53 ms, p95 32.76 ms, p99 33.87 ms, max 34.69 ms, stalls 0 (spans > 2x p50, n = 384)")
    lines.append("phys_footprint (in-app cross-check; Xcode gauge is the metric of record): 549.3 MB")
    return "\n".join(lines)


def gen_line(i, overall, window, tokens, wall, stop, elapsed=None, truncated=False):
    w = f", window {window:.2f} tok/s" if window is not None else ""
    line = f"  gen {i}: overall {overall:.2f} tok/s{w} — {tokens} tokens in {wall:.1f} s (stop: {stop})"
    if elapsed is not None:  # external harness form
        line += f" [elapsed {elapsed:.1f} s]"
        if truncated:
            line += " [truncated by operator stop]"
    return line


def default_gens(n=6, window0=31.5, decay=1.5, tokens=1601, wall=55.0):
    return [(i, window0 - i * decay - 2.0, window0 - i * decay, tokens, wall, "eos") for i in range(n)]


def ours_loop(mode="energy", gens=None, health=100, soc=(80, 70, 80.0, 70.0), cycle_wall=None,
              cumulative_tokens=None, ended="operator stop", round_=ROUND_ENERGY, **hdr):
    gens = gens or default_gens()
    lines = ours_header(round_=round_, health=health, soc=soc if mode == "energy" else None, **hdr)
    lines.append("prompt: decode-essay (84 prompt tokens)")
    lines.append(f"mode: {mode} | cold/warm: warm")
    lines.append("sampling: greedy | Metal API validation: confirm OFF and record (P0A-1 addendum)")
    total_tokens = sum(g[3] for g in gens)
    sum_wall = sum(g[4] for g in gens)
    cycle_wall = sum_wall + 3.0 if cycle_wall is None else cycle_wall
    if mode == "energy":
        lines.append(f"energy cycle: {len(gens)} generations — ended by {ended}")
        lines.append(
            f"  cumulative: {cumulative_tokens if cumulative_tokens is not None else total_tokens} generated tokens; "
            f"Σ generation wall {sum_wall:.1f} s; cycle wall {cycle_wall:.1f} s (idle-baseline pro-rata basis)")
    else:
        lines.append(f"sustained loop: {len(gens)} generations over {cycle_wall / 60:.1f} min")
    for g in gens:
        lines.append(gen_line(*g))
    lines.append("last generation per-token: median GPU 40.1 ms, median wall 41.5 ms, median wall-GPU 1.40 ms, 200 dispatches/token")
    if mode == "energy":
        lines.append("energy J/token: not computed in-app — tools/phase6_analyze.py (P6-3) derives it from the health/SoC fields and the token total above")
    return "\n".join(lines)


def ours_json(gens=None, mode="energy", round_=ROUND_ENERGY, ios=IOS, health="100", soc=("80", "70"),
              cycle_wall=None, elapsed_basis="loopClock"):
    gens = gens or default_gens()
    entries = []
    elapsed = 0.0
    for (i, overall, window, tokens, wall, stop) in gens:
        elapsed += wall
        entries.append({
            "index": i, "generatedTokens": tokens, "wallSeconds": wall,
            "elapsedAtEndSeconds": elapsed + 0.5 * i, "overallTokensPerSecond": overall,
            "canonicalWindowTokensPerSecond": window, "stopReason": stop, "truncated": False,
        })
    sum_wall = sum(g[4] for g in gens)
    doc = {
        "schema": "qwen-metal-timeline/1", "phase": "Phase 6", "round": round_,
        "provisional": round_ is None, "date": "2026-10-04", "device": "iPhone16,1", "osVersion": ios,
        "engine": {"weightsFormat": "q4g64", "residency": "mmap", "kernelPath": "fused",
                   "prefillPath": "tiled", "prefillChunkSize": 512, "prefillAttention": "queryTiled"},
        "promptName": "decode-essay", "promptTokenCount": 84, "mode": mode, "coldOrWarm": "warm",
        "battery": {"healthNote": health, "operatorSoCStart": soc[0], "operatorSoCEnd": soc[1],
                    "programmaticSoCStartPercent": 80.0, "programmaticSoCEndPercent": 70.0},
        "cycle": {"generations": len(gens), "totalGeneratedTokens": sum(g[3] for g in gens),
                  "sumGenerationWallSeconds": sum_wall,
                  "cycleWallSeconds": sum_wall + 3.0 if cycle_wall is None else cycle_wall,
                  "endedBy": "operatorStop", "lastGenerationTruncated": False},
        "elapsedBasis": elapsed_basis, "generations": entries, "physFootprintBytes": 576000000,
    }
    return json.dumps(doc, indent=2, sort_keys=True)


def mlx_prompt_line(prompt="decode-essay", count=84, parity="MATCH"):
    expected = 84 if prompt == "decode-essay" else 852
    if parity == "template":
        return (f"prompt: {prompt} ({count} tokens; feeding: raw text + LLMEval template with its default "
                f"system message, enable_thinking=false — the Phase 0 mode; qwen-metal rendered count is "
                f"{expected}, delta to be named in limitations)")
    verdict = ("MATCH — parity signature holds" if parity == "MATCH" else
               "MISMATCH — rendered mode INVALID for this row (fallback: template mode, delta named in limitations)")
    return (f"prompt: {prompt} ({count} tokens; feeding: RENDERED tokenizer-direct, special tokens parsed, "
            f"no template; parity target {expected}: {verdict})")


def llama_prompt_line(prompt="decode-essay", count=92, differs=False):
    rec = {"decode-essay": (92, 84), "prefill-summarize": (852, 852)}[prompt]
    note = (f"DIFFERS from the llama.cpp count on record ({rec[0]}) — investigate before using the row; HF/qwen-metal count {rec[1]}"
            if differs else
            f"matches the llama.cpp count on record ({rec[0]}); HF/qwen-metal count {rec[1]} — tokenizer difference, reported as pinned")
    return f"prompt: {prompt} ({count} tokens, llama.cpp tokenizer, parse_special=true; feeding: RENDERED bundled string; {note})"


def external(engine="mlx-llmeval", mode="burst", prompt_line=None, round_=ROUND_SPEED, cold_warm="warm",
             health=100, soc=(80, 70, 80.0, 70.0), gens=None, generated=640, window=39.1, overall=38.9,
             prefill=(84, 0.309), cycle_wall=None, ended_by_operator=True, cumulative_tokens=None):
    marker = "PROVISIONAL" if round_ is None else f"round: {round_}"
    lines = [
        f"qwen-metal Phase 6 row export ({marker}) — engine: {engine}",
        "round: (none — PROVISIONAL)" if round_ is None else f"round: {round_}",
        "harness: mlx-swift-examples 378f2449 + qwen-metal-p0a1 + P6-2 patch" if engine == "mlx-llmeval"
        else "harness: llama.cpp b9999 47c78692 llama.swiftui + qwen-metal-p0a1 (97e552a, bceddff) + P6-2 patch",
        "checkpoint: mlx-community/Qwen3-1.7B-4bit @ 3b1b1768" if engine == "mlx-llmeval"
        else "checkpoint: locally converted Q4_K_M GGUF (qwen3-1.7b-70d244cc-Q4_K_M.gguf, sha256 in qwen-metal DECISIONS.md), from Qwen/Qwen3-1.7B @ 70d244cc",
        f"mode: {mode}",
        prompt_line or (mlx_prompt_line() if engine == "mlx-llmeval" else llama_prompt_line()),
        "sampling: greedy (temperature 0); thinking: off; burst cap 640 new tokens; loop cap = 4096 − prompt (reset on EOS or context fill)",
        "Metal API validation: OFF (LLMEval scheme pins enableGPUValidationMode=1 = Disabled; detached home-screen launches carry no validation layer at all)",
        f"cold/warm: {cold_warm}",
        f"battery health: {_pct(health)} {HEALTH_TAIL}",
    ]
    if mode == "energy-loop":
        lines.append("state of charge (Settings → Battery, operator-read at the band marks — value of record): "
                     f"start {_pct(soc[0])}, end {_pct(soc[1])}")
        lines.append("state of charge (programmatic UIDevice.batteryLevel — cross-check only, never the value of record): "
                     f"start {soc[2]:.1f}%, end {soc[3]:.1f}%")
    lines.append("memory (cross-check only — Xcode gauge phys_footprint is the value of record): task_info phys_footprint 1020.1 MB")
    lines.append("definitions: decode tok/s = 384 / (t512 − t128), t_k = host monotonic time …")

    def generation_lines(generated, window, overall, prefill, stop="maxNewTokens"):
        out = [f"generated: {generated} tokens in 17.1 s (stop: {stop})"]
        if prefill is None:
            out.append("prefill span: n/a (no token generated)")
        else:
            ptok, psec = prefill
            tail = ("library promptTime cross-check 0.301 s; tokenize 0.8 ms excluded)" if engine == "mlx-llmeval"
                    else "tokenize 1.2 ms inside the span; prompt llama_decode enqueue 0.3 ms, async)")
            out.append(f"prefill span: {ptok} tokens in {psec:.3f} s = {ptok / psec:.2f} tok/s (of record — wall, prompt-processing start to first generated token available; {tail}")
        win = f"{window:.2f} tok/s" if window is not None else f"n/a (needs >= 512 generated tokens, got {generated})"
        out.append(f"decode rate: overall {overall:.2f} tok/s, canonical window (tokens 128-512) {win}")
        return out

    if mode == "burst":
        lines += generation_lines(generated, window, overall, prefill)
    else:
        gens = gens or default_gens(window0=39.0)
        total = sum(g[3] for g in gens)
        sum_wall = sum(g[4] for g in gens)
        cw = sum_wall + 2.0 if cycle_wall is None else cycle_wall
        lines.append(f"energy cycle: {len(gens)} generations — ended by {'operator stop' if ended_by_operator else 'loop halt'}")
        lines.append(f"cumulative: {cumulative_tokens if cumulative_tokens is not None else total} generated tokens; Σ generation wall {sum_wall:.1f} s; cycle wall {cw:.1f} s")
        lines.append("energy J/token is derived by tools/phase6_analyze.py from these raw fields (never in-harness)")
        elapsed = 0.0
        for (i, overall_, window_, tokens, wall, stop) in gens:
            elapsed += wall
            lines.append(gen_line(i, overall_, window_, tokens, wall, stop, elapsed=elapsed))
        lines.append("last generation:")
        last = gens[-1]
        lines += generation_lines(last[3], last[2], last[1], (84, 0.31), stop=last[5])
    return "\n".join(lines) + "\n"


def external_json(engine="mlx-llmeval", gens=None, round_=ROUND_ENERGY, mode="energy-loop", parity="match"):
    gens = gens or default_gens(window0=39.0)
    entries, elapsed = [], 0.0
    for (i, overall, window, tokens, wall, stop) in gens:
        elapsed += wall
        entries.append({"index": i, "promptTokenCount": 84, "generatedTokens": tokens, "prefillSpanSeconds": 0.31,
                        "tokenizeSeconds": 0.001, "wallSeconds": wall, "elapsedAtEndSeconds": elapsed,
                        "overallTokensPerSecond": overall, "canonicalWindowTokensPerSecond": window,
                        "stopReason": stop, "truncated": False})
    sum_wall = sum(g[4] for g in gens)
    doc = {"schema": "qwen-metal-timeline/1-external", "engine": engine, "phase": "6", "round": round_, "mode": mode,
           "promptName": "decode-essay", "promptTokenCount": 84, "feedingMode": "rendered", "feedingParity": parity,
           "elapsedBasis": "loopClock",
           "battery": {"healthNote": "100", "operatorSoCStart": "80", "operatorSoCEnd": "70",
                       "programmaticSoCStartPercent": 80.0, "programmaticSoCEndPercent": 70.0},
           "cycleTotals": {"generations": len(gens), "totalGeneratedTokens": sum(g[3] for g in gens),
                           "sumGenerationWallSeconds": sum_wall, "cycleWallSeconds": sum_wall + 2.0,
                           "endedBy": "operatorStop", "lastGenerationTruncated": False},
           "entries": entries}
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def bandwidth(kernel="triad", sustained=43.84, lo=42.19, hi=44.45, diagnostic=False, gate_ok=True):
    head = ("qwen-metal bandwidth microbench — STREAM triad (P0B-4)" if kernel == "triad"
            else "qwen-metal bandwidth microbench — read-only streaming reduce (BW-1)")
    lines = [head]
    if diagnostic:
        lines.append("DIAGNOSTIC — non-pinned geometry (8 float4/thread/buffer; pin is 64) — NEVER a row")
    lines += ["date: 2026-10-01", "device: iPhone16,1 (iOS 26.6.1)", "gpu: Apple A17 Pro GPU",
              "protocol: 3 × 100663296 fp32 buffers (1.125 GiB read/iteration), 3 warmup discarded + 10 measured, sustained = median (GPU timestamps; wall alongside)",
              "bytes read/iteration: 1207959552",
              "correctness: 3 sampled partials within pre-committed gate (max |Δ| 0 <= 1e-06)" if gate_ok
              else "correctness gate FAILED — figure WITHHELD",
              "per-iteration GB/s (GPU time; wall recorded alongside):",
              "  iter  1:   43.80 GB/s  (gpu 0.0276s, wall 0.0284s, overhead 0.0008s)",
              f"sustained (median): {sustained:.2f} GB/s",
              f"spread: min {lo:.2f} / max {hi:.2f} GB/s"]
    return "\n".join(lines)


def manifest(speed_ios=IOS, energy_ios=IOS, idle=None, footprint=None, include_energy=True):
    if idle is None:
        idle = {"health_percent": 100, "soc_start": 80, "soc_end": 75, "wall_seconds": 4500, "ios_build": energy_ios}
    m = {"device": "iPhone16,1 (iPhone 15 Pro)", "engine_build": {"tag": "phase6-build", "commit": "0000000"},
         "sessions": {"speed": {"round": ROUND_SPEED, "dir": "speed", "ios_build": speed_ios, "date": "2026-10-01",
                                "xcode": "26.6 (17F113)",
                                "footprint_mb": footprint or {"ours-mmap": 573.2, "ours-wired": 1430.0, "mlx": 1020.0, "llama": 480.0}}}}
    if include_energy:
        m["sessions"]["energy"] = {"round": ROUND_ENERGY, "dir": "energy", "ios_build": energy_ios, "date": "2026-10-04"}
        if idle is not False:
            m["sessions"]["energy"]["idle_baseline"] = idle
    return m


def write_archive(root: Path, manifest_doc, speed_files=None, energy_files=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest_doc, indent=2))
    for sub, files in (("speed", speed_files or {}), ("energy", energy_files or {})):
        (root / sub).mkdir(exist_ok=True)
        for name, text in files.items():
            (root / sub / name).write_text(text)
    return root


def full_speed_files(ours_window=(31.7, 31.5, 31.9), mlx_window=(39.2, 39.0, 39.4), llama_window=(32.4, 32.6, 32.2),
                     mlx_parity="MATCH"):
    f = {}
    f["ours-bookend-open.txt"] = ours_burst(window=ours_window[0])
    for i, w in enumerate(ours_window[1:], 1):
        f[f"ours-decode-warm-{i}.txt"] = ours_burst(window=w)
    f["ours-decode-cold-0.txt"] = ours_burst(window=31.1, cold_warm="cold")
    f["ours-bookend-close.txt"] = ours_burst(window=31.6)
    for i in range(3):
        f[f"ours-prefill-warm-{i}.txt"] = ours_burst(prompt="prefill-summarize", tokens=852, generated=343,
                                                     window=None, prefill=(852, 3.537 + 0.1 * i), stop="eos")
    for i, w in enumerate(mlx_window):
        f[f"mlx-decode-warm-{i}.txt"] = external(window=w, prompt_line=mlx_prompt_line(parity=mlx_parity))
    for i in range(3):
        f[f"mlx-prefill-warm-{i}.txt"] = external(prompt_line=mlx_prompt_line("prefill-summarize", 852, mlx_parity),
                                                  generated=409, window=None, prefill=(852, 2.3 + 0.05 * i))
    for i, w in enumerate(llama_window):
        f[f"llama-decode-warm-{i}.txt"] = external(engine="llama-swiftui", window=w)
    for i in range(3):
        f[f"llama-prefill-warm-{i}.txt"] = external(engine="llama-swiftui", prompt_line=llama_prompt_line("prefill-summarize", 852),
                                                    generated=617, window=None, prefill=(852, 1.9 + 0.05 * i))
    f["ours-sustained-0.txt"] = ours_loop(mode="sustained", round_=ROUND_SPEED)
    f["ours-sustained-0.json"] = ours_json(mode="sustained", round_=ROUND_SPEED)
    f["mlx-sustained-0.txt"] = external(mode="energy-loop")
    f["llama-sustained-0.txt"] = external(engine="llama-swiftui", mode="energy-loop")
    f["bandwidth-triad-0.txt"] = bandwidth("triad", 43.84)
    f["bandwidth-readonly-0.txt"] = bandwidth("read-only", 45.10, 44.5, 45.4)
    f["bandwidth-triad-1.txt"] = bandwidth("triad", 43.70)
    f["bandwidth-readonly-1.txt"] = bandwidth("read-only", 44.90, 44.5, 45.4)
    return f


def full_energy_files(n_ours=3, n_mlx=3, n_llama=3, ours_soc=None):
    f = {}
    for i in range(n_ours):
        soc = ours_soc[i] if ours_soc else (80, 70, 80.0, 70.0)
        f[f"ours-energy-{i}.txt"] = ours_loop(soc=soc, gens=default_gens(n=18, wall=58.0))
        f[f"ours-energy-{i}.json"] = ours_json(gens=default_gens(n=18, wall=58.0))
    for i in range(n_mlx):
        f[f"mlx-energy-{i}.txt"] = external(mode="energy-loop", round_=ROUND_ENERGY, gens=default_gens(n=18, window0=39.0, wall=45.0))
        f[f"mlx-energy-{i}.json"] = external_json(gens=default_gens(n=18, window0=39.0, wall=45.0))
    for i in range(n_llama):
        f[f"llama-energy-{i}.txt"] = external(engine="llama-swiftui", mode="energy-loop", round_=ROUND_ENERGY,
                                              gens=default_gens(n=18, window0=32.0, decay=0.9, wall=60.0))
    return f


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------

def test_ours_burst_parser_reads_window_prefill_and_header_fields():
    row = ex.parse_export_text(ours_burst())
    assert row.engine == "ours" and row.mode == "burst" and row.provisional is False
    assert row.round == ROUND_SPEED and row.os_version == IOS
    assert row.prompt_name == "decode-essay" and row.prompt_token_count == 84
    assert row.window_tok_s == pytest.approx(31.67)
    assert row.overall_tok_s == pytest.approx(31.5)
    assert row.prefill.tokens == 84 and row.prefill.seconds == pytest.approx(0.412)
    assert row.prefill.tok_s == pytest.approx(84 / 0.412, rel=1e-3)
    assert row.battery_health_percent == 100 and row.cold_warm == "warm"
    assert row.phys_footprint_mb == pytest.approx(549.3)


def test_window_parser_returns_none_below_512_tokens():
    row = ex.parse_export_text(ours_burst(generated=343, window=None, stop="eos"))
    assert row.window_tok_s is None
    assert row.generated_tokens == 343


def test_provisional_export_is_flagged():
    row = ex.parse_export_text(ours_burst(round_=None))
    assert row.provisional is True and row.round is None


def test_missing_health_parses_as_none():
    row = ex.parse_export_text(ours_loop(health=None))
    assert row.battery_health_percent is None


def test_ours_energy_parser_reads_soc_cumulative_and_generations():
    row = ex.parse_export_text(ours_loop())
    assert row.mode == "energy" and row.soc_start == 80 and row.soc_end == 70
    assert row.programmatic_soc_start == pytest.approx(80.0)
    assert row.cumulative.tokens == 6 * 1601
    assert row.cumulative.cycle_wall_s == pytest.approx(6 * 55.0 + 3.0)
    assert len(row.generations) == 6
    assert row.generations[2].window_tok_s == pytest.approx(31.5 - 2 * 1.5)
    assert row.generations[2].tokens == 1601 and row.generations[2].stop_reason == "eos"
    assert row.ended_by == "operator stop"


def test_mlx_parser_reads_engine_parity_and_metrics():
    row = ex.parse_export_text(external())
    assert row.engine == "mlx" and row.parity == "MATCH"
    assert row.window_tok_s == pytest.approx(39.1) and row.prefill.tok_s == pytest.approx(84 / 0.309, rel=1e-3)
    row2 = ex.parse_export_text(external(prompt_line=mlx_prompt_line(parity="MISMATCH")))
    assert row2.parity == "MISMATCH"
    row3 = ex.parse_export_text(external(prompt_line=mlx_prompt_line(count=94, parity="template")))
    assert row3.parity == "template" and row3.prompt_token_count == 94


def test_llama_parser_reads_count_verdict_and_loop_lines():
    row = ex.parse_export_text(external(engine="llama-swiftui"))
    assert row.engine == "llama" and row.llama_count_matches is True and row.prompt_token_count == 92
    bad = ex.parse_export_text(external(engine="llama-swiftui", prompt_line=llama_prompt_line(count=93, differs=True)))
    assert bad.llama_count_matches is False
    loop = ex.parse_export_text(external(engine="llama-swiftui", mode="energy-loop"))
    assert loop.mode == "energy" and len(loop.generations) == 6
    assert loop.generations[1].elapsed_s == pytest.approx(110.0)
    assert loop.cumulative.tokens == 6 * 1601


def test_timeline_json_parsers_cover_both_schemas():
    ours = ex.parse_timeline_json(ours_json())
    assert ours.schema == "qwen-metal-timeline/1" and ours.engine == "ours"
    assert [e.window_tok_s for e in ours.generations][:2] == pytest.approx([31.5, 30.0])
    extern = ex.parse_timeline_json(external_json())
    assert extern.schema == "qwen-metal-timeline/1-external" and extern.engine == "mlx"
    assert extern.generations[0].elapsed_s == pytest.approx(55.0)
    with pytest.raises(ex.ParseError):
        ex.parse_timeline_json(json.dumps({"schema": "something-else/9"}))


def test_bandwidth_parser_and_diagnostic_refusal():
    bw = ex.parse_bandwidth_export(bandwidth("read-only", 45.10, 44.5, 45.4))
    assert bw.kernel == "read-only" and bw.sustained_gbps == pytest.approx(45.10)
    assert bw.spread == pytest.approx((44.5, 45.4)) and bw.gate_ok is True
    with pytest.raises(ex.Refusal, match="DIAGNOSTIC"):
        ex.parse_bandwidth_export(bandwidth("read-only", diagnostic=True))
    with pytest.raises(ex.Refusal, match="gate"):
        ex.parse_bandwidth_export(bandwidth("triad", gate_ok=False))


# --------------------------------------------------------------------------
# Edge test 5 — energy arithmetic and validity
# --------------------------------------------------------------------------

def test_joules_per_soc_percent_derives_from_health_only():
    assert an.joules_per_soc_percent(100) == pytest.approx(453.6)
    assert an.joules_per_soc_percent(85) == pytest.approx(385.56)
    with pytest.raises(an.Refusal, match="health"):
        an.joules_per_soc_percent(None)


def test_energy_cycle_math_net_joules_watts_and_verdict():
    idle = an.IdleBaseline(health_percent=100, soc_start=80, soc_end=75, wall_seconds=4500, ios_build=IOS)
    cycle = an.analyze_energy_cycle(ex.parse_export_text(ours_loop(gens=default_gens(n=18, wall=58.0))), idle)
    # ΔSoC 10 × 453.6 J = 4536 J gross over 18×58+3 = 1047 s ⇒ 4.33 W
    assert cycle.gross_joules == pytest.approx(4536.0)
    assert cycle.implied_watts == pytest.approx(4536.0 / 1047.0, rel=1e-6)
    # idle: 5% over 4500 s ⇒ scaled to 1047 s = 1.1633% ⇒ 527.7 J
    assert cycle.idle_soc_scaled == pytest.approx(5 * 1047 / 4500)
    assert cycle.net_joules == pytest.approx((10 - 5 * 1047 / 4500) * 453.6)
    assert cycle.tokens == 18 * 1601
    assert cycle.net_joules_per_token == pytest.approx(cycle.net_joules / (18 * 1601))
    assert cycle.valid is True and cycle.invalid_reasons == []


def test_out_of_anchor_cycle_is_invalid_and_excluded():
    idle = an.IdleBaseline(100, 80, 75, 4500, IOS)
    hot = an.analyze_energy_cycle(ex.parse_export_text(ours_loop(gens=default_gens(n=6, wall=55.0))), idle)
    # 4536 J over 333 s ⇒ 13.6 W: outside the 3–9 W anchor
    assert hot.valid is False and any("3–9 W" in r for r in hot.invalid_reasons)
    shallow = an.analyze_energy_cycle(ex.parse_export_text(ours_loop(soc=(80, 73, 80.0, 73.0), gens=default_gens(n=18, wall=58.0))), idle)
    assert shallow.valid is False and any("8%" in r for r in shallow.invalid_reasons)
    off_band = an.analyze_energy_cycle(ex.parse_export_text(ours_loop(soc=(90, 80, 90.0, 80.0), gens=default_gens(n=18, wall=58.0))), idle)
    assert off_band.valid is False and any("80" in r for r in off_band.invalid_reasons)


def test_cycle_refuses_when_health_or_soc_missing_or_cumulative_inconsistent():
    idle = an.IdleBaseline(100, 80, 75, 4500, IOS)
    with pytest.raises(an.Refusal, match="health"):
        an.analyze_energy_cycle(ex.parse_export_text(ours_loop(health=None, gens=default_gens(n=18, wall=58.0))), idle)
    with pytest.raises(an.Refusal, match="state of charge"):
        an.analyze_energy_cycle(ex.parse_export_text(ours_loop(soc=(80, None, 80.0, 70.0), gens=default_gens(n=18, wall=58.0))), idle)
    with pytest.raises(an.Refusal, match="cumulative"):
        an.analyze_energy_cycle(ex.parse_export_text(ours_loop(cumulative_tokens=99, gens=default_gens(n=18, wall=58.0))), idle)


def test_idle_baseline_below_four_percent_refuses():
    with pytest.raises(an.Refusal, match="4%"):
        an.IdleBaseline(100, 80, 77, 3000, IOS).validate()
    an.IdleBaseline(100, 80, 76, 3000, IOS).validate()  # exactly 4% passes


def test_engine_summary_requires_three_valid_cycles(tmp_path):
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(), full_energy_files(n_mlx=2))
    report = an.analyze_archive(archive)
    energy = report.energy
    assert energy["ours"].headline is not None and energy["ours"].n_valid == 3
    assert energy["mlx"].headline is None and energy["mlx"].n_valid == 2
    assert any("≥3" in r or ">=3" in r for r in energy["mlx"].refusals)
    assert energy["ours"].headline.mean_j_per_token > 0
    assert energy["ours"].headline.spread == pytest.approx((energy["ours"].headline.mean_j_per_token,) * 2, rel=1e-9)
    assert 0.05 < energy["ours"].headline.systematic_fraction < 0.25


def test_missing_idle_baseline_refuses_every_energy_headline(tmp_path):
    archive = write_archive(tmp_path / "a", manifest(idle=False), full_speed_files(), full_energy_files())
    report = an.analyze_archive(archive)
    for eng in ("ours", "mlx", "llama"):
        assert report.energy[eng].headline is None
        assert any("idle" in r.lower() for r in report.energy[eng].refusals)


def test_os_build_mismatch_refuses(tmp_path):
    # energy round on a different iOS build than the speed session
    archive = write_archive(tmp_path / "a", manifest(energy_ios="26.6.2"), full_speed_files(), full_energy_files())
    report = an.analyze_archive(archive)
    assert all(report.energy[e].headline is None for e in ("ours", "mlx", "llama"))
    assert any("iOS" in r for r in report.energy["ours"].refusals)
    # our own export whose device line disagrees with the session's iOS build
    files = full_speed_files()
    files["ours-decode-warm-1.txt"] = ours_burst(window=31.5, ios="26.6.0")
    archive2 = write_archive(tmp_path / "b", manifest(), files, full_energy_files())
    report2 = an.analyze_archive(archive2)
    assert report2.speed["ours"].decode_warm.n == 3  # 4 warm rows offered, 1 refused
    assert any("iOS" in r and "ours-decode-warm-1" in r for r in report2.refusals)


# --------------------------------------------------------------------------
# Speed-session analysis
# --------------------------------------------------------------------------

def test_speed_headlines_medians_verdict_and_tripwires(tmp_path):
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(), full_energy_files())
    r = an.analyze_archive(archive)
    ours, mlx, llama = r.speed["ours"], r.speed["mlx"], r.speed["llama"]
    assert ours.decode_warm.median == pytest.approx(31.65)  # 31.7, 31.5, 31.9, 31.6 (bookends count as warm)
    assert ours.decode_warm.n == 4 and ours.decode_cold.median == pytest.approx(31.1)
    assert mlx.decode_warm.median == pytest.approx(39.2) and llama.decode_warm.median == pytest.approx(32.4)
    assert ours.prefill_warm.median == pytest.approx(852 / 3.637, rel=1e-3) and ours.prefill_warm.token_count == 852
    assert mlx.prefill_warm.token_count == 852 and llama.prefill_warm.token_count == 852
    assert r.bookend_delta == pytest.approx(31.6 - 31.7)
    assert r.verdict.ratio == pytest.approx(31.65 / 39.2) and r.verdict.met is True
    assert r.verdict.constant_29_4 == 29.4
    assert r.tripwires["decode"].passed is True and r.tripwires["decode"].floor == 24.0
    assert r.tripwires["prefill"].passed is True and r.tripwires["prefill"].floor == 135.0
    assert ours.footprint_mb["ours-mmap"] == 573.2 and mlx.footprint_mb["mlx"] == 1020.0


def test_verdict_not_met_when_ratio_below_three_quarters(tmp_path):
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(ours_window=(28.0, 28.1, 27.9), mlx_window=(40.0, 40.0, 40.0)), {})
    r = an.analyze_archive(archive)
    assert r.verdict.met is False and r.verdict.ratio < 0.75


def test_mlx_mismatch_refuses_row_and_template_mode_is_labeled(tmp_path):
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(mlx_parity="MISMATCH"), {})
    r = an.analyze_archive(archive)
    assert r.speed["mlx"].decode_warm.n == 0 and r.speed["mlx"].decode_warm.median is None
    assert r.verdict.met is None and any("MLX" in x for x in r.verdict.refusals)
    archive2 = write_archive(tmp_path / "b", manifest(), full_speed_files(mlx_parity="template"), {})
    r2 = an.analyze_archive(archive2)
    assert r2.speed["mlx"].decode_warm.n == 3
    assert any("template" in n for n in r2.speed["mlx"].notes)


def test_llama_count_differs_refuses_row(tmp_path):
    files = full_speed_files()
    files["llama-decode-warm-0.txt"] = external(engine="llama-swiftui", window=32.4, prompt_line=llama_prompt_line(count=95, differs=True))
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), files, {}))
    assert r.speed["llama"].decode_warm.n == 2
    assert any("llama-decode-warm-0" in x and "count" in x for x in r.refusals)


def test_provisional_and_wrong_round_rows_are_not_headline(tmp_path):
    files = full_speed_files()
    files["ours-decode-warm-1.txt"] = ours_burst(window=31.5, round_=None)
    files["ours-decode-warm-2.txt"] = ours_burst(window=31.9, round_="phase6-speed-2026-09-30")
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), files, {}))
    assert r.speed["ours"].decode_warm.n == 2
    assert any("PROVISIONAL" in x for x in r.refusals) and any("round" in x for x in r.refusals)


def test_decode_row_without_window_is_refused(tmp_path):
    files = full_speed_files()
    files["ours-decode-warm-1.txt"] = ours_burst(generated=300, window=None, stop="eos")
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), files, {}))
    assert r.speed["ours"].decode_warm.n == 3
    assert any("window" in x for x in r.refusals)


def test_sustained_rows_report_first_last_and_plateau(tmp_path):
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), full_speed_files(), {}))
    s = r.speed["ours"].sustained
    assert s.generations == 6 and s.first_window == pytest.approx(31.5) and s.last_window == pytest.approx(24.0)
    assert s.plateau_second_half_median == pytest.approx(25.5)  # windows 27.0, 25.5, 24.0
    assert r.speed["mlx"].sustained.first_window == pytest.approx(39.0)


def test_bandwidth_pair_median_and_ratio(tmp_path):
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), full_speed_files(), {}))
    bw = r.bandwidth
    assert bw["triad"].median == pytest.approx((43.84 + 43.70) / 2) and bw["triad"].n == 2
    assert bw["read-only"].median == pytest.approx((45.10 + 44.90) / 2)
    assert bw["ratio_readonly_over_triad"] == pytest.approx([45.10 / 43.84, 44.90 / 43.70])


def test_bandwidth_diagnostic_export_in_archive_is_refused_not_averaged(tmp_path):
    files = full_speed_files()
    files["bandwidth-readonly-2.txt"] = bandwidth("read-only", 60.0, diagnostic=True)
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), files, {}))
    assert r.bandwidth["read-only"].n == 2 and any("DIAGNOSTIC" in x for x in r.refusals)


def test_roofline_implied_bandwidth_uses_measured_denominator(tmp_path):
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), full_speed_files(), {}))
    ro = r.roofline["ours"]
    assert ro.bytes_per_token == an.WEIGHT_BYTES + an.KV_BYTES_PER_POSITION * (84 + 320)
    assert ro.implied_gbps == pytest.approx(31.65 * ro.bytes_per_token / 1e9)
    assert ro.fraction_of_triad == pytest.approx(ro.implied_gbps / ((43.84 + 43.70) / 2))


# --------------------------------------------------------------------------
# Edge test 6 — figures render from the D1 timeline, never the cumulative line
# --------------------------------------------------------------------------

def test_thermal_chart_series_come_from_timeline_entries_not_cumulative(tmp_path):
    files = full_energy_files()
    # a deliberately inconsistent cumulative line on the mlx cycle must not reach the chart
    files["mlx-energy-0.txt"] = external(mode="energy-loop", round_=ROUND_ENERGY,
                                         gens=default_gens(n=18, window0=39.0, wall=45.0), cumulative_tokens=5)
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(), files)
    r = an.analyze_archive(archive)
    series = an.thermal_series(r)
    ours_cycles = [s for s in series if s.engine == "ours"]
    assert len(ours_cycles) == 3
    assert ours_cycles[0].source == "json"  # the JSON timeline is preferred when archived
    assert ours_cycles[0].elapsed_s[0] == pytest.approx(58.0) and ours_cycles[0].window_tok_s[0] == pytest.approx(31.5)
    assert len(ours_cycles[0].window_tok_s) == 18
    mlx0 = [s for s in series if s.engine == "mlx"][0]
    assert mlx0.window_tok_s[0] == pytest.approx(39.0) and len(mlx0.window_tok_s) == 18
    out = an.render_thermal_chart(series, tmp_path / "thermal.png")
    assert out.exists() and out.stat().st_size > 1000


def test_energy_bars_render_with_both_error_bars(tmp_path):
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(), full_energy_files(n_llama=2))
    r = an.analyze_archive(archive)
    out = an.render_energy_bars(r.energy, tmp_path / "energy.png")
    assert out.exists() and out.stat().st_size > 1000
    bars = an.energy_bar_data(r.energy)
    assert [b.engine for b in bars] == ["ours", "mlx"]  # llama (n=2) is not drawn as a headline bar
    assert all(b.statistical_halfwidth >= 0 and b.systematic_halfwidth > 0 for b in bars)


# --------------------------------------------------------------------------
# CLI end-to-end
# --------------------------------------------------------------------------

def test_cli_writes_tables_figures_and_refusals(tmp_path):
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(), full_energy_files(n_llama=2))
    out = tmp_path / "out"
    rc = an.main([str(archive), "--out", str(out)])
    assert rc == 0
    tables = (out / "tables.md").read_text()
    assert "31.65" in tables and "39.20" in tables
    assert "below the ≥3 pin" in tables  # llama energy labeled, never averaged in
    assert (out / "thermal.png").exists() and (out / "energy.png").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["verdict"]["met"] is True
    assert summary["energy"]["llama"]["headline"] is None


def test_cli_exit_code_two_when_manifest_missing(tmp_path):
    (tmp_path / "empty").mkdir()
    assert an.main([str(tmp_path / "empty"), "--out", str(tmp_path / "o")]) == 2


# --------------------------------------------------------------------------
# Post-task review additions: protocol-shape refusals
# --------------------------------------------------------------------------

def test_cycle_not_ended_by_operator_stop_is_refused():
    idle = an.IdleBaseline(100, 80, 75, 4500, IOS)
    bound = ours_loop(gens=default_gens(n=18, wall=58.0),
                      ended="duration bound (safety cap — NOT the protocol's operator stop)")
    with pytest.raises(an.Refusal, match="operator stop"):
        an.analyze_energy_cycle(ex.parse_export_text(bound), idle)
    halt = external(mode="energy-loop", round_=ROUND_ENERGY, gens=default_gens(n=18, wall=45.0), ended_by_operator=False)
    with pytest.raises(an.Refusal, match="operator stop"):
        an.analyze_energy_cycle(ex.parse_export_text(halt), idle)


def test_loop_on_wrong_prompt_is_refused(tmp_path):
    files = full_energy_files()
    files["ours-energy-1.txt"] = ours_loop(gens=default_gens(n=18, wall=58.0)).replace(
        "prompt: decode-essay (84 prompt tokens)", "prompt: prefill-summarize (852 prompt tokens)")
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), full_speed_files(), files))
    assert r.energy["ours"].n_valid == 2 and r.energy["ours"].headline is None
    assert any("ours-energy-1" in x and "decode-essay" in x for x in r.refusals)
    speed = full_speed_files()
    speed["ours-decode-warm-1.txt"] = ours_burst(prompt="other-prompt", tokens=10, window=31.5)
    r2 = an.analyze_archive(write_archive(tmp_path / "b", manifest(), speed, {}))
    assert r2.speed["ours"].decode_warm.n == 3 and any("other-prompt" in x for x in r2.refusals)


def test_roofline_depth_uses_each_engines_prompt_token_count(tmp_path):
    m = manifest()
    m["roofline"] = {"weight_bytes": {"llama": 1_190_000_000}}
    r = an.analyze_archive(write_archive(tmp_path / "a", m, full_speed_files(), {}))
    assert r.roofline["llama"].bytes_per_token == 1_190_000_000 + an.KV_BYTES_PER_POSITION * (92 + 320)


def test_llama_count_equal_to_hf_count_is_parity(tmp_path):
    """P6-4 finding: the P6-2 harness carries 92 as the llama.cpp decode-essay count on
    record, but 92 was the Phase 0 Mac `llama-completion` figure; the Phase 0 device
    rows (results.md) and the P6-4 rows read 84 = the HF count. A count equal to the
    HF/qwen-metal count is the strongest parity outcome and must not be refused."""
    line = ("prompt: decode-essay (84 tokens, llama.cpp tokenizer, parse_special=true; feeding: RENDERED "
            "bundled string; DIFFERS from the llama.cpp count on record (92) — investigate before using "
            "the row; HF/qwen-metal count 84)")
    row = ex.parse_export_text(external(engine="llama-swiftui", prompt_line=line))
    assert row.llama_count_matches is True and row.prompt_token_count == 84
    truly_off = ex.parse_export_text(external(engine="llama-swiftui", prompt_line=llama_prompt_line(count=95, differs=True)))
    assert truly_off.llama_count_matches is False


def test_manifest_operator_overrides_supply_missing_fields_and_are_labeled(tmp_path):
    """P6-4 (decided by James 2026-09-27): rows exported without the round marker /
    cold-warm / health typed are accepted when the operator supplies those fields in
    manifest.json; the export text is never edited, the override is labeled, and an
    export whose OWN round disagrees with the session is still refused."""
    files = full_speed_files()
    files["ours-decode-warm-1.txt"] = ours_burst(window=31.5, round_=None)
    files["mlx-decode-warm-1.txt"] = external(window=39.0, round_=None, cold_warm="(record manually)", health=None)
    files["ours-decode-warm-2.txt"] = ours_burst(window=31.9, round_="phase6-speed-2026-09-30")
    m = manifest()
    m["sessions"]["speed"]["operator_overrides"] = {
        "note": "typed post hoc by the operator",
        "defaults": {"round": ROUND_SPEED, "battery_health_percent": 100},
        "rows": {"mlx-decode-warm-1": {"cold_warm": "warm"}},
    }
    r = an.analyze_archive(write_archive(tmp_path / "a", m, files, {}))
    assert r.speed["ours"].decode_warm.n == 3  # warm-1 rescued, warm-2 (own round disagrees) still refused
    assert r.speed["mlx"].decode_warm.n == 3
    assert any("ours-decode-warm-2" in x and "round" in x for x in r.refusals)
    assert any("ours-decode-warm-1" in x and "operator-supplied" in x for x in r.notes)
    assert any("mlx-decode-warm-1" in x and "cold/warm" in x for x in r.notes)
    # without the override block the same rows are refused (the existing rule)
    r2 = an.analyze_archive(write_archive(tmp_path / "b", manifest(), files, {}))
    assert r2.speed["ours"].decode_warm.n == 2 and r2.speed["mlx"].decode_warm.n == 2


def test_soc_marks_can_be_operator_supplied_and_are_labeled(tmp_path):
    """P6-5 (decided by James 2026-10-13): the external panels' SoC marks were left
    blank; the operator supplies them via manifest overrides, labeled."""
    files = full_energy_files(n_mlx=0, n_llama=0)
    files["mlx-energy-0.txt"] = external(mode="energy-loop", round_=None, health=None,
                                         soc=(None, None, 70.0, 60.0), gens=default_gens(n=18, window0=39.0, wall=45.0))
    m = manifest()
    m["sessions"]["energy"]["operator_overrides"] = {
        "defaults": {"round": ROUND_ENERGY, "battery_health_percent": 100},
        "rows": {"mlx-energy-0": {"cold_warm": "warm", "soc_start": 80, "soc_end": 70}},
    }
    r = an.analyze_archive(write_archive(tmp_path / "a", m, full_speed_files(), files))
    assert len(r.energy["mlx"].cycles) == 1 and r.energy["mlx"].cycles[0].delta_soc == 10
    assert any("mlx-energy-0" in x and "SoC marks" in x for x in r.notes)


def test_any_start_band_rule_accepts_off_band_cycles_labeled(tmp_path):
    """P6-5 (decided by James 2026-10-13): the 80→70 start anchor is waived for the
    round; ≥8% burned and 3–9 W still bind; off-band cycles are labeled."""
    files = full_energy_files(n_mlx=0, n_llama=0, ours_soc=[(99, 89, 100.0, 90.0), (80, 70, 80.0, 70.0), (71, 61, 70.0, 60.0)])
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(), files)
    r = an.analyze_archive(archive)
    assert r.energy["ours"].n_valid == 1  # the pinned rule: two off-band cycles INVALID
    m = manifest()
    m["sessions"]["energy"]["band_start_rule"] = {"rule": "any-start", "decision": "James 2026-10-13"}
    r2 = an.analyze_archive(write_archive(tmp_path / "b", m, full_speed_files(), files))
    assert r2.energy["ours"].n_valid == 3 and r2.energy["ours"].headline is not None
    off = [c for c in r2.energy["ours"].cycles if c.off_band_start]
    assert [c.soc_start for c in off] == [99, 71]
    assert any("off-band start" in n for n in r2.energy["ours"].notes)
    shallow = ours_loop(soc=(99, 93, 99.0, 93.0), gens=default_gens(n=18, wall=58.0))
    files["ours-energy-3.txt"] = shallow
    r3 = an.analyze_archive(write_archive(tmp_path / "c", m, full_speed_files(), files))
    assert r3.energy["ours"].n_valid == 3  # the ≥8% floor still binds under any-start


def test_round_aliases_map_app_typed_markers_to_the_session(tmp_path):
    """P6-5: the operator typed '1'/'2'/'3' (the Latin-square round index) into the
    apps instead of the session marker; a listed alias is accepted and labeled, an
    unlisted marker is still refused."""
    files = full_energy_files(n_mlx=0, n_llama=0)
    files["ours-energy-1.txt"] = ours_loop(round_="3", gens=default_gens(n=18, wall=58.0))
    files["ours-energy-2.txt"] = ours_loop(round_="7", gens=default_gens(n=18, wall=58.0))
    m = manifest()
    m["sessions"]["energy"]["round_aliases"] = ["1", "2", "3"]
    r = an.analyze_archive(write_archive(tmp_path / "a", m, full_speed_files(), files))
    assert len(r.energy["ours"].cycles) == 2
    assert any("ours-energy-1" in x and "alias" in x for x in r.notes)
    assert any("ours-energy-2" in x and "round" in x for x in r.refusals)


def test_programmatic_soc_disagreement_is_noted_not_refused():
    idle = an.IdleBaseline(100, 80, 75, 4500, IOS)
    row = ex.parse_export_text(ours_loop(soc=(83, 73, 85.0, 70.0), gens=default_gens(n=18, wall=58.0)))
    c = an.analyze_energy_cycle(row, idle, require_band_start=False)
    assert c.valid and c.cross_check_note and "15" in c.cross_check_note
    ok = an.analyze_energy_cycle(ex.parse_export_text(ours_loop(gens=default_gens(n=18, wall=58.0))), idle)
    assert ok.cross_check_note is None


def test_in_band_subset_is_reported_alongside_the_any_start_headline(tmp_path):
    """P6-5 round 4: with the any-start rule the headline mean covers every valid cycle;
    the cycles that honored the pinned start anchor (start == 80) are summarized
    separately so the writeup can cite the protocol number too."""
    files = full_energy_files(n_mlx=0, n_llama=0,
                              ours_soc=[(99, 89, 100.0, 90.0), (80, 70, 80.0, 70.0), (80, 69, 80.0, 70.0)])
    m = manifest()
    m["sessions"]["energy"]["band_start_rule"] = {"rule": "any-start", "decision": "test"}
    r = an.analyze_archive(write_archive(tmp_path / "a", m, full_speed_files(), files))
    e = r.energy["ours"]
    assert e.headline.n == 3 and e.in_band.n == 2
    assert e.in_band.values == pytest.approx([c.net_joules_per_token for c in e.cycles if c.soc_start == 80])
    assert e.in_band.mean_j_per_token == pytest.approx(sum(e.in_band.values) / 2)


# --------------------------------------------------------------------------
# P6-6 additions (writeup §5): roofline rows for every engine with a labeled
# basis, the 43.84 GB/s denominator of record alongside the session triad,
# the derived ceilings, and the roofline figure
# --------------------------------------------------------------------------

def test_roofline_external_engines_on_our_bytes_basis_are_labeled(tmp_path):
    """The external engines' per-token byte counts are not measured; without a
    manifest figure their implied bandwidth is reported on the qwen-metal
    bytes/token basis and LABELED as such (never silently as measured)."""
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), full_speed_files(), {}))
    assert set(r.roofline) == {"ours", "mlx", "llama"}
    assert r.roofline["ours"].basis == an.BASIS_MEASURED
    mlx = r.roofline["mlx"]
    assert mlx.basis == an.BASIS_OURS_PROXY
    assert mlx.bytes_per_token == an.WEIGHT_BYTES + an.KV_BYTES_PER_POSITION * (84 + 320)
    assert mlx.implied_gbps == pytest.approx(39.2 * mlx.bytes_per_token / 1e9)
    llama = r.roofline["llama"]
    assert llama.basis == an.BASIS_OURS_PROXY
    assert llama.bytes_per_token == an.WEIGHT_BYTES + an.KV_BYTES_PER_POSITION * (92 + 320)


def test_roofline_manifest_weight_bytes_override_the_proxy_basis(tmp_path):
    m = manifest()
    m["roofline"] = {"weight_bytes": {"llama": 1_190_000_000}}
    r = an.analyze_archive(write_archive(tmp_path / "a", m, full_speed_files(), {}))
    assert r.roofline["llama"].basis == an.BASIS_MANIFEST
    assert r.roofline["llama"].bytes_per_token == 1_190_000_000 + an.KV_BYTES_PER_POSITION * (92 + 320)
    assert r.roofline["mlx"].basis == an.BASIS_OURS_PROXY


def test_roofline_fraction_of_record_denominators_and_ceilings(tmp_path):
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), full_speed_files(), {}))
    ro = r.roofline["ours"]
    session_triad = (43.84 + 43.70) / 2
    assert an.ROOFLINE_GBPS_OF_RECORD == 43.84  # DECISIONS.md 2026-08-22, consumed verbatim
    assert ro.fraction_of_record == pytest.approx(ro.implied_gbps / an.ROOFLINE_GBPS_OF_RECORD)
    assert ro.ceiling_tok_s_record == pytest.approx(an.ROOFLINE_GBPS_OF_RECORD * 1e9 / ro.bytes_per_token)
    assert ro.ceiling_tok_s_session == pytest.approx(session_triad * 1e9 / ro.bytes_per_token)
    d = r.roofline_denominators
    assert d["triad_of_record_gbps"] == an.ROOFLINE_GBPS_OF_RECORD
    assert d["triad_session_gbps"] == pytest.approx(session_triad)
    assert d["readonly_session_gbps"] == pytest.approx((45.10 + 44.90) / 2)
    assert d["sequential_prefill_ceiling_tok_s"] == pytest.approx(an.ROOFLINE_GBPS_OF_RECORD * 1e9 / an.WEIGHT_BYTES)
    assert d["weight_bytes"] == an.WEIGHT_BYTES and d["kv_bytes_per_position"] == an.KV_BYTES_PER_POSITION


def test_roofline_without_bandwidth_rows_keeps_only_the_record_denominator(tmp_path):
    files = {k: v for k, v in full_speed_files().items() if not k.startswith("bandwidth")}
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), files, {}))
    ro = r.roofline["ours"]
    assert ro.fraction_of_triad is None and ro.ceiling_tok_s_session is None
    assert ro.fraction_of_record is not None and ro.ceiling_tok_s_record is not None
    assert r.roofline_denominators["triad_session_gbps"] is None
    assert r.roofline_denominators["readonly_session_gbps"] is None


def test_roofline_engine_without_a_window_row_has_no_implied_bandwidth(tmp_path):
    files = {k: v for k, v in full_speed_files().items() if not k.startswith("mlx-decode")}
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), files, {}))
    assert r.roofline["mlx"].implied_gbps is None and r.roofline["mlx"].fraction_of_record is None
    assert [b.engine for b in an.roofline_bar_data(r.roofline)] == ["ours", "llama"]


def test_roofline_figure_renders_every_engine_with_the_denominator_lines(tmp_path):
    r = an.analyze_archive(write_archive(tmp_path / "a", manifest(), full_speed_files(), {}))
    bars = an.roofline_bar_data(r.roofline)
    assert [b.engine for b in bars] == ["ours", "mlx", "llama"]
    assert bars[0].implied_gbps == pytest.approx(r.roofline["ours"].implied_gbps)
    assert bars[1].basis == an.BASIS_OURS_PROXY
    out = an.render_roofline_chart(bars, r.roofline_denominators, tmp_path / "roofline.png")
    assert out.exists() and out.stat().st_size > 1000


def test_cli_writes_roofline_figure_tables_and_denominators(tmp_path):
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(), full_energy_files())
    out = tmp_path / "out"
    assert an.main([str(archive), "--out", str(out)]) == 0
    assert (out / "roofline.png").exists()
    tables = (out / "tables.md").read_text()
    assert an.BASIS_OURS_PROXY in tables and "43.84" in tables
    summary = json.loads((out / "summary.json").read_text())
    assert summary["roofline_denominators"]["triad_of_record_gbps"] == 43.84
    assert summary["roofline"]["mlx"]["basis"] == an.BASIS_OURS_PROXY


def test_energy_cycles_double_as_thermal_rows_in_report_and_tables(tmp_path):
    """D4: each energy cycle is the sustained-thermal run of record; the report lists
    first/last/second-half-median window rates per cycle (from the D1 timeline), so
    the writeup's §3 plateau readings are regenerated, never typed."""
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(), full_energy_files())
    r = an.analyze_archive(archive)
    rows = r.thermal_cycles
    assert [t.file for t in rows] == ["llama-energy-0", "llama-energy-1", "llama-energy-2",
                                      "mlx-energy-0", "mlx-energy-1", "mlx-energy-2",
                                      "ours-energy-0", "ours-energy-1", "ours-energy-2"]
    ours0 = next(t for t in rows if t.file == "ours-energy-0")
    assert ours0.generations == 18 and ours0.first_window == pytest.approx(31.5)
    assert ours0.last_window == pytest.approx(31.5 - 17 * 1.5)
    assert ours0.plateau_second_half_median is not None and ours0.minutes == pytest.approx((18 * 58.0 + 3.0) / 60)
    from phase6_tables import render_tables
    tables = render_tables(r)
    assert "Energy cycles as thermal runs" in tables and "ours-energy-0" in tables
    summary = an.summary_dict(r)
    assert summary["thermal_cycles"][0]["file"] == "llama-energy-0"


def test_idle_baseline_watts_are_reported(tmp_path):
    archive = write_archive(tmp_path / "a", manifest(), full_speed_files(), full_energy_files())
    r = an.analyze_archive(archive)
    idle = r.idle_summary
    assert idle["soc_drop_percent"] == 5 and idle["wall_seconds"] == 4500
    assert idle["joules_per_percent"] == pytest.approx(453.6)
    assert idle["watts"] == pytest.approx(5 * 453.6 / 4500)
    from phase6_tables import render_tables
    assert f"{idle['watts']:.3f} W" in render_tables(r)
    assert an.summary_dict(r)["idle_baseline"]["watts"] == pytest.approx(idle["watts"])


def test_idle_summary_is_none_when_the_baseline_is_refused(tmp_path):
    archive = write_archive(tmp_path / "a", manifest(idle=False), full_speed_files(), full_energy_files())
    r = an.analyze_archive(archive)
    assert r.idle_summary is None and an.summary_dict(r)["idle_baseline"] is None
