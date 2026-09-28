#!/usr/bin/env python3
"""Phase 6 analyzer: archived exports → tables + figures (phase-6.md D8; task P6-3).

Every number the Phase 6 writeup shows is regenerated here from the raw
exports archived under benchmarks/phase6/ (layout: benchmarks/phase6/README.md)
— none is typed by hand (METHODOLOGY 4). The script consumes the pre-committed
validity criteria and reused tripwires verbatim (DECISIONS.md 2026-09-23
"Phase 6 gates pre-committed"; hard rule 6 — never adjusted here) and REFUSES
a headline value whenever an input violates one: the refusal is written into
the tables in place of a number, never averaged around.

Usage:
    .venv/bin/python tools/phase6_analyze.py benchmarks/phase6 --out docs/writeup
    (--no-figures skips the PNGs; exit 0 = analysis written, 2 = archive unreadable)

Outputs: tables.md (every table + the refusal ledger), summary.json (the same
values, machine-readable), thermal.png (D4 sustained-thermal chart from the D1
timeline), energy.png (net J/token with both error bars).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase6_exports as ex  # noqa: E402
from phase6_exports import Refusal  # noqa: E402,F401  (re-exported for callers)

# --- Pre-committed constants, consumed verbatim (never tuned here) -----------
RATED_WATT_HOURS = 12.6                       # iPhone 15 Pro rated capacity (DECISIONS 2026-09-05)
JOULES_PER_PERCENT_AT_FULL_HEALTH = RATED_WATT_HOURS * 36   # 453.6 J per 1% SoC at 100% health
WATTS_ANCHOR = (3.0, 9.0)                     # PLAN energy protocol: implied gross W must land here
MIN_SOC_BURN_PERCENT = 8.0                    # PLAN: ≥8% burned per cycle
BAND_START_PERCENT = 80.0                     # the 80→70 band anchor (start mark of record)
MIN_VALID_CYCLES = 3                          # PLAN: ≥3 repeats per engine
MIN_IDLE_DROP_PERCENT = 4.0                   # D7 gate: idle baseline ≥4% SoC drop
SOC_READING_QUANTUM_PERCENT = 0.5             # ±0.5% per Settings reading (systematic bar, D4)
TRIPWIRE_DECODE_TOK_S = 24.0                  # Phase 4/5 constant, reused (D7)
TRIPWIRE_PREFILL_TOK_S = 135.0                # Phase 5 constant, reused (D7)
SUCCESS_RATIO = 0.75                          # PLAN success metric: ours / MLX ≥ 0.75
CONSTANT_29_4 = 29.4                          # the Phase 0 planning constant, reported alongside
WEIGHT_BYTES = 967_753_728                    # packed q4g64 weights streamed per token (P4-EXEC)
KV_BYTES_PER_POSITION = 28 * 2 * 8 * 128 * 2  # 28 layers × K,V × 8 kv heads × head_dim 128 × fp16
WINDOW_MEDIAN_DEPTH_OFFSET = (128 + 512) // 2  # generated-token depth at the window's median
DEFAULT_DECODE_PROMPT_TOKENS = 84             # decode-essay, HF count (used only when no decode row parsed)
ENGINES = ("ours", "mlx", "llama")
DECODE_PROMPT = "decode-essay"                # decode rows, sustained loops, energy cycles (PLAN pins)
PREFILL_PROMPT = "prefill-summarize"          # prefill rows
ENERGY_PROMPT = DECODE_PROMPT
ENGINE_LABELS = {"ours": "qwen-metal", "mlx": "MLX LLMEval", "llama": "llama.cpp"}
# dataviz reference palette, categorical slots 1–3 (validated all-pairs, both modes)
ENGINE_COLORS = {"ours": "#2a78d6", "mlx": "#eb6834", "llama": "#1baf7a"}


# --------------------------------------------------------------------------
# Result shapes
# --------------------------------------------------------------------------

@dataclass
class IdleBaseline:
    health_percent: Optional[float]
    soc_start: Optional[float]
    soc_end: Optional[float]
    wall_seconds: Optional[float]
    ios_build: Optional[str]

    @property
    def soc_drop(self) -> float:
        return float(self.soc_start) - float(self.soc_end)

    def validate(self) -> None:
        if None in (self.health_percent, self.soc_start, self.soc_end, self.wall_seconds):
            raise Refusal("idle baseline incomplete (health, SoC start/end, wall seconds all required)")
        if self.soc_drop < MIN_IDLE_DROP_PERCENT:
            raise Refusal(f"idle baseline dropped {self.soc_drop:g}% SoC — below the ≥4% rule")
        if float(self.wall_seconds) <= 0:
            raise Refusal("idle baseline wall time must be positive")


@dataclass
class EnergyCycle:
    file: str
    engine: str
    health_percent: float
    delta_soc: float
    cycle_wall_s: float
    tokens: int
    gross_joules: float
    implied_watts: float
    idle_soc_scaled: float
    net_joules: float
    net_joules_per_token: float
    net_watts: float
    systematic_fraction: float
    valid: bool
    invalid_reasons: list[str]
    soc_start: float
    soc_end: float


@dataclass
class EnergyHeadline:
    n: int
    mean_j_per_token: float
    spread: tuple[float, float]
    systematic_fraction: float
    implied_watts_mean: float
    values: list[float]


@dataclass
class EngineEnergy:
    engine: str
    cycles: list[EnergyCycle] = field(default_factory=list)
    n_valid: int = 0
    headline: Optional[EnergyHeadline] = None
    refusals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Stat:
    values: list[float] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    token_count: Optional[int] = None

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def median(self) -> Optional[float]:
        return statistics.median(self.values) if self.values else None


@dataclass
class Sustained:
    file: str
    generations: int
    first_window: Optional[float]
    last_window: Optional[float]
    plateau_second_half_median: Optional[float]
    minutes: Optional[float]
    basis: str


@dataclass
class EngineSpeed:
    engine: str
    decode_warm: Stat = field(default_factory=Stat)
    decode_cold: Stat = field(default_factory=Stat)
    prefill_warm: Stat = field(default_factory=Stat)
    prefill_cold: Stat = field(default_factory=Stat)
    sustained: Optional[Sustained] = None
    footprint_mb: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    ours_median: Optional[float]
    mlx_median: Optional[float]
    ratio: Optional[float]
    met: Optional[bool]
    constant_29_4: float = CONSTANT_29_4
    refusals: list[str] = field(default_factory=list)


@dataclass
class Tripwire:
    name: str
    floor: float
    value: Optional[float]
    passed: Optional[bool]


@dataclass
class BandwidthStat:
    kernel: str
    values: list[float] = field(default_factory=list)
    spreads: list[tuple[float, float]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def median(self) -> Optional[float]:
        return statistics.median(self.values) if self.values else None


@dataclass
class Roofline:
    engine: str
    window_tok_s: Optional[float]
    bytes_per_token: int
    implied_gbps: Optional[float]
    denominator_gbps: Optional[float]
    fraction_of_triad: Optional[float]


@dataclass
class ThermalSeries:
    engine: str
    file: str
    elapsed_s: list[float]
    window_tok_s: list[float]
    source: str          # json | text
    basis: str           # loopClock | sumOfGenerationWall
    bold: bool = False


@dataclass
class EnergyBar:
    engine: str
    mean: float
    statistical_halfwidth: float
    systematic_halfwidth: float


@dataclass
class Report:
    manifest: dict
    speed: dict
    energy: dict
    bandwidth: dict
    roofline: dict
    verdict: Verdict
    tripwires: dict
    bookend_delta: Optional[float]
    refusals: list[str]
    notes: list[str]
    energy_rows: list  # (file, RowExport, Timeline|None) for the thermal chart


# --------------------------------------------------------------------------
# Energy arithmetic (edge test 5)
# --------------------------------------------------------------------------

def joules_per_soc_percent(health_percent: Optional[float]) -> float:
    """J per 1% SoC from battery HEALTH (max capacity %) — never from SoC."""
    if health_percent is None:
        raise Refusal("battery health (maximum capacity %) not recorded — capacity basis undefined")
    return JOULES_PER_PERCENT_AT_FULL_HEALTH * float(health_percent) / 100.0


def analyze_energy_cycle(row: ex.RowExport, idle: IdleBaseline, file: str = "") -> EnergyCycle:
    """One D4 cycle: gross → idle-subtracted net J/token, watts, validity."""
    j_per_pct = joules_per_soc_percent(row.battery_health_percent)
    if row.soc_start is None or row.soc_end is None:
        raise Refusal(f"{file}: state of charge at the band marks not recorded")
    if row.cumulative is None or not row.generations:
        raise Refusal(f"{file}: energy export lacks the cumulative line or the per-generation timeline")
    if not (row.ended_by or "").startswith("operator stop"):
        raise Refusal(f"{file}: cycle ended by {row.ended_by!r}, not the protocol's operator stop")
    if row.prompt_name != ENERGY_PROMPT:
        raise Refusal(f"{file}: energy cycle ran {row.prompt_name!r}, not the pinned {ENERGY_PROMPT}")
    summed = sum(g.tokens for g in row.generations)
    if summed != row.cumulative.tokens:
        raise Refusal(f"{file}: cumulative tokens {row.cumulative.tokens} ≠ Σ per-generation tokens {summed}")
    idle.validate()
    delta = row.soc_start - row.soc_end
    wall = row.cumulative.cycle_wall_s
    gross = delta * j_per_pct
    watts = gross / wall
    idle_scaled = idle.soc_drop * wall / float(idle.wall_seconds)
    net_delta = delta - idle_scaled
    net = net_delta * j_per_pct
    reasons = _cycle_validity(row.soc_start, delta, watts)
    quantum = 2 * SOC_READING_QUANTUM_PERCENT
    systematic = (quantum + idle_scaled * quantum / idle.soc_drop) / net_delta if net_delta > 0 else float("inf")
    return EnergyCycle(
        file=file, engine=row.engine, health_percent=row.battery_health_percent, delta_soc=delta,
        cycle_wall_s=wall, tokens=summed, gross_joules=gross, implied_watts=watts,
        idle_soc_scaled=idle_scaled, net_joules=net, net_joules_per_token=net / summed,
        net_watts=net / wall, systematic_fraction=systematic, valid=not reasons,
        invalid_reasons=reasons, soc_start=row.soc_start, soc_end=row.soc_end)


def _cycle_validity(soc_start: float, delta: float, watts: float) -> list[str]:
    reasons = []
    if soc_start != BAND_START_PERCENT:
        reasons.append(f"start mark {soc_start:g}% is not the 80% band anchor")
    if delta < MIN_SOC_BURN_PERCENT:
        reasons.append(f"burned {delta:g}% SoC — below the ≥8% floor")
    if not (WATTS_ANCHOR[0] <= watts <= WATTS_ANCHOR[1]):
        reasons.append(f"implied gross {watts:.2f} W outside the 3–9 W anchor")
    return reasons


def summarize_engine_energy(engine: str, cycles: list[EnergyCycle], refusals: list[str]) -> EngineEnergy:
    valid = [c for c in cycles if c.valid]
    result = EngineEnergy(engine=engine, cycles=cycles, n_valid=len(valid), refusals=list(refusals))
    for c in cycles:
        if not c.valid:
            result.notes.append(f"{c.file}: INVALID — " + "; ".join(c.invalid_reasons))
    if refusals:
        return result
    if len(valid) < MIN_VALID_CYCLES:
        result.refusals.append(
            f"{len(valid)} valid cycle(s) — below the ≥3 pin — not a comparative energy row")
        return result
    values = [c.net_joules_per_token for c in valid]
    result.headline = EnergyHeadline(
        n=len(valid), mean_j_per_token=statistics.fmean(values), spread=(min(values), max(values)),
        systematic_fraction=statistics.fmean(c.systematic_fraction for c in valid),
        implied_watts_mean=statistics.fmean(c.implied_watts for c in valid), values=values)
    return result


# --------------------------------------------------------------------------
# Archive loading
# --------------------------------------------------------------------------

def load_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.exists():
        raise Refusal(f"{path} missing — the operator-typed session facts are required")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise Refusal(f"{path} does not parse: {e}") from e


def load_session_rows(root: Path, session: dict, refusals: list[str]):
    """Yield (name, RowExport|BandwidthExport, Timeline|None) for one session dir."""
    directory = root / session.get("dir", "")
    if not directory.is_dir():
        raise Refusal(f"session directory {directory} missing")
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text()
        try:
            if ex.is_bandwidth_export(text):
                yield path.stem, ex.parse_bandwidth_export(text), None
                continue
            row = ex.parse_export_text(text)
        except Refusal as e:
            refusals.append(f"{path.stem}: {e}")
            continue
        except ex.ParseError as e:
            refusals.append(f"{path.stem}: unreadable export — {e}")
            continue
        timeline = _sidecar_timeline(path, refusals)
        yield path.stem, row, timeline


def _sidecar_timeline(path: Path, refusals: list[str]) -> Optional[ex.Timeline]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        return ex.parse_timeline_json(sidecar.read_text())
    except ex.ParseError as e:
        refusals.append(f"{sidecar.name}: unreadable timeline JSON — {e}")
        return None


def headline_eligibility(row: ex.RowExport, session: dict, name: str) -> tuple[list[str], list[str]]:
    """(refusal reasons, notes) for a row against the D7 validity criteria."""
    reasons, notes = [], []
    if row.provisional:
        reasons.append("PROVISIONAL export (no round marker)")
    elif row.round != session.get("round"):
        reasons.append(f"round marker {row.round!r} ≠ session round {session.get('round')!r}")
    if row.engine == "ours" and row.os_version != session.get("ios_build"):
        reasons.append(f"export iOS {row.os_version!r} ≠ session iOS build {session.get('ios_build')!r}")
    if row.engine == "mlx":
        if row.parity == "MISMATCH":
            reasons.append("MLX rendered-mode parity MISMATCH — row invalid (D2)")
        elif row.parity == "template":
            notes.append(f"{name}: MLX template-mode fallback (Phase 0 feeding, {row.prompt_token_count} tokens) — delta named in limitations")
        elif row.parity is None:
            reasons.append("MLX export carries no parity line")
    if row.engine == "llama" and row.llama_count_matches is not True:
        reasons.append("llama.cpp prompt token count DIFFERS from the count on record (or missing)")
    if row.cold_warm is None and row.mode == "burst":
        reasons.append("cold/warm not recorded")
    return reasons, notes


# --------------------------------------------------------------------------
# Speed session
# --------------------------------------------------------------------------

def analyze_speed(root: Path, manifest: dict, refusals: list[str], notes: list[str]):
    session = manifest.get("sessions", {}).get("speed")
    speed = {e: EngineSpeed(engine=e) for e in ENGINES}
    bandwidth = {"triad": BandwidthStat("triad"), "read-only": BandwidthStat("read-only"), "pairs": {}}
    bookends: dict[str, float] = {}
    if session is None:
        refusals.append("manifest has no speed session — no headline speed rows")
        return speed, bandwidth, None
    for name, item, timeline in load_session_rows(root, session, refusals):
        if isinstance(item, ex.BandwidthExport):
            _add_bandwidth(bandwidth, name, item)
            continue
        reasons, row_notes = headline_eligibility(item, session, name)
        notes.extend(row_notes)
        speed[item.engine].notes.extend(row_notes)
        if reasons:
            refusals.extend(f"{name}: {r}" for r in reasons)
            continue
        _add_speed_row(speed[item.engine], name, item, timeline, refusals, bookends)
    for key, value in (session.get("footprint_mb") or {}).items():
        engine = key.split("-")[0]
        if engine in speed:
            speed[engine].footprint_mb[key] = float(value)
    _finish_bandwidth(bandwidth)
    delta = (bookends["close"] - bookends["open"]) if {"open", "close"} <= bookends.keys() else None
    return speed, bandwidth, delta


def _add_bandwidth(bandwidth: dict, name: str, bw: ex.BandwidthExport) -> None:
    bandwidth[bw.kernel].values.append(bw.sustained_gbps)
    bandwidth[bw.kernel].spreads.append(bw.spread)
    m = re.search(r"(\d+)$", name)
    if m:
        bandwidth["pairs"].setdefault(int(m.group(1)), {})[bw.kernel] = bw.sustained_gbps


def _finish_bandwidth(bandwidth: dict) -> None:
    ratios = [p["read-only"] / p["triad"] for _, p in sorted(bandwidth["pairs"].items())
              if {"triad", "read-only"} <= p.keys()]
    bandwidth["ratio_readonly_over_triad"] = ratios


def _add_speed_row(engine: EngineSpeed, name: str, row: ex.RowExport, timeline, refusals, bookends) -> None:
    if row.mode in ("sustained", "energy"):
        if row.prompt_name != DECODE_PROMPT:
            refusals.append(f"{name}: sustained loop ran {row.prompt_name!r}, not the pinned {DECODE_PROMPT}")
            return
        engine.sustained = sustained_from(name, row, timeline)
        return
    if row.mode != "burst":
        refusals.append(f"{name}: unrecognized mode {row.mode!r}")
        return
    is_cold = (row.cold_warm or "").lower().startswith("cold")
    if row.prompt_name == PREFILL_PROMPT:
        if row.prefill is None or row.warm_prefix:
            refusals.append(f"{name}: no prefill span of record on a prefill row")
            return
        stat = engine.prefill_cold if is_cold else engine.prefill_warm
        stat.values.append(row.prefill.tok_s)
        stat.files.append(name)
        stat.token_count = row.prefill.tokens
        return
    if row.prompt_name != DECODE_PROMPT:
        refusals.append(f"{name}: burst on {row.prompt_name!r} is neither the decode nor the prefill prompt")
        return
    if row.window_tok_s is None:
        refusals.append(f"{name}: decode row without a canonical-window rate (needs ≥512 generated tokens)")
        return
    stat = engine.decode_cold if is_cold else engine.decode_warm
    stat.values.append(row.window_tok_s)
    stat.files.append(name)
    stat.token_count = row.prompt_token_count
    if "bookend-open" in name:
        bookends["open"] = row.window_tok_s
    elif "bookend-close" in name:
        bookends["close"] = row.window_tok_s


def sustained_from(name: str, row: ex.RowExport, timeline: Optional[ex.Timeline]) -> Sustained:
    gens = timeline.generations if timeline and timeline.generations else row.generations
    windows = [g.window_tok_s for g in gens if g.window_tok_s is not None]
    half = windows[len(windows) // 2:] if windows else []
    minutes = row.sustained_minutes
    if minutes is None and row.cumulative:
        minutes = row.cumulative.cycle_wall_s / 60.0
    elif minutes is None and timeline and timeline.cycle:
        minutes = timeline.cycle.cycle_wall_s / 60.0
    return Sustained(
        file=name, generations=len(gens), first_window=windows[0] if windows else None,
        last_window=windows[-1] if windows else None,
        plateau_second_half_median=statistics.median(half) if half else None, minutes=minutes,
        basis=(timeline.elapsed_basis if timeline and timeline.elapsed_basis else "text gen lines"))


# --------------------------------------------------------------------------
# Energy round
# --------------------------------------------------------------------------

def analyze_energy(root: Path, manifest: dict, refusals: list[str]) -> tuple[dict, list]:
    sessions = manifest.get("sessions", {})
    session = sessions.get("energy")
    energy = {e: EngineEnergy(engine=e) for e in ENGINES}
    if session is None:
        for e in energy.values():
            e.refusals.append("manifest has no energy session")
        return energy, []
    session_refusals = _energy_session_refusals(session, sessions.get("speed"))
    idle = _idle_baseline(session, session_refusals)
    cycles: dict[str, list[EnergyCycle]] = {e: [] for e in ENGINES}
    rows = []
    for name, row, timeline in load_session_rows(root, session, refusals):
        if isinstance(row, ex.BandwidthExport):
            refusals.append(f"{name}: bandwidth export in the energy session ignored")
            continue
        if row.mode != "energy":
            refusals.append(f"{name}: not an energy-cycle export (mode {row.mode!r})")
            continue
        rows.append((name, row, timeline))
        reasons, _ = headline_eligibility(row, session, name)
        if reasons:
            refusals.extend(f"{name}: {r}" for r in reasons)
            continue
        if idle is None:
            continue
        try:
            cycles[row.engine].append(analyze_energy_cycle(row, idle, file=name))
        except Refusal as e:
            refusals.append(str(e))
    for e in ENGINES:
        energy[e] = summarize_engine_energy(e, cycles[e], session_refusals)
    return energy, rows


def _energy_session_refusals(session: dict, speed: Optional[dict]) -> list[str]:
    out = []
    if speed is not None and session.get("ios_build") != speed.get("ios_build"):
        out.append(f"energy round iOS build {session.get('ios_build')!r} ≠ speed session iOS build "
                   f"{speed.get('ios_build')!r} — not the same-build head-to-head")
    return out


def _idle_baseline(session: dict, session_refusals: list[str]) -> Optional[IdleBaseline]:
    raw = session.get("idle_baseline")
    if not raw:
        session_refusals.append("no idle baseline recorded for this iOS build — net energy undefined")
        return None
    idle = IdleBaseline(raw.get("health_percent"), raw.get("soc_start"), raw.get("soc_end"),
                        raw.get("wall_seconds"), raw.get("ios_build"))
    try:
        idle.validate()
    except Refusal as e:
        session_refusals.append(f"idle baseline: {e}")
        return None
    if idle.ios_build != session.get("ios_build"):
        session_refusals.append(f"idle baseline iOS build {idle.ios_build!r} ≠ energy round iOS build "
                                f"{session.get('ios_build')!r}")
        return None
    return idle


# --------------------------------------------------------------------------
# Verdict, tripwires, roofline
# --------------------------------------------------------------------------

def success_verdict(speed: dict) -> Verdict:
    ours, mlx = speed["ours"].decode_warm.median, speed["mlx"].decode_warm.median
    refusals = []
    if ours is None:
        refusals.append("no valid qwen-metal warm-burst window row")
    if mlx is None:
        refusals.append("no valid MLX warm-burst window row (same session, D1 definition)")
    if refusals:
        return Verdict(ours, mlx, None, None, refusals=refusals)
    ratio = ours / mlx
    return Verdict(ours, mlx, ratio, ratio >= SUCCESS_RATIO)


def tripwires(speed: dict) -> dict:
    decode = speed["ours"].decode_warm.median
    prefill = speed["ours"].prefill_warm.median
    return {
        "decode": Tripwire("decode warm-burst window median", TRIPWIRE_DECODE_TOK_S, decode,
                           None if decode is None else decode >= TRIPWIRE_DECODE_TOK_S),
        "prefill": Tripwire("prefill warm span median", TRIPWIRE_PREFILL_TOK_S, prefill,
                            None if prefill is None else prefill >= TRIPWIRE_PREFILL_TOK_S),
    }


def roofline(speed: dict, bandwidth: dict, manifest: dict) -> dict:
    denominator = bandwidth["triad"].median
    out = {}
    weight_bytes = {"ours": WEIGHT_BYTES, **{k: int(v) for k, v in (manifest.get("roofline", {}).get("weight_bytes") or {}).items()}}
    for engine, stat in speed.items():
        if engine not in weight_bytes:
            continue
        prompt_tokens = stat.decode_warm.token_count or DEFAULT_DECODE_PROMPT_TOKENS
        bytes_per_token = weight_bytes[engine] + KV_BYTES_PER_POSITION * (prompt_tokens + WINDOW_MEDIAN_DEPTH_OFFSET)
        window = stat.decode_warm.median
        implied = None if window is None else window * bytes_per_token / 1e9
        fraction = None if (implied is None or not denominator) else implied / denominator
        out[engine] = Roofline(engine, window, bytes_per_token, implied, denominator, fraction)
    return out


# --------------------------------------------------------------------------
# Figures (edge test 6): the D1 timeline, never the cumulative line
# --------------------------------------------------------------------------

def thermal_series(report: Report) -> list[ThermalSeries]:
    series = []
    for name, row, timeline in report.energy_rows:
        if timeline and timeline.generations:
            gens, source, basis = timeline.generations, "json", timeline.elapsed_basis or "loopClock"
        else:
            gens, source = row.generations, "text"
            basis = "loopClock" if all(g.elapsed_s is not None for g in gens) else "sumOfGenerationWall"
        elapsed, windows, running = [], [], 0.0
        for g in gens:
            running += g.wall_s
            if g.window_tok_s is None:
                continue
            elapsed.append(g.elapsed_s if g.elapsed_s is not None else running)
            windows.append(g.window_tok_s)
        series.append(ThermalSeries(row.engine, name, elapsed, windows, source, basis))
    _mark_median_cycles(series)
    return series


def _mark_median_cycles(series: list[ThermalSeries]) -> None:
    for engine in ENGINES:
        cycles = [s for s in series if s.engine == engine and s.window_tok_s]
        if not cycles:
            continue
        keyed = sorted(cycles, key=lambda s: statistics.median(s.window_tok_s))
        keyed[len(keyed) // 2].bold = True


def render_thermal_chart(series: list[ThermalSeries], out: Path) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    for s in series:
        if not s.window_tok_s:
            continue
        ax.plot([e / 60 for e in s.elapsed_s], s.window_tok_s, color=ENGINE_COLORS[s.engine],
                linewidth=2.0 if s.bold else 1.2, alpha=1.0 if s.bold else 0.35,
                label=ENGINE_LABELS[s.engine] if s.bold else None)
        if s.bold:
            ax.annotate(ENGINE_LABELS[s.engine], (s.elapsed_s[-1] / 60, s.window_tok_s[-1]),
                        textcoords="offset points", xytext=(6, 0), fontsize=9, color="#52514e", va="center")
    ax.set_xlabel("elapsed in the energy cycle (min)")
    ax.set_ylabel("decode tok/s, canonical window (128–512)")
    ax.set_title("Sustained-thermal: per-generation window rate, all cycles (median cycle bold)")
    _style_axes(ax)
    if any(s.bold for s in series):
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def energy_bar_data(energy: dict) -> list[EnergyBar]:
    bars = []
    for engine in ENGINES:
        h = energy[engine].headline
        if h is None:
            continue
        bars.append(EnergyBar(engine, h.mean_j_per_token, (h.spread[1] - h.spread[0]) / 2,
                              h.mean_j_per_token * h.systematic_fraction))
    return bars


def render_energy_bars(energy: dict, out: Path) -> Path:
    plt = _pyplot()
    bars = energy_bar_data(energy)
    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=150)
    xs = list(range(len(bars)))
    for x, b in zip(xs, bars):
        ax.bar(x, b.mean, width=0.55, color=ENGINE_COLORS[b.engine], linewidth=0)
        ax.errorbar(x - 0.08, b.mean, yerr=b.statistical_halfwidth, fmt="none", ecolor="#0b0b0b", capsize=4,
                    linewidth=1.2, label="statistical (min–max over n cycles)" if x == 0 else None)
        ax.errorbar(x + 0.08, b.mean, yerr=b.systematic_halfwidth, fmt="none", ecolor="#52514e", capsize=4,
                    linewidth=1.2, linestyle="--", label="systematic (SoC-reading quantization)" if x == 0 else None)
        ax.annotate(f"{b.mean:.3f} J", (x - 0.28, b.mean), textcoords="offset points", xytext=(0, 6),
                    ha="left", fontsize=9)
    ax.set_xticks(xs, [ENGINE_LABELS[b.engine] for b in bars])
    ax.set_ylabel("net J per token")
    ax.set_title("Energy per token — sustained, idle-subtracted")
    if bars:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    refused = [ENGINE_LABELS[e] for e in ENGINES if energy[e].headline is None]
    if refused:
        fig.text(0.01, 0.01, "no headline (below the ≥3 valid-cycle pin or refused): " + ", ".join(refused),
                 fontsize=8, color="#52514e", ha="left", va="bottom")
    _style_axes(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return out


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor("#fcfcfb")


# --------------------------------------------------------------------------
# Whole archive
# --------------------------------------------------------------------------

def analyze_archive(root: Path) -> Report:
    root = Path(root)
    manifest = load_manifest(root)
    refusals: list[str] = []
    notes: list[str] = []
    speed, bandwidth, bookend_delta = analyze_speed(root, manifest, refusals, notes)
    energy, energy_rows = analyze_energy(root, manifest, refusals)
    return Report(
        manifest=manifest, speed=speed, energy=energy, bandwidth=bandwidth,
        roofline=roofline(speed, bandwidth, manifest), verdict=success_verdict(speed),
        tripwires=tripwires(speed), bookend_delta=bookend_delta, refusals=refusals, notes=notes,
        energy_rows=energy_rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def summary_dict(r: Report) -> dict:
    return {
        "sessions": r.manifest.get("sessions", {}),
        "speed": {e: {**_jsonable(s), "decode_warm_median": s.decode_warm.median, "prefill_warm_median": s.prefill_warm.median}
                  for e, s in r.speed.items()},
        "energy": {e: _jsonable(en) for e, en in r.energy.items()},
        "bandwidth": {k: (_jsonable(v) if not isinstance(v, (list, dict)) else v) for k, v in r.bandwidth.items()},
        "roofline": _jsonable(r.roofline), "verdict": _jsonable(r.verdict), "tripwires": _jsonable(r.tripwires),
        "bookend_delta": r.bookend_delta, "refusals": r.refusals, "notes": r.notes,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", help="benchmarks/phase6 (contains manifest.json + session dirs)")
    parser.add_argument("--out", required=True, help="output directory for tables.md, summary.json, PNGs")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = analyze_archive(Path(args.archive))
    except Refusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from phase6_tables import render_tables  # sibling module (kept separate for the 800-line rule)
    (out / "tables.md").write_text(render_tables(report))
    (out / "summary.json").write_text(json.dumps(summary_dict(report), indent=2, ensure_ascii=False) + "\n")
    if not args.no_figures:
        render_thermal_chart(thermal_series(report), out / "thermal.png")
        render_energy_bars(report.energy, out / "energy.png")
    print(f"wrote {out / 'tables.md'} ({len(report.refusals)} refusal(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
