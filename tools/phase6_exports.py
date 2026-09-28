"""Parsers for the Phase 6 export formats (docs/phases/phase-6.md D8; task P6-3).

Three row-export formats and two bandwidth-export headers, all archived under
benchmarks/phase6/ (see benchmarks/phase6/README.md):

* ours — `BenchmarkReport.exportText()` (text of record) and `TimelineExport`
  JSON (schema `qwen-metal-timeline/1`);
* MLX LLMEval and llama.swiftui — the P6-2 harness exports (same vocabulary:
  `prefill span:`, `decode rate:`, `gen N:`, `cumulative:`, the battery lines)
  plus JSON schema `qwen-metal-timeline/1-external`;
* the two bandwidth microbench exports (`STREAM triad (P0B-4)` /
  `read-only streaming reduce (BW-1)`).

Parsing is line-anchored on the producing code's format strings. Anything
this module cannot read stays None; the analyzer (phase6_analyze.py) decides
what a None refuses. Two inputs are refused HERE because the producing code
says they are never rows: a bandwidth export stamped DIAGNOSTIC, and one
whose correctness gate did not pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional


class ParseError(ValueError):
    """The text is not one of the Phase 6 export formats."""


class Refusal(Exception):
    """A headline value is refused; the message names why."""


ENGINE_NAMES = {"mlx-llmeval": "mlx", "llama-swiftui": "llama"}
PLACEHOLDER = "(record manually)"


@dataclass(frozen=True)
class Prefill:
    tokens: int
    seconds: float
    tok_s: float


@dataclass(frozen=True)
class Generation:
    index: int
    tokens: int
    wall_s: float
    elapsed_s: Optional[float]
    overall_tok_s: Optional[float]
    window_tok_s: Optional[float]
    stop_reason: str
    truncated: bool


@dataclass(frozen=True)
class Cumulative:
    tokens: int
    sum_wall_s: float
    cycle_wall_s: float


@dataclass
class RowExport:
    engine: str                       # ours | mlx | llama
    provisional: bool
    round: Optional[str]
    mode: Optional[str] = None        # burst | sustained | energy (external energy-loop → energy)
    date: Optional[str] = None
    device: Optional[str] = None
    os_version: Optional[str] = None  # ours only (the external exports carry no device line)
    engine_line: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt_token_count: Optional[int] = None
    cold_warm: Optional[str] = None
    battery_health_percent: Optional[float] = None
    soc_start: Optional[float] = None
    soc_end: Optional[float] = None
    programmatic_soc_start: Optional[float] = None
    programmatic_soc_end: Optional[float] = None
    generated_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    prefill: Optional[Prefill] = None
    warm_prefix: bool = False
    window_tok_s: Optional[float] = None
    overall_tok_s: Optional[float] = None
    generations: list[Generation] = field(default_factory=list)
    cumulative: Optional[Cumulative] = None
    ended_by: Optional[str] = None
    sustained_minutes: Optional[float] = None
    parity: Optional[str] = None      # MLX: MATCH | MISMATCH | template
    llama_count_matches: Optional[bool] = None
    phys_footprint_mb: Optional[float] = None
    validation_line: Optional[str] = None


@dataclass
class Timeline:
    schema: str
    engine: str
    round: Optional[str]
    mode: Optional[str]
    elapsed_basis: Optional[str]
    generations: list[Generation]
    cycle: Optional[Cumulative]
    os_version: Optional[str] = None
    provisional: Optional[bool] = None


@dataclass(frozen=True)
class BandwidthExport:
    kernel: str                       # triad | read-only
    sustained_gbps: float
    spread: tuple[float, float]
    gate_ok: bool
    date: Optional[str]
    device: Optional[str]


# --------------------------------------------------------------------------
# Row exports (text)
# --------------------------------------------------------------------------

_NUM = r"(\d+(?:\.\d+)?)"
_HEADER = re.compile(
    r"^qwen-metal Phase \d+ row export \((?:(PROVISIONAL)|round: (.+?))\)"
    r"(?: — engine: (mlx-llmeval|llama-swiftui))?\s*$")
_DEVICE = re.compile(r"^device: (.+?) \(iOS (.+)\)\s*$")
_HEALTH = re.compile(rf"^battery health: (?:{_NUM}%|\(record manually\))")
_PCT_OR_PLACEHOLDER = rf"(?:{_NUM}%|\(record manually\))"
_SOC_OPERATOR = re.compile(
    rf"^state of charge \(Settings.*\): start {_PCT_OR_PLACEHOLDER}, end {_PCT_OR_PLACEHOLDER}\s*$")
_SOC_PROGRAMMATIC = re.compile(
    rf"^state of charge \(programmatic.*\): start (?:{_NUM}%|n/a[^,]*), end (?:{_NUM}%|n/a.*)$")
_PROMPT_OURS = re.compile(r"^prompt: (\S+) \((\d+) prompt tokens\)\s*$")
_PROMPT_MLX = re.compile(r"^prompt: (\S+) \((\d+) tokens; feeding: (RENDERED|raw text)")
_PROMPT_LLAMA = re.compile(r"^prompt: (\S+) \((\d+) tokens, llama\.cpp tokenizer")
_MODE_OURS = re.compile(r"^mode: (burst|sustained|energy) \| cold/warm: (.+?)\s*$")
_MODE_EXTERNAL = re.compile(r"^mode: (burst|energy-loop)\s*$")
_COLD_WARM = re.compile(r"^cold/warm: (.+?)\s*$")
_GENERATED = re.compile(rf"^generated: (\d+) tokens in {_NUM} s \(stop: (\w+)\)")
_PREFILL = re.compile(rf"^prefill span: (\d+) tokens in {_NUM} s = {_NUM} tok/s \(of record")
_DECODE_RATE = re.compile(
    rf"^decode rate: overall (?:{_NUM} tok/s|n/a), canonical window \(tokens 128-512\) (?:{_NUM} tok/s|n/a)")
_SUSTAINED = re.compile(rf"^sustained loop: (\d+) generations over {_NUM} min")
_ENERGY_CYCLE = re.compile(r"^energy cycle: (\d+) generations — ended by (.+?)(?: \(final generation truncated.*\))?\s*$")
_CUMULATIVE = re.compile(rf"^\s*cumulative: (\d+) generated tokens; Σ generation wall {_NUM} s; cycle wall {_NUM} s")
_GEN = re.compile(
    rf"^\s*gen (\d+): overall (?:{_NUM} tok/s|n/a)(?:, window {_NUM} tok/s)? — (\d+) tokens in {_NUM} s "
    rf"\(stop: (\w+)\)(?: \[elapsed {_NUM} s\])?( \[truncated by operator stop\])?\s*$")
_FOOTPRINT_OURS = re.compile(rf"^phys_footprint \(in-app cross-check.*\): {_NUM} MB\s*$")
_FOOTPRINT_EXTERNAL = re.compile(rf"^memory \(cross-check.*\): task_info phys_footprint {_NUM} MB")


def _float(s: Optional[str]) -> Optional[float]:
    return None if s is None else float(s)


def parse_export_text(text: str) -> RowExport:
    """Parse one row export (ours, MLX, or llama.cpp) into a RowExport."""
    lines = text.splitlines()
    if not lines:
        raise ParseError("empty export")
    header = _HEADER.match(lines[0])
    if header is None:
        raise ParseError(f"not a Phase 6 row export header: {lines[0]!r}")
    provisional_word, round_marker, external_engine = header.groups()
    row = RowExport(
        engine=ENGINE_NAMES.get(external_engine or "", "ours"),
        provisional=provisional_word is not None,
        round=round_marker.strip() if round_marker else None)
    in_last_generation = False
    for line in lines[1:]:
        if line.strip() == "last generation:":
            in_last_generation = True
            continue
        if in_last_generation:
            continue  # the external loop export repeats the last generation's burst block
        _parse_row_line(row, line)
    return row


def _parse_row_line(row: RowExport, line: str) -> None:
    """Dispatch one export line onto the RowExport fields (no-op when unknown)."""
    if _parse_header_field(row, line) or _parse_battery_line(row, line):
        return
    if _parse_prompt_line(row, line) or _parse_mode_line(row, line):
        return
    _parse_metric_line(row, line)


def _parse_header_field(row: RowExport, line: str) -> bool:
    if line.startswith("date: "):
        row.date = line[len("date: "):].strip()
        return True
    m = _DEVICE.match(line)
    if m:
        row.device, row.os_version = m.group(1), m.group(2).strip()
        return True
    if line.startswith("engine: qwen-metal"):
        row.engine_line = line[len("engine: "):]
        return True
    if line.startswith("Metal API validation:"):
        row.validation_line = line
        return True
    return False


def _parse_battery_line(row: RowExport, line: str) -> bool:
    m = _HEALTH.match(line)
    if m:
        row.battery_health_percent = _float(m.group(1))
        return True
    m = _SOC_OPERATOR.match(line)
    if m:
        row.soc_start, row.soc_end = _float(m.group(1)), _float(m.group(2))
        return True
    m = _SOC_PROGRAMMATIC.match(line)
    if m:
        row.programmatic_soc_start = _float(m.group(1))
        row.programmatic_soc_end = _float(m.group(2))
        return True
    return False


def _parse_prompt_line(row: RowExport, line: str) -> bool:
    m = _PROMPT_OURS.match(line)
    if m:
        row.prompt_name, row.prompt_token_count = m.group(1), int(m.group(2))
        return True
    m = _PROMPT_MLX.match(line)
    if m:
        row.prompt_name, row.prompt_token_count = m.group(1), int(m.group(2))
        if m.group(3) == "raw text":
            row.parity = "template"
        elif "MISMATCH" in line:
            row.parity = "MISMATCH"
        elif "MATCH" in line:
            row.parity = "MATCH"
        return True
    m = _PROMPT_LLAMA.match(line)
    if m:
        row.prompt_name, row.prompt_token_count = m.group(1), int(m.group(2))
        hf = re.search(r"HF/qwen-metal count (\d+)", line)
        if "matches the llama.cpp count on record" in line:
            row.llama_count_matches = True
        elif hf and int(hf.group(1)) == row.prompt_token_count:
            # Equal to the HF count is the strongest parity outcome (P6-4 finding:
            # the harness's "92 on record" was the Phase 0 Mac CLI figure; the
            # Phase 0 device rows read 84).
            row.llama_count_matches = True
        elif "DIFFERS from the llama.cpp count on record" in line:
            row.llama_count_matches = False
        return True
    return False


def _parse_mode_line(row: RowExport, line: str) -> bool:
    m = _MODE_OURS.match(line)
    if m:
        row.mode, row.cold_warm = m.group(1), _note(m.group(2))
        return True
    m = _MODE_EXTERNAL.match(line)
    if m:
        row.mode = "energy" if m.group(1) == "energy-loop" else m.group(1)
        return True
    m = _COLD_WARM.match(line)
    if m:
        row.cold_warm = _note(m.group(1))
        return True
    return False


def _parse_metric_line(row: RowExport, line: str) -> None:
    m = _GEN.match(line)
    if m:
        row.generations.append(Generation(
            index=int(m.group(1)), overall_tok_s=_float(m.group(2)), window_tok_s=_float(m.group(3)),
            tokens=int(m.group(4)), wall_s=float(m.group(5)), stop_reason=m.group(6),
            elapsed_s=_float(m.group(7)), truncated=m.group(8) is not None))
        return
    m = _GENERATED.match(line)
    if m and row.generated_tokens is None:
        row.generated_tokens, row.stop_reason = int(m.group(1)), m.group(3)
        return
    m = _PREFILL.match(line)
    if m and row.prefill is None:
        row.prefill = Prefill(int(m.group(1)), float(m.group(2)), float(m.group(3)))
        row.warm_prefix = "[WARM PREFIX" in line
        return
    m = _DECODE_RATE.match(line)
    if m and row.window_tok_s is None and row.overall_tok_s is None:
        row.overall_tok_s, row.window_tok_s = _float(m.group(1)), _float(m.group(2))
        return
    m = _CUMULATIVE.match(line)
    if m:
        row.cumulative = Cumulative(int(m.group(1)), float(m.group(2)), float(m.group(3)))
        return
    m = _ENERGY_CYCLE.match(line)
    if m:
        row.ended_by = m.group(2).strip()
        return
    m = _SUSTAINED.match(line)
    if m:
        row.sustained_minutes = float(m.group(2))
        return
    m = _FOOTPRINT_OURS.match(line) or _FOOTPRINT_EXTERNAL.match(line)
    if m:
        row.phys_footprint_mb = float(m.group(1))


def _note(value: str) -> Optional[str]:
    value = value.strip()
    return None if not value or value == PLACEHOLDER else value


# --------------------------------------------------------------------------
# Timeline JSON (both schemas)
# --------------------------------------------------------------------------

def parse_timeline_json(text: str) -> Timeline:
    """Parse a timeline JSON export (ours `/1` or external `/1-external`)."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(f"timeline JSON does not parse: {e}") from e
    schema = doc.get("schema")
    if schema == "qwen-metal-timeline/1":
        entries, cycle, engine = doc.get("generations", []), doc.get("cycle"), "ours"
    elif schema == "qwen-metal-timeline/1-external":
        entries, cycle = doc.get("entries", []), doc.get("cycleTotals")
        engine = ENGINE_NAMES.get(doc.get("engine", ""))
        if engine is None:
            raise ParseError(f"unknown external engine {doc.get('engine')!r}")
    else:
        raise ParseError(f"unknown timeline schema {schema!r}")
    mode = doc.get("mode")
    return Timeline(
        schema=schema, engine=engine, round=doc.get("round"),
        mode="energy" if mode == "energy-loop" else mode,
        elapsed_basis=doc.get("elapsedBasis"),
        generations=[_generation_from_json(e) for e in entries],
        cycle=_cumulative_from_json(cycle),
        os_version=doc.get("osVersion"), provisional=doc.get("provisional"))


def _generation_from_json(e: dict) -> Generation:
    return Generation(
        index=int(e["index"]), tokens=int(e["generatedTokens"]), wall_s=float(e["wallSeconds"]),
        elapsed_s=_float(e.get("elapsedAtEndSeconds")),
        overall_tok_s=_float(e.get("overallTokensPerSecond")),
        window_tok_s=_float(e.get("canonicalWindowTokensPerSecond")),
        stop_reason=str(e.get("stopReason")), truncated=bool(e.get("truncated", False)))


def _cumulative_from_json(cycle: Optional[dict]) -> Optional[Cumulative]:
    if not cycle:
        return None
    return Cumulative(int(cycle["totalGeneratedTokens"]), float(cycle["sumGenerationWallSeconds"]),
                      float(cycle["cycleWallSeconds"]))


# --------------------------------------------------------------------------
# Bandwidth microbench exports
# --------------------------------------------------------------------------

_BW_HEADERS = {
    "qwen-metal bandwidth microbench — STREAM triad (P0B-4)": "triad",
    "qwen-metal bandwidth microbench — read-only streaming reduce (BW-1)": "read-only",
}
_BW_SUSTAINED = re.compile(rf"^sustained \(median\): {_NUM} GB/s")
_BW_SPREAD = re.compile(rf"^spread: min {_NUM} / max {_NUM} GB/s")
_BW_DEVICE = re.compile(r"^device: (.+?) \((.+)\)\s*$")


def is_bandwidth_export(text: str) -> bool:
    return text.splitlines()[0].strip() in _BW_HEADERS if text.strip() else False


def parse_bandwidth_export(text: str) -> BandwidthExport:
    """Parse a bandwidth export; refuses DIAGNOSTIC geometry and a failed gate."""
    lines = text.splitlines()
    kernel = _BW_HEADERS.get(lines[0].strip() if lines else "")
    if kernel is None:
        raise ParseError("not a bandwidth microbench export")
    if any(line.startswith("DIAGNOSTIC — non-pinned geometry") for line in lines):
        raise Refusal(f"{kernel} bandwidth export is stamped DIAGNOSTIC (non-pinned geometry) — never a row")
    gate_ok = any(line.startswith("correctness:") and "within pre-committed gate" in line for line in lines)
    if not gate_ok:
        raise Refusal(f"{kernel} bandwidth export carries no passing correctness gate line — figure withheld")
    sustained = spread = date = device = None
    for line in lines:
        m = _BW_SUSTAINED.match(line)
        if m:
            sustained = float(m.group(1))
        m = _BW_SPREAD.match(line)
        if m:
            spread = (float(m.group(1)), float(m.group(2)))
        if line.startswith("date: "):
            date = line[6:].strip()
        m = _BW_DEVICE.match(line)
        if m:
            device = m.group(1)
    if sustained is None or spread is None:
        raise ParseError(f"{kernel} bandwidth export lacks the sustained/spread lines")
    return BandwidthExport(kernel, sustained, spread, True, date, device)
