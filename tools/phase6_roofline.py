"""Decode roofline inputs for the Phase 6 analyzer (tools/phase6_analyze.py; P6-3 → P6-6).

Writeup §5 (docs/phases/phase-6.md D8): decode tok/s against a MEASURED
bandwidth denominator divided by bytes-per-token. Two denominators are
carried side by side and both are measurements on record — the 43.84 GB/s
triad of record (DECISIONS.md 2026-08-22, PLAN.md invariant 1) and the
same-session triad median from the speed session's bandwidth pair — with the
session read-only median listed as the BW-1 bound. Nothing here is tuned:
every constant is a recorded measurement or a model-config byte count.

Basis labels: our own bytes/token is measured (the packed q4g64 artifact).
The external engines' per-token byte counts are NOT measured by this project;
unless the manifest supplies `roofline.weight_bytes.<engine>`, their implied
bandwidth is computed on the qwen-metal bytes/token basis and LABELED as such
in every table (never presented as a measured figure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- Measured / pinned inputs, consumed verbatim ------------------------------
ROOFLINE_GBPS_OF_RECORD = 43.84               # iPhone 15 Pro triad, DECISIONS.md 2026-08-22 (invariant 1)
WEIGHT_BYTES = 967_753_728                    # packed q4g64 weights streamed per token (P4-EXEC)
KV_BYTES_PER_POSITION = 28 * 2 * 8 * 128 * 2  # 28 layers × K,V × 8 kv heads × head_dim 128 × fp16
WINDOW_MEDIAN_DEPTH_OFFSET = (128 + 512) // 2  # generated-token depth at the window's median
DEFAULT_DECODE_PROMPT_TOKENS = 84             # decode-essay, HF count (used only when no decode row parsed)

BASIS_MEASURED = "measured (qwen-metal packed weights + fp16 KV)"
BASIS_MANIFEST = "manifest weight bytes + fp16 KV"
BASIS_OURS_PROXY = "qwen-metal bytes/token basis (this engine's per-token bytes are not measured — labeled)"


@dataclass
class Roofline:
    engine: str
    window_tok_s: Optional[float]
    bytes_per_token: int
    basis: str
    implied_gbps: Optional[float]
    denominator_gbps: Optional[float]          # the session triad median (None without a bandwidth pair)
    fraction_of_triad: Optional[float]         # implied ÷ session triad
    fraction_of_record: Optional[float]        # implied ÷ 43.84 (the denominator of record)
    ceiling_tok_s_record: Optional[float]      # 43.84 GB/s ÷ bytes/token
    ceiling_tok_s_session: Optional[float]     # session triad ÷ bytes/token


def denominators(bandwidth: dict) -> dict:
    """Every denominator the writeup cites, in one place (all measured)."""
    triad = bandwidth["triad"].median if "triad" in bandwidth else None
    readonly = bandwidth["read-only"].median if "read-only" in bandwidth else None
    return {
        "triad_of_record_gbps": ROOFLINE_GBPS_OF_RECORD,
        "triad_session_gbps": triad,
        "readonly_session_gbps": readonly,
        "weight_bytes": WEIGHT_BYTES,
        "kv_bytes_per_position": KV_BYTES_PER_POSITION,
        "window_median_depth_offset": WINDOW_MEDIAN_DEPTH_OFFSET,
        # weights re-streamed once per token at 100% of the record denominator (the Phase 5 structural ceiling)
        "sequential_prefill_ceiling_tok_s": ROOFLINE_GBPS_OF_RECORD * 1e9 / WEIGHT_BYTES,
    }


def roofline(speed: dict, bandwidth: dict, manifest: dict) -> dict:
    """One Roofline per engine. `speed[engine].decode_warm` supplies the window
    median and the prompt token count (the KV depth at the window's median)."""
    session = bandwidth["triad"].median if "triad" in bandwidth else None
    manifest_bytes = {k: int(v) for k, v in (manifest.get("roofline", {}).get("weight_bytes") or {}).items()}
    out = {}
    for engine, stat in speed.items():
        if engine == "ours":
            weight_bytes, basis = WEIGHT_BYTES, BASIS_MEASURED
        elif engine in manifest_bytes:
            weight_bytes, basis = manifest_bytes[engine], BASIS_MANIFEST
        else:
            weight_bytes, basis = WEIGHT_BYTES, BASIS_OURS_PROXY
        prompt_tokens = stat.decode_warm.token_count or DEFAULT_DECODE_PROMPT_TOKENS
        bytes_per_token = weight_bytes + KV_BYTES_PER_POSITION * (prompt_tokens + WINDOW_MEDIAN_DEPTH_OFFSET)
        window = stat.decode_warm.median
        implied = None if window is None else window * bytes_per_token / 1e9
        out[engine] = Roofline(
            engine=engine, window_tok_s=window, bytes_per_token=bytes_per_token, basis=basis,
            implied_gbps=implied, denominator_gbps=session,
            fraction_of_triad=None if (implied is None or not session) else implied / session,
            fraction_of_record=None if implied is None else implied / ROOFLINE_GBPS_OF_RECORD,
            ceiling_tok_s_record=ROOFLINE_GBPS_OF_RECORD * 1e9 / bytes_per_token,
            ceiling_tok_s_session=None if not session else session * 1e9 / bytes_per_token)
    return out
