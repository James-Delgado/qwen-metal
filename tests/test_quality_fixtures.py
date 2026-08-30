"""Integrity tests for the committed P3-3 quality-gate fixture set.

Validates tests/fixtures/qwen3-1.7b-quality/ (band-setters + ppl slice,
produced by tools/dump_quality_gate.py) the way test_fixtures.py validates
the frozen P1-1 set. Stdlib-only, repo-root .venv:

    .venv/bin/python -m pytest tests/test_quality_fixtures.py -q

These tests do NOT re-run any model — regeneration forensics is
tools/dump_quality_gate.py (see tools/README.md). The gate FORMULAS are
binding in DECISIONS.md ("Phase 3 gates pre-committed", 2026-08-25) and are
applied by the Swift QuantQualityGateTests; this file only pins the
fixture set's integrity and internal consistency.
"""

import hashlib
import json
import math
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUALITY_ROOT = REPO_ROOT / "tests" / "fixtures" / "qwen3-1.7b-quality"

# Independent copies of the pins (deliberately NOT imported from tools/pins.py
# so a pin edit there cannot silently re-validate stale fixtures).
EXPECTED_MODEL_REPO = "Qwen/Qwen3-1.7B"
EXPECTED_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
EXPECTED_MLX_REPO = "mlx-community/Qwen3-1.7B-4bit"
EXPECTED_MLX_REVISION = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"
EXPECTED_WIKITEXT_REPO = "Salesforce/wikitext"
EXPECTED_WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
EXPECTED_WIKITEXT_CONFIG = "wikitext-2-raw-v1"
EXPECTED_WIKITEXT_SPLIT = "test"

SLICE_TOKENS = 4096
VOCAB_SIZE = 151936
GENERATION_STEPS = 50
PROMPT_ORDER = [
    "short_english",
    "multi_sentence",
    "code_snippet",
    "non_ascii",
    "chat_template",
]
STEPS_TOTAL = len(PROMPT_ORDER) * GENERATION_STEPS
DTYPE_SIZE = {"float32": 4, "int32": 4}
SIZE_BUDGET_BYTES = 200_000  # two small blobs + band.json


def load_band():
    return json.loads((QUALITY_ROOT / "band.json").read_text())


def test_band_pins():
    band = load_band()
    assert band["model"]["repo"] == EXPECTED_MODEL_REPO
    assert band["model"]["revision"] == EXPECTED_MODEL_REVISION
    assert band["mlx_model"]["repo"] == EXPECTED_MLX_REPO
    assert band["mlx_model"]["revision"] == EXPECTED_MLX_REVISION
    dataset = band["dataset"]
    assert dataset["repo"] == EXPECTED_WIKITEXT_REPO
    assert dataset["revision"] == EXPECTED_WIKITEXT_REVISION
    assert dataset["config"] == EXPECTED_WIKITEXT_CONFIG
    assert dataset["split"] == EXPECTED_WIKITEXT_SPLIT
    assert len(dataset["parquet_sha256"]) == 64


def test_band_versions_match_requirements():
    band = load_band()
    pinned = {}
    for line in (REPO_ROOT / "tools" / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            pinned[name.replace("-", "_").lower()] = version
    for pkg, version in band["versions"].items():
        key = pkg.replace("-", "_").lower()
        assert key in pinned, f"{pkg} recorded in band.json but not pinned"
        assert version == pinned[key], f"{pkg}: band {version} != pinned {pinned[key]}"


def test_blobs_match_index_and_no_orphans():
    band = load_band()
    total = 0
    for relpath, entry in band["files"].items():
        path = QUALITY_ROOT / relpath
        assert path.is_file(), f"missing blob: {relpath}"
        data = path.read_bytes()
        assert len(data) == entry["byte_len"], f"byte_len mismatch: {relpath}"
        element_count = 1
        for dim in entry["shape"]:
            element_count *= dim
        assert element_count * DTYPE_SIZE[entry["dtype"]] == entry["byte_len"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], (
            f"sha256 mismatch (fixture corrupted?): {relpath}"
        )
        total += entry["byte_len"]
    on_disk = {
        str(p.relative_to(QUALITY_ROOT))
        for p in QUALITY_ROOT.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    }
    orphans = on_disk - set(band["files"]) - {"band.json"}
    assert not orphans, f"files on disk not tracked by band.json: {sorted(orphans)}"
    total += (QUALITY_ROOT / "band.json").stat().st_size
    assert total <= SIZE_BUDGET_BYTES, f"quality fixtures exceed budget: {total} bytes"


def test_token_slice_valid():
    band = load_band()
    entry = band["files"]["ppl_slice_tokens.bin"]
    assert entry["dtype"] == "int32"
    assert entry["shape"] == [SLICE_TOKENS]
    blob = (QUALITY_ROOT / "ppl_slice_tokens.bin").read_bytes()
    ids = struct.unpack(f"<{SLICE_TOKENS}i", blob)
    assert all(0 <= t < VOCAB_SIZE for t in ids)
    assert len(set(ids)) > 100, "degenerate token slice"


def test_band_setters_internally_consistent():
    band = load_band()
    setters = band["band_setters"]
    assert setters["steps_total"] == STEPS_TOTAL
    assert isinstance(setters["A_mlx_count"], int)
    assert 0 <= setters["A_mlx_count"] <= STEPS_TOTAL
    assert math.isclose(setters["A_mlx"], setters["A_mlx_count"] / STEPS_TOTAL,
                        rel_tol=0, abs_tol=1e-12)
    assert math.isfinite(setters["KL_mlx_nats"]) and setters["KL_mlx_nats"] >= 0
    ppl_fp32 = band["reference"]["ppl_fp32"]
    assert math.isfinite(ppl_fp32) and ppl_fp32 > 1
    assert math.isfinite(setters["ppl_mlx"]) and setters["ppl_mlx"] > 1
    assert math.isclose(setters["dppl_mlx"], setters["ppl_mlx"] - ppl_fp32,
                        rel_tol=1e-9, abs_tol=1e-9)
    per_prompt = setters["per_prompt"]
    assert set(per_prompt) == set(PROMPT_ORDER)
    assert sum(p["agree"] for p in per_prompt.values()) == setters["A_mlx_count"]


def test_ref_nll_blob_consistent_with_ppl_fp32():
    band = load_band()
    entry = band["files"]["ppl_slice_ref_nll.bin"]
    assert entry["dtype"] == "float32"
    assert entry["shape"] == [SLICE_TOKENS - 1]
    blob = (QUALITY_ROOT / "ppl_slice_ref_nll.bin").read_bytes()
    nlls = struct.unpack(f"<{SLICE_TOKENS - 1}f", blob)
    assert all(math.isfinite(v) and v >= 0 for v in nlls)
    # fp32-rounded diagnostics must reproduce the float64 scalar of record
    # to fp32 rounding error.
    ppl_from_blob = math.exp(sum(nlls) / len(nlls))
    assert math.isclose(ppl_from_blob, band["reference"]["ppl_fp32"], rel_tol=1e-4)


def test_ref_logits_artifact_record():
    band = load_band()
    artifact = band["reference"]["ref_logits_artifact"]
    assert artifact["path"] == "models/qwen3-1.7b-70d244cc-ref-logits-250.bin"
    assert artifact["dtype"] == "float32"
    assert artifact["shape"] == [len(PROMPT_ORDER), GENERATION_STEPS, VOCAB_SIZE]
    assert artifact["byte_len"] == STEPS_TOTAL * VOCAB_SIZE * 4
    assert artifact["prompt_order"] == PROMPT_ORDER
    assert len(artifact["sha256"]) == 64
