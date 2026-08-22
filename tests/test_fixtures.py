"""Integrity tests for the committed oracle fixture set (P1-1).

Validates tests/fixtures/qwen3-1.7b/ against its manifest and against the
phase-0-1.md spec requirements. Stdlib-only so it runs under the repo-root
.venv alongside test_priorities.py:

    .venv/bin/python -m pytest tests/test_fixtures.py -q

These tests do NOT re-run the model — full regeneration forensics is
tools/dump_reference.py (see tools/README.md).
"""

import hashlib
import json
import struct
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "qwen3-1.7b"

# Independent copies of the pins (deliberately NOT imported from tools/pins.py
# so a pin edit there cannot silently re-validate stale fixtures).
EXPECTED_REPO = "Qwen/Qwen3-1.7B"
EXPECTED_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PROMPT_IDS = (
    "short_english",
    "multi_sentence",
    "code_snippet",
    "non_ascii",
    "chat_template",
)
GENERATION_STEPS = 50
CHECKPOINT_STEPS = (0, 1, 24, 49)
TOP_K = 64
VOCAB_SIZE = 151936
HIDDEN_SIZE = 2048
ACTIVATION_SLICES = (
    "embeddings_output",
    "layer0_pre_attn_norm_output",
    "layer0_attn_output",
    "layer0_block_output",
    "last_layer_output",
    "final_norm_output",
)
DTYPE_SIZE = {"float32": 4, "int32": 4}
SIZE_BUDGET_BYTES = 15_000_000  # spec: ~13 MB, plain git

STEP_KEYS = {"step", "logsumexp", "mean", "std", "top1_id", "top2_id", "margin_top1_top2"}


def load_manifest():
    return json.loads((FIXTURE_ROOT / "manifest.json").read_text())


def test_manifest_pins_and_versions():
    manifest = load_manifest()
    assert manifest["model"]["repo"] == EXPECTED_REPO
    assert manifest["model"]["revision"] == EXPECTED_REVISION

    pinned = {}
    for line in (REPO_ROOT / "tools" / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            pinned[name.replace("-", "_").lower()] = version
    for pkg, version in manifest["versions"].items():
        key = pkg.replace("-", "_").lower()
        assert key in pinned, f"{pkg} recorded in manifest but not pinned"
        assert version == pinned[key], f"{pkg}: manifest {version} != pinned {pinned[key]}"

    protocol = manifest["protocol"]
    assert protocol["generation_steps"] == GENERATION_STEPS
    assert tuple(protocol["checkpoint_steps"]) == CHECKPOINT_STEPS
    assert protocol["top_k"] == TOP_K
    assert protocol["enable_thinking"] is False


def test_every_manifest_blob_matches_on_disk():
    manifest = load_manifest()
    assert manifest["files"], "manifest has no file entries"
    total = 0
    for relpath, entry in manifest["files"].items():
        path = FIXTURE_ROOT / relpath
        assert path.is_file(), f"missing blob: {relpath}"
        data = path.read_bytes()
        assert len(data) == entry["byte_len"], f"byte_len mismatch: {relpath}"
        element_count = 1
        for dim in entry["shape"]:
            element_count *= dim
        assert element_count * DTYPE_SIZE[entry["dtype"]] == entry["byte_len"], (
            f"shape/dtype inconsistent with byte_len: {relpath}"
        )
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], (
            f"sha256 mismatch (fixture corrupted?): {relpath}"
        )
        total += entry["byte_len"]
    assert total <= SIZE_BUDGET_BYTES, f"fixture blobs exceed budget: {total} bytes"


def test_no_orphan_files_on_disk():
    manifest = load_manifest()
    known_json = {
        "manifest.json",
        "tokenizer_ids.json",
        "mlx_secondary.json",
    } | {f"prompts/{pid}/steps.json" for pid in PROMPT_IDS}
    on_disk = {
        str(p.relative_to(FIXTURE_ROOT))
        for p in FIXTURE_ROOT.rglob("*")
        if p.is_file() and not p.name.startswith(".")  # e.g. .DS_Store
    }
    orphans = on_disk - set(manifest["files"]) - known_json
    assert not orphans, f"files on disk not tracked by manifest: {sorted(orphans)}"


def test_per_prompt_contents_match_spec():
    manifest = load_manifest()
    files = manifest["files"]
    for pid in PROMPT_IDS:
        for step in CHECKPOINT_STEPS:
            key = f"prompts/{pid}/logits_step{step:04d}.bin"
            assert key in files, f"missing checkpoint: {key}"
            assert files[key]["dtype"] == "float32"
            assert files[key]["shape"] == [VOCAB_SIZE]
        values = files[f"prompts/{pid}/top64_values.bin"]
        indices = files[f"prompts/{pid}/top64_indices.bin"]
        assert values["dtype"] == "float32"
        assert values["shape"] == [GENERATION_STEPS, TOP_K]
        assert indices["dtype"] == "int32"
        assert indices["shape"] == [GENERATION_STEPS, TOP_K]


def test_steps_json_internally_consistent():
    for pid in PROMPT_IDS:
        payload = json.loads((FIXTURE_ROOT / "prompts" / pid / "steps.json").read_text())
        assert payload["prompt_id"] == pid
        argmax = payload["argmax_token_ids"]
        steps = payload["steps"]
        assert len(argmax) == GENERATION_STEPS
        assert len(steps) == GENERATION_STEPS
        indices_blob = (FIXTURE_ROOT / "prompts" / pid / "top64_indices.bin").read_bytes()
        top_ids = struct.unpack(f"<{GENERATION_STEPS * TOP_K}i", indices_blob)
        for i, record in enumerate(steps):
            assert STEP_KEYS <= set(record), f"{pid} step {i} missing keys"
            assert record["step"] == i
            assert record["margin_top1_top2"] >= 0.0
            assert 0 <= record["top1_id"] < VOCAB_SIZE
            # top-1 of the recorded top-64 slab is the argmax token
            assert record["top1_id"] == top_ids[i * TOP_K]
            assert record["top1_id"] == argmax[i]


def test_activation_slices_match_spec():
    manifest = load_manifest()
    files = manifest["files"]
    tokenizer_ids = json.loads((FIXTURE_ROOT / "tokenizer_ids.json").read_text())
    activation_prompt = manifest["protocol"]["activation_prompt"]
    seq_len = len(tokenizer_ids[activation_prompt]["input_ids"])
    for name in ACTIVATION_SLICES:
        key = f"activations/{name}.bin"
        assert key in files, f"missing activation slice: {key}"
        assert files[key]["dtype"] == "float32"
        assert files[key]["shape"] == [seq_len, HIDDEN_SIZE], (
            f"{name}: shape {files[key]['shape']} != [{seq_len}, {HIDDEN_SIZE}]"
        )


def test_tokenizer_ids_cover_all_prompts():
    tokenizer_ids = json.loads((FIXTURE_ROOT / "tokenizer_ids.json").read_text())
    assert set(tokenizer_ids) == set(PROMPT_IDS)
    for pid, entry in tokenizer_ids.items():
        assert entry["input_ids"], f"{pid}: empty input_ids"
        assert all(0 <= t < VOCAB_SIZE for t in entry["input_ids"])
        if entry["kind"] == "chat":
            assert entry["input_text"] != entry["raw_text"], (
                "chat prompt was not template-rendered"
            )
        else:
            assert entry["input_text"] == entry["raw_text"]


def test_mlx_secondary_dump_present_and_labeled():
    payload = json.loads((FIXTURE_ROOT / "mlx_secondary.json").read_text())
    assert "NOT an oracle" in payload["role"]
    assert set(payload["prompts"]) == set(PROMPT_IDS)
    for pid, entry in payload["prompts"].items():
        assert len(entry["generated_token_ids"]) == GENERATION_STEPS, pid
