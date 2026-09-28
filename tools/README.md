# tools/ — reference & fixture tooling (Python)

Pinned-dependency Python scripts that produce the Phase 1 correctness oracle
(docs/phases/phase-0-1.md). **Regenerating fixtures with unpinned versions is a
bug**; all pins live in `pins.py` (model repo/revision, generation protocol)
and `requirements.txt` (exact package versions).

## Setup (one time)

```sh
python3 -m venv tools/.venv
tools/.venv/bin/pip install -r tools/requirements.txt
```

## Regenerating the committed fixtures (tests/fixtures/qwen3-1.7b/)

Order matters — the mlx dump reuses the rendered chat-template text from the
primary dump so the template of record stays the pinned base revision's.

```sh
tools/.venv/bin/python tools/dump_reference.py   # primary oracle: HF transformers, CPU, fp32
tools/.venv/bin/python tools/dump_mlx.py         # secondary ecosystem check: mlx-lm 4-bit
```

The primary dump is byte-reproducible on the same machine/pins (verified at
P1-1; see DECISIONS.md). `tests/test_fixtures.py` (stdlib-only, run via the
repo root `.venv` pytest) validates the committed set against `manifest.json`
(sha256, shapes, spec-required contents, pins).

## One-time shard consolidation

The pinned checkpoint ships 2 safetensors shards; the engine parser is
single-file-only by design (PLAN.md). Produce the single-file artifact the
engine consumes (multi-GB, never committed — record sha256 in DECISIONS.md):

```sh
tools/.venv/bin/python tools/consolidate_shards.py --out <path>.safetensors
```

## Files

| File | Role |
|---|---|
| `pins.py` | single source of truth for model/protocol pins |
| `fixture_prompts.json` | the 5 pinned fixture prompts |
| `dump_reference.py` | primary oracle dump (full-vocab logit checkpoints, fingerprints, top-64, margins, argmax, per-module activations, tokenizer ids) |
| `dump_mlx.py` | secondary mlx-lm dump (loose, argmax-level; Phase 3 quality-gate comparator) |
| `dump_quality_gate.py` | P3-3 quality-gate dump: band-setters vs mlx-lm 4-bit, WikiText-2 ppl slice, local-only reference-logits artifact (see its module docstring for the full protocol) |
| `consolidate_shards.py` | shard → single-file safetensors consolidation |
| `requirements.txt` | exact pinned package versions (this venv) |
| `phase6_analyze.py` | P6-3: Phase 6 archive (`benchmarks/phase6/`) → writeup tables + figures; refuses headline values that violate the pre-committed validity criteria. Runs under the REPO-ROOT `.venv` |
| `phase6_exports.py` | parsers for the three row-export formats + the two bandwidth exports |
| `phase6_tables.py` | Markdown rendering of the analyzer's report |
| `requirements-phase6.txt` | pins for the analyzer (repo-root `.venv`: matplotlib, numpy, pytest) |

## Quality-gate fixtures (tests/fixtures/qwen3-1.7b-quality/)

Produced by `tools/.venv/bin/python tools/dump_quality_gate.py` (P3-3;
run after the primary fixtures exist). Aborts unless the HF fp32 pass
reproduces the committed checkpoint blobs byte-identically. Also writes the
local-only `models/qwen3-1.7b-70d244cc-ref-logits-250.bin` (152 MB, never
committed — sha256 in DECISIONS.md). Validated by
`tests/test_quality_fixtures.py`; consumed by the Swift
`QuantQualityGateTests` (run release-mode).

## Phase 6 analyzer (tests/test_phase6_analyze.py; runs under the repo-root .venv)

```sh
.venv/bin/pip install -r tools/requirements-phase6.txt
.venv/bin/python tools/phase6_analyze.py benchmarks/phase6 --out docs/writeup
.venv/bin/python -m pytest tests/test_phase6_analyze.py -q
```

Archive layout and the operator manifest: `benchmarks/phase6/README.md`.
Runbook: `benchmarks/phase6-runbook.md`.
