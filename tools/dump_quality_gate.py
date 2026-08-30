"""P3-3 quality-gate dump: band-setters vs mlx-lm 4-bit + perplexity slice.

Implements the tools/ side of the Phase 3 quality gate (docs/phases/phase-3.md
D6; binding gate record: DECISIONS.md 2026-08-25 "Phase 3 gates pre-committed",
veto-closed 2026-08-26). Produces:

  1. LOCAL-ONLY reference-logits artifact (models/, never committed — sha256
     recorded in DECISIONS.md): HF transformers fp32 CPU logits, teacher-forced
     on the COMMITTED argmax sequences (5 prompts x 50 steps), full vocab,
     fp32 little-endian, shape [5, 50, 151936] in fixture_prompts.json order.
     Integrity: rows at the committed checkpoint steps {0,1,24,49} must be
     BYTE-IDENTICAL to tests/fixtures/qwen3-1.7b/prompts/*/logits_step*.bin,
     and the running argmax must reproduce steps.json — any mismatch aborts.
  2. COMMITTED quality fixtures (tests/fixtures/qwen3-1.7b-quality/ — a
     sibling of the frozen P1-1 set, which is never touched after Phase 1):
     ppl_slice_tokens.bin (int32 [4096]), ppl_slice_ref_nll.bin (fp32 [4095],
     diagnostics), band.json (band-setters + protocol + blob/artifact index).
  3. BAND-SETTER numbers for DECISIONS.md: A_mlx, KL_mlx, ppl_fp32, ppl_mlx,
     dppl_mlx. Per the gates entry these are recorded in DECISIONS.md BEFORE
     any metric of ours (CPU-quant) is computed.

Perplexity-slice protocol (the OV#12 named slice, documented here per spec):
  - Dataset: Salesforce/wikitext @ WIKITEXT_REVISION (pinned in pins.py),
    config wikitext-2-raw-v1, TEST split, single parquet file (sha256
    recorded in band.json).
  - Document concatenation: `"".join(rows)` of the parquet `text` column in
    stored row order — non-empty rows carry their trailing "\\n", empty rows
    are "" — i.e. the deterministic concatenation of the rows exactly as
    stored in the pinned file. No normalization, no shuffling.
  - Tokenization: pinned HF tokenizer (Qwen/Qwen3-1.7B @ MODEL_REVISION),
    add_special_tokens=False (raw corpus, no chat template); first 4096 ids.
  - ppl = exp(mean NLL) over positions 1..4095 (position i predicted from
    ids[0..i-1]); one full context window, no striding. NLL computed in
    float64 over fp32 logits (logsumexp - logit[target]).
  - HF reference forward uses KV-cached chunked prefill (chunk 512) — one
    causal window, mathematically the full-window forward. mlx uses the
    same chunking via its prompt cache; logits cast fp32 before float64 NLL.

Teacher-forced metric protocol (identical for band-setter and engine side):
  - Sequences: committed input_ids + committed reference argmax ids, i.e.
    identical prefixes for every engine at every one of the 250 steps.
  - Top-1 agreement: argmax(engine logits, first-index tie-break) equals the
    committed reference argmax id, counted over 250 steps.
  - KL: KL(P_fp32 || P_engine) in nats, float64 softmax over the full vocab,
    mean over 250 steps.

Run (order matters: after dump_reference.py fixtures exist; ~10-20 min, CPU):

    tools/.venv/bin/python tools/dump_quality_gate.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pins import (
    FIXTURE_DIRNAME,
    GENERATION_STEPS,
    CHECKPOINT_STEPS,
    MLX_REPO,
    MLX_REVISION,
    MODEL_REPO,
    MODEL_REVISION,
    PPL_SLICE_TOKENS,
    QUALITY_FIXTURE_DIRNAME,
    VOCAB_SIZE,
    WIKITEXT_CONFIG,
    WIKITEXT_REPO,
    WIKITEXT_REVISION,
    WIKITEXT_SPLIT,
    WIKITEXT_TEST_FILE,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / FIXTURE_DIRNAME
QUALITY_ROOT = REPO_ROOT / "tests" / "fixtures" / QUALITY_FIXTURE_DIRNAME
MODELS_DIR = REPO_ROOT / "models"
REF_LOGITS_NAME = "qwen3-1.7b-70d244cc-ref-logits-250.bin"

PPL_CHUNK = 512  # KV-cached prefill chunk (protocol constant, documented above)

PINNED_PACKAGES = (
    "torch", "transformers", "tokenizers", "numpy",
    "mlx", "mlx-lm", "pyarrow", "huggingface_hub",
)


# ---------------------------------------------------------------- float64 math

def logsumexp64(x64: np.ndarray) -> float:
    m = float(x64.max())
    return m + float(np.log(np.exp(x64 - m).sum()))


def nll64(logits32: np.ndarray, target: int) -> float:
    """-log softmax(logits)[target], float64 over the fp32 logit vector."""
    x = logits32.astype(np.float64)
    return logsumexp64(x) - float(x[target])


def kl64(ref32: np.ndarray, eng32: np.ndarray) -> float:
    """KL(P_ref || P_eng) in nats, float64 full-vocab softmax."""
    p = ref32.astype(np.float64)
    q = eng32.astype(np.float64)
    p_log = p - logsumexp64(p)
    q_log = q - logsumexp64(q)
    return float(np.sum(np.exp(p_log) * (p_log - q_log)))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------- fixture loading

def load_prompt_order() -> list:
    with open(Path(__file__).resolve().parent / "fixture_prompts.json") as f:
        return [p["id"] for p in json.load(f)["prompts"]]


def load_fixture(pid: str) -> dict:
    ids = json.loads((FIXTURE_ROOT / "tokenizer_ids.json").read_text())
    steps = json.loads((FIXTURE_ROOT / f"prompts/{pid}/steps.json").read_text())
    assert steps["prompt_id"] == pid
    assert len(steps["argmax_token_ids"]) == GENERATION_STEPS
    return {
        "input_ids": ids[pid]["input_ids"],
        "argmax_ids": steps["argmax_token_ids"],
    }


# --------------------------------------------------- phase A: HF fp32 reference

def hf_teacher_forced_logits(model, fixture: dict, pid: str) -> np.ndarray:
    """[50, vocab] fp32 logits teacher-forced on the committed argmax ids,
    with byte-identity verification against the committed checkpoint blobs
    and argmax verification against steps.json (abort on any mismatch)."""
    rows = np.zeros((GENERATION_STEPS, VOCAB_SIZE), dtype=np.float32)
    out = model(
        input_ids=torch.tensor([fixture["input_ids"]], dtype=torch.long),
        use_cache=True,
    )
    past = out.past_key_values
    logits = out.logits[0, -1]
    for step in range(GENERATION_STEPS):
        row = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        assert row.shape == (VOCAB_SIZE,)
        rows[step] = row

        if step in CHECKPOINT_STEPS:
            committed = (
                FIXTURE_ROOT / f"prompts/{pid}/logits_step{step:04d}.bin"
            ).read_bytes()
            if rows[step].astype("<f4", copy=False).tobytes() != committed:
                sys.exit(
                    f"ABORT: {pid} step {step}: teacher-forced fp32 logits are "
                    "not byte-identical to the committed checkpoint blob — "
                    "environment drift; do not record band-setters from this run"
                )
        ref_argmax = fixture["argmax_ids"][step]
        ours = int(np.argmax(rows[step]))
        if ours != ref_argmax:
            sys.exit(
                f"ABORT: {pid} step {step}: argmax {ours} != committed "
                f"{ref_argmax} — environment drift"
            )
        if step < GENERATION_STEPS - 1:
            out = model(
                input_ids=torch.tensor([[ref_argmax]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[0, -1]
    return rows


def hf_ppl_nll(model, token_ids: list) -> np.ndarray:
    """Per-position float64 NLL for positions 1..len-1 (KV-cached chunked
    prefill, one causal window)."""
    nlls = []
    past = None
    for start in range(0, len(token_ids), PPL_CHUNK):
        chunk = token_ids[start:start + PPL_CHUNK]
        out = model(
            input_ids=torch.tensor([chunk], dtype=torch.long),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logits = out.logits[0].detach().cpu().numpy().astype(np.float32, copy=False)
        for j in range(len(chunk)):
            target_pos = start + j + 1
            if target_pos < len(token_ids):
                nlls.append(nll64(logits[j], token_ids[target_pos]))
        print(f"    ppl positions {start}..{start + len(chunk) - 1} done", flush=True)
    return np.array(nlls, dtype=np.float64)


def build_ppl_slice(tokenizer):
    """First PPL_SLICE_TOKENS ids of the concatenated pinned test split."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    parquet_path = hf_hub_download(
        WIKITEXT_REPO, WIKITEXT_TEST_FILE,
        repo_type="dataset", revision=WIKITEXT_REVISION,
    )
    parquet_sha = sha256_bytes(Path(parquet_path).read_bytes())
    rows = pq.read_table(parquet_path).column("text").to_pylist()
    text = "".join(rows)  # protocol: rows exactly as stored (see module doc)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < PPL_SLICE_TOKENS:
        sys.exit(f"ABORT: test split tokenizes to only {len(ids)} tokens")
    return ids[:PPL_SLICE_TOKENS], parquet_sha, len(rows)


# ------------------------------------------------- phase B: mlx-lm band-setters

def mlx_teacher_forced_metrics(model, fixture: dict, ref_rows: np.ndarray) -> dict:
    """Agreement count + per-step KL for one prompt (full re-forward per step
    — sequences are short; mlx logits cast fp32 before float64 stats)."""
    import mlx.core as mx

    agree = 0
    kls = []
    prefix = list(fixture["input_ids"])
    for step in range(GENERATION_STEPS):
        logits = model(mx.array([prefix]))[0, -1]
        row = np.array(logits.astype(mx.float32), copy=False)
        assert row.shape == (VOCAB_SIZE,)
        if int(np.argmax(row)) == fixture["argmax_ids"][step]:
            agree += 1
        kls.append(kl64(ref_rows[step], row))
        prefix.append(fixture["argmax_ids"][step])
    return {"agree": agree, "kls": kls}


def mlx_ppl_nll(model, token_ids: list) -> np.ndarray:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    nlls = []
    for start in range(0, len(token_ids), PPL_CHUNK):
        chunk = token_ids[start:start + PPL_CHUNK]
        logits = model(mx.array([chunk]), cache=cache)[0]
        rows = np.array(logits.astype(mx.float32), copy=False)
        for j in range(len(chunk)):
            target_pos = start + j + 1
            if target_pos < len(token_ids):
                nlls.append(nll64(rows[j], token_ids[target_pos]))
    return np.array(nlls, dtype=np.float64)


# ------------------------------------------------------------------------ main

def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    prompt_order = load_prompt_order()
    fixtures = {pid: load_fixture(pid) for pid in prompt_order}

    # ---- Phase A: HF fp32 reference ----------------------------------------
    print(f"loading {MODEL_REPO} @ {MODEL_REVISION[:8]} (fp32, CPU)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION,
        dtype=torch.float32, attn_implementation="sdpa",
    )
    model.eval()

    ref = np.zeros((len(prompt_order), GENERATION_STEPS, VOCAB_SIZE), dtype=np.float32)
    for i, pid in enumerate(prompt_order):
        print(f"  teacher-forced fp32 logits: {pid}", flush=True)
        ref[i] = hf_teacher_forced_logits(model, fixtures[pid], pid)
    print("  checkpoint byte-identity + argmax verification: OK (all prompts)",
          flush=True)

    MODELS_DIR.mkdir(exist_ok=True)
    ref_bytes = ref.astype("<f4", copy=False).tobytes()
    ref_path = MODELS_DIR / REF_LOGITS_NAME
    ref_path.write_bytes(ref_bytes)
    ref_sha = sha256_bytes(ref_bytes)
    print(f"  wrote {ref_path} ({len(ref_bytes)/1e6:.1f} MB) sha256={ref_sha[:16]}...",
          flush=True)

    print("building ppl slice (pinned WikiText-2 test split)...", flush=True)
    slice_ids, parquet_sha, row_count = build_ppl_slice(tokenizer)
    print(f"  {row_count} rows -> first {len(slice_ids)} tokens", flush=True)
    print("  HF fp32 NLL over the window...", flush=True)
    ref_nll = hf_ppl_nll(model, slice_ids)
    assert ref_nll.shape == (PPL_SLICE_TOKENS - 1,)
    ppl_fp32 = float(np.exp(ref_nll.mean()))
    print(f"  ppl_fp32 = {ppl_fp32:.6f}", flush=True)

    del model  # free ~7 GB before loading mlx

    # ---- Phase B: mlx-lm 4-bit band-setters --------------------------------
    from huggingface_hub import snapshot_download
    from mlx_lm import load as mlx_load

    print(f"loading {MLX_REPO} @ {MLX_REVISION[:8]} (revision-pinned)...", flush=True)
    # Programmatic provenance check (gates entry): the pinned revision is
    # requested explicitly; snapshot_download fails loudly if it is absent.
    local = snapshot_download(MLX_REPO, revision=MLX_REVISION)
    mlx_model, _mlx_tok = mlx_load(local)

    agree_total = 0
    kl_all = []
    per_prompt = {}
    for i, pid in enumerate(prompt_order):
        m = mlx_teacher_forced_metrics(mlx_model, fixtures[pid], ref[i])
        agree_total += m["agree"]
        kl_all.extend(m["kls"])
        per_prompt[pid] = {
            "agree": m["agree"],
            "mean_kl_nats": float(np.mean(m["kls"])),
        }
        print(f"  {pid}: agree {m['agree']}/{GENERATION_STEPS}, "
              f"mean KL {per_prompt[pid]['mean_kl_nats']:.6f}", flush=True)

    steps_total = len(prompt_order) * GENERATION_STEPS
    a_mlx = agree_total / steps_total
    kl_mlx = float(np.mean(kl_all))

    print("  mlx NLL over the ppl window...", flush=True)
    mlx_nll = mlx_ppl_nll(mlx_model, slice_ids)
    assert mlx_nll.shape == (PPL_SLICE_TOKENS - 1,)
    ppl_mlx = float(np.exp(mlx_nll.mean()))
    dppl_mlx = ppl_mlx - ppl_fp32
    print(f"  A_mlx = {agree_total}/{steps_total} = {a_mlx:.6f}", flush=True)
    print(f"  KL_mlx = {kl_mlx:.6f} nats", flush=True)
    print(f"  ppl_mlx = {ppl_mlx:.6f}, dppl_mlx = {dppl_mlx:.6f}", flush=True)

    # ---- Committed quality fixtures ----------------------------------------
    QUALITY_ROOT.mkdir(parents=True, exist_ok=True)
    files = {}

    tokens_bytes = np.array(slice_ids, dtype="<i4").tobytes()
    (QUALITY_ROOT / "ppl_slice_tokens.bin").write_bytes(tokens_bytes)
    files["ppl_slice_tokens.bin"] = {
        "dtype": "int32", "shape": [PPL_SLICE_TOKENS],
        "byte_len": len(tokens_bytes), "sha256": sha256_bytes(tokens_bytes),
    }
    nll_bytes = ref_nll.astype("<f4").tobytes()
    (QUALITY_ROOT / "ppl_slice_ref_nll.bin").write_bytes(nll_bytes)
    files["ppl_slice_ref_nll.bin"] = {
        "dtype": "float32", "shape": [PPL_SLICE_TOKENS - 1],
        "byte_len": len(nll_bytes), "sha256": sha256_bytes(nll_bytes),
        "note": "diagnostic per-position fp32 NLL of the fp32 reference; "
                "ppl_fp32 in this file is the float64 scalar of record",
    }

    band = {
        "role": "P3-3 quality-gate band-setters + ppl slice. Binding gate "
                "record: DECISIONS.md 2026-08-25 'Phase 3 gates pre-committed' "
                "(veto-closed 2026-08-26); this file carries the measured "
                "numbers and the protocol.",
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "mlx_model": {"repo": MLX_REPO, "revision": MLX_REVISION},
        "dataset": {
            "repo": WIKITEXT_REPO, "revision": WIKITEXT_REVISION,
            "config": WIKITEXT_CONFIG, "split": WIKITEXT_SPLIT,
            "file": WIKITEXT_TEST_FILE, "parquet_sha256": parquet_sha,
            "row_count": row_count,
        },
        "versions": {p: pkg_version(p) for p in PINNED_PACKAGES},
        "python": sys.version.split()[0],
        "protocol": {
            "generator": "tools/dump_quality_gate.py",
            "command": "tools/.venv/bin/python tools/dump_quality_gate.py",
            "concatenation": '"".join(parquet text rows, stored order) — '
                             'non-empty rows carry their trailing newline',
            "tokenization": "pinned HF tokenizer, add_special_tokens=False, "
                            f"first {PPL_SLICE_TOKENS} tokens",
            "ppl": "exp(mean NLL) over positions 1..4095, one causal window, "
                   "float64 NLL over fp32 logits, KV-cached chunked prefill "
                   f"(chunk {PPL_CHUNK})",
            "teacher_forcing": "committed input_ids + committed reference "
                               "argmax ids (identical prefixes, 250 steps)",
            "agreement": "argmax(engine fp32 logits, first-index tie-break) "
                         "== committed reference argmax id",
            "kl": "KL(P_fp32 || P_engine) nats, float64 full-vocab softmax, "
                  "mean over 250 steps",
            "mlx_compute": "as-shipped mlx-lm 4-bit (fp16 activations); "
                           "logits cast fp32 before float64 statistics",
            "hf_reference_verification": "byte-identical committed checkpoint "
                                         "blobs at steps {0,1,24,49} x 5 "
                                         "prompts; argmax == steps.json at "
                                         "all 250 steps",
        },
        "reference": {
            "ppl_fp32": ppl_fp32,
            "ref_logits_artifact": {
                "path": f"models/{REF_LOGITS_NAME}",
                "dtype": "float32", "shape": list(ref.shape),
                "byte_len": len(ref_bytes), "sha256": ref_sha,
                "prompt_order": prompt_order,
                "note": "local-only (152 MB >> fixture budget); sha256 also "
                        "recorded in DECISIONS.md; Swift suite skips if absent",
            },
        },
        "band_setters": {
            "A_mlx": a_mlx,
            "A_mlx_count": agree_total,
            "steps_total": steps_total,
            "KL_mlx_nats": kl_mlx,
            "ppl_mlx": ppl_mlx,
            "dppl_mlx": dppl_mlx,
            "per_prompt": per_prompt,
        },
        "gates_reminder": {
            "agreement": "A_ours >= A_mlx - 0.04",
            "kl": "KL_ours <= 1.5 * KL_mlx_nats",
            "dppl": "dppl_ours <= 1.5 * dppl_mlx + 0.01",
            "note": "constants are binding in DECISIONS.md, restated here "
                    "for the reader only",
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    out = QUALITY_ROOT / "band.json"
    out.write_text(json.dumps(band, indent=1) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
