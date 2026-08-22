"""Primary reference dump: HF transformers, CPU, fp32 (docs/phases/phase-0-1.md).

Produces the committed oracle fixture set under tests/fixtures/qwen3-1.7b/:
  - Full-vocab fp32 logits at generation steps {0, 1, 24, 49} of a 50-step
    greedy generation, for each of the 5 pinned prompts.
  - Per-step scalar fingerprints (logsumexp, mean, std — computed in float64
    over the fp32 logit vector), top-1-vs-top-2 margin, argmax sequence.
  - Per-step top-64 logit values + indices.
  - Per-module activation slices for the activation prompt (prompt #1) from the
    prefill forward pass.
  - Python tokenizer ids for all 5 prompts (tokenizer_ids.json).
  - manifest.json with pins, versions, protocol, and sha256 of every blob.

Run (from repo root):  tools/.venv/bin/python tools/dump_reference.py

Determinism notes: greedy argmax (torch.argmax, first-index tie-break), fp32
weights/activations, single fixed process configuration. Generation does NOT
stop on EOS — fixtures always cover all 50 steps. The decode loop uses the HF
KV-cache path (prefill once, then one token per forward).
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
    ACTIVATION_PROMPT_ID,
    CHECKPOINT_STEPS,
    ENABLE_THINKING,
    FIXTURE_DIRNAME,
    GENERATION_STEPS,
    MODEL_REPO,
    MODEL_REVISION,
    TOP_K,
    VOCAB_SIZE,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / FIXTURE_DIRNAME

# Hook points documented for the Swift per-module tests (phase-0-1.md list).
# All slices are [seq, hidden] fp32 from the PREFILL forward of the
# activation prompt.
ACTIVATION_SLICES = {
    "embeddings_output": "output of model.model.embed_tokens",
    "layer0_pre_attn_norm_output": "output of model.model.layers[0].input_layernorm",
    "layer0_attn_output": "output of model.model.layers[0].self_attn (after o_proj, before residual add)",
    "layer0_block_output": "output of model.model.layers[0] (post-MLP, residuals included)",
    "last_layer_output": "output of model.model.layers[-1] (residuals included)",
    "final_norm_output": "output of model.model.norm",
}

PINNED_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "numpy")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_blob(files: dict, relpath: str, array: np.ndarray) -> None:
    """Write a little-endian binary blob and record it in the manifest index."""
    if array.dtype == np.float32:
        dtype = "float32"
        data = array.astype("<f4", copy=False).tobytes()
    elif array.dtype == np.int32:
        dtype = "int32"
        data = array.astype("<i4", copy=False).tobytes()
    else:
        raise ValueError(f"unsupported fixture dtype: {array.dtype}")
    path = FIXTURE_ROOT / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    files[relpath] = {
        "dtype": dtype,
        "shape": list(array.shape),
        "byte_len": len(data),
        "sha256": sha256_bytes(data),
    }


def load_prompts() -> list:
    with open(Path(__file__).resolve().parent / "fixture_prompts.json") as f:
        return json.load(f)["prompts"]


def render_input_text(tokenizer, prompt: dict) -> str:
    if prompt["kind"] == "raw":
        return prompt["text"]
    if prompt["kind"] == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt["text"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    raise ValueError(f"unknown prompt kind: {prompt['kind']}")


def fingerprints(logits32: np.ndarray) -> dict:
    """Scalar fingerprints computed in float64 over the fp32 logit vector."""
    x = logits32.astype(np.float64)
    m = x.max()
    return {
        "logsumexp": float(m + np.log(np.exp(x - m).sum())),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def dump_prompt(model, input_ids: torch.Tensor, prompt_id: str, files: dict) -> dict:
    """Greedy-decode GENERATION_STEPS tokens; return the steps.json payload."""
    prompt_dir = f"prompts/{prompt_id}"
    steps = []
    argmax_ids = []
    top_values = np.zeros((GENERATION_STEPS, TOP_K), dtype=np.float32)
    top_indices = np.zeros((GENERATION_STEPS, TOP_K), dtype=np.int32)

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[0, -1]

    for step in range(GENERATION_STEPS):
        logits32 = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        assert logits32.shape == (VOCAB_SIZE,), logits32.shape

        if step in CHECKPOINT_STEPS:
            write_blob(files, f"{prompt_dir}/logits_step{step:04d}.bin", logits32)

        order = np.argsort(-logits32, kind="stable")[:TOP_K]
        top_indices[step] = order.astype(np.int32)
        top_values[step] = logits32[order]

        record = fingerprints(logits32)
        record["step"] = step
        record["top1_id"] = int(order[0])
        record["top2_id"] = int(order[1])
        record["margin_top1_top2"] = float(
            np.float64(logits32[order[0]]) - np.float64(logits32[order[1]])
        )
        steps.append(record)

        next_id = int(torch.argmax(logits).item())
        argmax_ids.append(next_id)

        if step < GENERATION_STEPS - 1:
            out = model(
                input_ids=torch.tensor([[next_id]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[0, -1]

    write_blob(files, f"{prompt_dir}/top64_values.bin", top_values)
    write_blob(files, f"{prompt_dir}/top64_indices.bin", top_indices)

    payload = {"prompt_id": prompt_id, "argmax_token_ids": argmax_ids, "steps": steps}
    path = FIXTURE_ROOT / prompt_dir / "steps.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return payload


def capture_activations(model, input_ids: torch.Tensor, files: dict) -> None:
    captured = {}

    def hook(name):
        def fn(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = (
                tensor.detach()[0].cpu().numpy().astype(np.float32, copy=False)
            )

        return fn

    inner = model.model
    handles = [
        inner.embed_tokens.register_forward_hook(hook("embeddings_output")),
        inner.layers[0].input_layernorm.register_forward_hook(
            hook("layer0_pre_attn_norm_output")
        ),
        inner.layers[0].self_attn.register_forward_hook(hook("layer0_attn_output")),
        inner.layers[0].register_forward_hook(hook("layer0_block_output")),
        inner.layers[-1].register_forward_hook(hook("last_layer_output")),
        inner.norm.register_forward_hook(hook("final_norm_output")),
    ]
    try:
        model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    missing = set(ACTIVATION_SLICES) - set(captured)
    if missing:
        raise RuntimeError(f"activation hooks missed: {sorted(missing)}")
    for name in ACTIVATION_SLICES:
        write_blob(files, f"activations/{name}.bin", captured[name])


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    print(f"loading {MODEL_REPO} @ {MODEL_REVISION[:8]} (fp32, CPU)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model.eval()

    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    files: dict = {}
    prompts = load_prompts()

    tokenizer_ids = {}
    for prompt in prompts:
        text = render_input_text(tokenizer, prompt)
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        tokenizer_ids[prompt["id"]] = {
            "kind": prompt["kind"],
            "raw_text": prompt["text"],
            "input_text": text,
            "input_ids": ids,
        }

    for prompt in prompts:
        pid = prompt["id"]
        ids = torch.tensor([tokenizer_ids[pid]["input_ids"]], dtype=torch.long)
        print(f"dumping prompt '{pid}' ({ids.shape[1]} tokens)...", flush=True)
        payload = dump_prompt(model, ids, pid, files)
        preview = tokenizer.decode(payload["argmax_token_ids"])
        print(f"  argmax continuation: {preview[:80]!r}", flush=True)
        if pid == ACTIVATION_PROMPT_ID:
            print("  capturing per-module activations...", flush=True)
            capture_activations(model, ids, files)

    ids_path = FIXTURE_ROOT / "tokenizer_ids.json"
    ids_path.write_text(json.dumps(tokenizer_ids, indent=1, ensure_ascii=False) + "\n")

    manifest = {
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "versions": {p: pkg_version(p) for p in PINNED_PACKAGES},
        "python": sys.version.split()[0],
        "protocol": {
            "generator": "tools/dump_reference.py",
            "command": "tools/.venv/bin/python tools/dump_reference.py",
            "dtype": "float32 weights and activations, CPU",
            "attn_implementation": "sdpa",
            "sampling": "greedy (torch.argmax, first-index tie-break), no EOS stop",
            "generation_steps": GENERATION_STEPS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "top_k": TOP_K,
            "enable_thinking": ENABLE_THINKING,
            "kv_cache": "HF cache path: prefill once, then one token per forward",
            "fingerprints": "logsumexp/mean/std computed in float64 over the fp32 logit vector",
            "activation_prompt": ACTIVATION_PROMPT_ID,
            "activation_slices": ACTIVATION_SLICES,
            "binary_format": "raw little-endian, row-major; dtype/shape per manifest entry",
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": dict(sorted(files.items())),
    }
    (FIXTURE_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    total = sum(entry["byte_len"] for entry in files.values())
    print(f"wrote {len(files)} blobs, {total / 1e6:.2f} MB total -> {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()
