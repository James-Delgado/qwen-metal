"""Secondary ecosystem dump: mlx-lm greedy generation (docs/phases/phase-0-1.md).

Loose-tolerance sanity check only — records what the MLX world produces for the
pinned 4-bit checkpoint (mlx-community/Qwen3-1.7B-4bit) on the 5 fixture
prompts. NOT an oracle: the primary HF fp32 dump is the oracle; this returns as
the Phase 3 quality-gate comparator.

Run AFTER dump_reference.py (it reuses the rendered chat-template text from
tokenizer_ids.json so the template of record stays the pinned base revision's):

    tools/.venv/bin/python tools/dump_mlx.py
"""

import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pins import FIXTURE_DIRNAME, GENERATION_STEPS, MLX_REPO, MLX_REVISION

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / FIXTURE_DIRNAME


def main() -> None:
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    import mlx.core as mx

    ids_path = FIXTURE_ROOT / "tokenizer_ids.json"
    if not ids_path.exists():
        sys.exit("tokenizer_ids.json missing — run dump_reference.py first")
    tokenizer_ids = json.loads(ids_path.read_text())

    print(f"loading {MLX_REPO} @ {MLX_REVISION[:8]}...", flush=True)
    local = snapshot_download(MLX_REPO, revision=MLX_REVISION)
    model, tokenizer = load(local)

    results = {}
    for pid, entry in tokenizer_ids.items():
        # Same input text as the primary dump (chat prompt already rendered
        # with the pinned base revision's template, enable_thinking=false).
        prompt_ids = tokenizer.encode(entry["input_text"])
        generated = []
        # sampler=None -> default greedy argmax
        for token, _logprobs in generate_step(
            mx.array(prompt_ids), model, max_tokens=GENERATION_STEPS, sampler=None
        ):
            generated.append(int(token))
        results[pid] = {
            "generated_token_ids": generated,
            "generated_text": tokenizer.decode(generated),
        }
        print(f"  {pid}: {results[pid]['generated_text'][:60]!r}", flush=True)

    payload = {
        "role": "secondary ecosystem check (loose, argmax-level) — NOT an oracle",
        "model": {"repo": MLX_REPO, "revision": MLX_REVISION},
        "versions": {p: pkg_version(p) for p in ("mlx", "mlx-lm")},
        "python": sys.version.split()[0],
        "protocol": {
            "generator": "tools/dump_mlx.py",
            "command": "tools/.venv/bin/python tools/dump_mlx.py",
            "sampling": "greedy (default argmax sampler), no EOS stop, "
            f"{GENERATION_STEPS} tokens",
            "input_text": "identical input_text strings as the primary dump "
            "(see tokenizer_ids.json)",
        },
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompts": results,
    }
    out = FIXTURE_ROOT / "mlx_secondary.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
