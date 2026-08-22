"""One-time shard consolidation (DECISIONS.md 2026-08-21 PIN-1 entry).

The pinned Qwen/Qwen3-1.7B checkpoint ships two safetensors shards; PLAN.md
pins a single-file-only engine parser that rejects shards loudly. This script
merges the pinned shards into the single-file artifact the engine consumes,
preserving dtypes (bf16) and byte content exactly. Tensor names are written in
sorted order for deterministic output.

Run (from repo root):

    tools/.venv/bin/python tools/consolidate_shards.py --out <path>.safetensors

The output is a local artifact (multi-GB, never committed). Record its sha256
+ source revision in DECISIONS.md when first produced.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pins import MODEL_REPO, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output .safetensors path")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    print(f"resolving {MODEL_REPO} @ {MODEL_REVISION[:8]}...", flush=True)
    snapshot = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION))

    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.exists():
        sys.exit("no model.safetensors.index.json — checkpoint is not sharded?")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    print(f"{len(weight_map)} tensors across {len(shards)} shards: {shards}", flush=True)

    tensors = {}
    for shard in shards:
        with safe_open(snapshot / shard, framework="pt") as f:
            for name in f.keys():
                if name in tensors:
                    sys.exit(f"duplicate tensor across shards: {name}")
                tensors[name] = f.get_tensor(name)
    if set(tensors) != set(weight_map):
        sys.exit("tensor set mismatch vs index weight_map")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        dict(sorted(tensors.items())),
        str(out_path),
        metadata={
            "format": "pt",
            "source_repo": MODEL_REPO,
            "source_revision": MODEL_REVISION,
            "consolidated_by": "tools/consolidate_shards.py",
        },
    )

    # Self-check: reopen the output and verify every tensor byte-for-byte.
    with safe_open(out_path, framework="pt") as f:
        out_names = set(f.keys())
        if out_names != set(tensors):
            sys.exit("output tensor set mismatch")
        for name in sorted(out_names):
            a, b = f.get_tensor(name), tensors[name]
            if a.dtype != b.dtype or a.shape != b.shape:
                sys.exit(f"dtype/shape mismatch for {name}")
            if a.view(-1).ne(b.view(-1)).any():
                sys.exit(f"byte content mismatch for {name}")
    print("self-check passed: all tensors match the source shards", flush=True)

    digest = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    size = out_path.stat().st_size
    print(f"wrote {out_path} ({size / 1e9:.3f} GB)")
    print(f"sha256: {digest.hexdigest()}")


if __name__ == "__main__":
    main()
