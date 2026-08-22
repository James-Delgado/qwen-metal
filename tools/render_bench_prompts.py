"""Render the pinned benchmark prompts through the pinned non-thinking chat
template (PIN-1 thinking-mode parity pin: template of record = the one in the
pinned base revision, enable_thinking=false).

Engines whose UI applies a chat template itself (e.g. LLMEval) take the raw
benchmarks/prompts/*.txt user text; engines fed a bare completion string
(e.g. llama.swiftui) take benchmarks/prompts/rendered/*.rendered.txt so every
engine sees the identical token stream. Each results row records which form
was used (parity pin: template application documented per engine).

Run (from repo root):  tools/.venv/bin/python tools/render_bench_prompts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pins import ENABLE_THINKING, MODEL_REPO, MODEL_REVISION

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "benchmarks" / "prompts"

PROMPT_NAMES = ("decode-essay", "prefill-summarize")


def main() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    out_dir = PROMPTS_DIR / "rendered"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in PROMPT_NAMES:
        raw = (PROMPTS_DIR / f"{name}.txt").read_text().strip()
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        out = out_dir / f"{name}.rendered.txt"
        out.write_text(rendered)
        count = len(tokenizer(rendered, add_special_tokens=True)["input_ids"])
        print(f"{name}: {count} tokens (HF tokenizer @ {MODEL_REVISION[:8]}) -> {out}")


if __name__ == "__main__":
    main()
