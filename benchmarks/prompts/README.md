# Pinned benchmark prompt set (parity pin — PLAN.md benchmark protocol)

Exact prompt strings for every comparative row. Changing any string invalidates
cross-engine comparability and requires a DECISIONS.md entry. Drafted at P0A-1
prep (2026-08-22); becomes load-bearing with the first baseline row.

## The prompts and their roles

| File | Role | Prompt tokens (HF @ 70d244cc, rendered, non-thinking) | Verified generation behavior |
|---|---|---|---|
| `decode-essay.txt` | **decode / sustained / energy rows** | 84 | greedy runs ≥600 tokens with NO EOS (verified via mlx-lm 4-bit) — safely covers the canonical window (generated tokens 128–512) |
| `prefill-summarize.txt` | **prefill rows only** | 852 | greedy EOSes at ~425 tokens (mlx-lm 4-bit) — NEVER use it for decode-window or sustained rows |

Role separation is part of the pin: decode tok/s, the regenerate-loop sustained
protocol, and battery-delta energy rows all use `decode-essay`; prefill tok/s
uses `prefill-summarize` (its generation length is irrelevant to prefill).

## Template application (thinking-mode parity pin)

The template of record is the pinned base revision's
(`Qwen/Qwen3-1.7B @ 70d244cc`), applied with `enable_thinking=false`. Two
equivalent feeding modes — each results row records which one was used:

- **Engine applies its own template** (e.g. MLX LLMEval): feed the raw
  `*.txt` user text; configure the engine for non-thinking mode.
- **Engine takes a bare completion string** (e.g. llama.cpp example app): feed
  the exact `rendered/*.rendered.txt` string (regenerate with
  `tools/.venv/bin/python tools/render_bench_prompts.py`).

Any `<think>` content in OUTPUT invalidates the row. (The rendered prompt
itself legitimately ends with an empty `<think>\n\n</think>` block — that IS
the non-thinking form; only thinking content in generated text is
disqualifying.)

Per-engine prompt token counts must be reported with every row (GGUF and HF
tokenizers can differ).

## Sustained regenerate-loop operationalization

Pinned policy (PLAN.md): reset and regenerate from the same prompt when the 4K
context fills. Operational detail for engines that stop at EOS: restart
generation on EOS **or** context-fill, whichever comes first — identical
policy for every engine in a comparative row.
