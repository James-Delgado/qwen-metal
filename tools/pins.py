"""Pinned constants for the qwen-metal reference tooling.

Single source of truth for every pin the dump scripts depend on. Changing any
value here invalidates committed fixtures and requires a DECISIONS.md entry
(regenerating with unpinned versions is a bug — see docs/phases/phase-0-1.md).
"""

# Model pin (DECISIONS.md 2026-08-21 PIN-1 entry)
MODEL_REPO = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# Secondary (ecosystem) comparator pin (same PIN-1 entry)
MLX_REPO = "mlx-community/Qwen3-1.7B-4bit"
MLX_REVISION = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"

# Generation protocol (docs/phases/phase-0-1.md, correctness harness)
GENERATION_STEPS = 50
# 0-based step indices whose full-vocab logits are checkpointed: {0, 1, mid, last}
CHECKPOINT_STEPS = (0, 1, 24, 49)
TOP_K = 64
VOCAB_SIZE = 151936

# Thinking-mode parity pin (PIN-1): all dumps use the NON-thinking template.
ENABLE_THINKING = False

# Layout
FIXTURE_DIRNAME = "qwen3-1.7b"  # under tests/fixtures/
ACTIVATION_PROMPT_ID = "short_english"  # per-module activations for prompt #1 only
