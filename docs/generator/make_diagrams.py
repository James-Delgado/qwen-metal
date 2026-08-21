#!/usr/bin/env python3
"""Render all architecture diagrams for the qwen-metal design doc."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")
os.makedirs(OUT, exist_ok=True)

INK    = "#1b2432"   # near-black slate
BLUE   = "#2563eb"
BLUEF  = "#dbeafe"
ORANGE = "#ea580c"
ORANGEF= "#ffedd5"
GREEN  = "#059669"
GREENF = "#d1fae5"
GRAY   = "#64748b"
GRAYF  = "#f1f5f9"
RED    = "#dc2626"
REDF   = "#fee2e2"
PURPLE = "#7c3aed"
PURPLEF= "#ede9fe"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.edgecolor": INK,
})

def box(ax, x, y, w, h, label, fc, ec, fs=9, sub=None, lw=1.4, subfs=7.5, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=lw, mutation_aspect=1.0))
    weight = "bold" if bold else "normal"
    if sub:
        ax.text(x+w/2, y+h*0.62, label, ha="center", va="center", fontsize=fs,
                fontweight=weight, color=INK)
        ax.text(x+w/2, y+h*0.28, sub, ha="center", va="center", fontsize=subfs,
                color=GRAY)
    else:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=fs,
                fontweight=weight, color=INK)

def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.5, style="-|>", ls="-", ms=13):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 color=color, lw=lw, linestyle=ls,
                                 mutation_scale=ms, shrinkA=2, shrinkB=2))

# ----------------------------------------------------------------------------
# D1: System context — targets, shared engine package, dev loop
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 5.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

# Engine package (center)
ax.add_patch(FancyBboxPatch((2.6, 1.15), 4.8, 3.6, boxstyle="round,pad=0.02",
                            fc="#f8fafc", ec=INK, lw=2))
ax.text(5.0, 4.52, "QwenMetalEngine — shared Swift package (all engine logic)",
        ha="center", fontsize=8.6, fontweight="bold")

box(ax, 2.85, 3.55, 1.42, 0.72, "WeightLoader", BLUEF, BLUE, 8,
    sub="safetensors, mmap\n4-bit packed fmt")
box(ax, 4.42, 3.55, 1.42, 0.72, "Tokenizer", GRAYF, GRAY, 8,
    sub="swift-transformers\n(borrowed)")
box(ax, 5.99, 3.55, 1.28, 0.72, "Config", GRAYF, GRAY, 8, sub="config.json\nparser")
box(ax, 2.85, 2.55, 2.2, 0.78, "CPU Reference Path", GREENF, GREEN, 8.5,
    sub="fp32 modules + Accelerate\nsgemm wrapper  (oracle)")
box(ax, 5.2, 2.55, 2.07, 0.78, "Metal GPU Path", ORANGEF, ORANGE, 8.5,
    sub=".metal kernels + dispatch\n(the product)")
box(ax, 2.85, 1.4, 2.2, 0.92, "Runtime", PURPLEF, PURPLE, 8.5,
    sub="decode loop, KV cache,\nsampler, MTLBuffer mgmt")
box(ax, 5.2, 1.4, 2.07, 0.92, "Test + Bench Harness", GRAYF, GRAY, 8.5,
    sub="fixture diffs, kernel tests,\ntok/s + battery logging")

# Consumers
box(ax, 0.25, 3.3, 1.95, 1.05, "qwen-metal-cli", BLUEF, BLUE, 9.5,
    sub="macOS target\ndev workhorse:\ntests, profiling", subfs=7.5)
box(ax, 0.25, 1.5, 1.95, 1.05, "QwenMetalApp", ORANGEF, ORANGE, 9.5,
    sub="iOS target (thin UI)\nofficial numbers,\nbenchmark mode", subfs=7.5)
arrow(ax, 2.2, 3.82, 2.6, 3.6); arrow(ax, 2.2, 2.02, 2.6, 2.2)

# External inputs (right)
box(ax, 7.85, 3.9, 1.9, 0.85, "HF Hub", GRAYF, GRAY, 9,
    sub="pinned Qwen ~1.5–2B\nbf16 safetensors")
box(ax, 7.85, 2.6, 1.9, 0.85, "tests/fixtures", GREENF, GREEN, 9,
    sub="fp32 logit oracles\n(~13 MB, plain git)")
box(ax, 7.85, 1.3, 1.9, 0.85, "Xcode tooling", GRAYF, GRAY, 9,
    sub="Instruments, Metal\ndebugger, signing")
arrow(ax, 7.85, 4.3, 7.4, 4.1, color=GRAY); arrow(ax, 7.85, 3.0, 7.4, 2.95, color=GREEN)
arrow(ax, 7.85, 1.7, 7.4, 1.85, color=GRAY)

# Dev loop banner
ax.add_patch(Rectangle((0.25, 0.15), 9.5, 0.75, fc="#fffbeb", ec="#d97706", lw=1.2))
ax.text(5.0, 0.52, "Dev loop:  build + test + first-pass profile on macOS (identical Metal API)\n→  deploy to physical iPhone for all official benchmark numbers",
        ha="center", va="center", fontsize=8.2, color="#92400e")
plt.tight_layout(); plt.savefig(f"{OUT}/d1_system.png", dpi=200, bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------------
# D2: Decode-step data flow through one transformer layer
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 6.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

ax.text(0.38, 6.78, "One decode step (batch = 1, incremental via KV cache from Phase 2 onward)",
        fontsize=10, fontweight="bold")
box(ax, 0.2, 6.1, 1.7, 0.65, "token id (t)", GRAYF, GRAY, 9)
box(ax, 2.25, 6.1, 2.0, 0.65, "Embedding lookup", BLUEF, BLUE, 9, sub="row read, fp16")
arrow(ax, 1.9, 6.42, 2.25, 6.42)

# layer block: top row runs left-to-right, middle row runs RIGHT-TO-LEFT,
# KV cache lives in the corridor between them. No long elbow routes.
ax.add_patch(FancyBboxPatch((0.35, 2.35), 9.3, 3.35, boxstyle="round,pad=0.02",
                            fc="#fcfcfd", ec=INK, lw=1.8))
ax.text(5.0, 5.42, "× N layers  (Qwen: ~28)", ha="center", fontsize=9.5, fontweight="bold")
arrow(ax, 3.25, 6.1, 3.25, 5.7)

# top row (attention half, L→R)
box(ax, 0.6, 4.45, 1.55, 0.7, "RMSNorm", GREENF, GREEN, 8.5, sub="fp32 accum")
box(ax, 2.45, 4.45, 2.05, 0.7, "QKV projections", ORANGEF, ORANGE, 8.5,
    sub="fused dequant-matvec\n(4-bit weights)")
box(ax, 4.8, 4.45, 1.35, 0.7, "RoPE", GREENF, GREEN, 8.5, sub="rotate Q,K\n@ position t")
box(ax, 6.45, 4.45, 1.5, 0.7, "cache append", PURPLEF, PURPLE, 8.5, sub="write K,V\n(fp16)")
box(ax, 8.2, 4.45, 1.25, 0.7, "SDPA", ORANGEF, ORANGE, 8.5, sub="GQA, fused\nover cache")
arrow(ax, 2.15, 4.8, 2.45, 4.8); arrow(ax, 4.5, 4.8, 4.8, 4.8)
arrow(ax, 6.15, 4.8, 6.45, 4.8); arrow(ax, 7.95, 4.8, 8.2, 4.8)

# KV cache in the inter-row corridor, under append/SDPA
box(ax, 5.35, 3.55, 2.8, 0.55, "KV cache (preallocated)", PURPLEF, PURPLE, 8,
    sub="[layers × kv-heads × 4K ctx × head-dim]", subfs=6.4)
arrow(ax, 7.2, 4.45, 7.2, 4.10, color=PURPLE)                # append writes
arrow(ax, 7.9, 4.10, 8.5, 4.45, color=PURPLE)               # SDPA reads
ax.text(8.35, 3.98, "read past K,V", fontsize=6.8, color=PURPLE, ha="left")

# middle row (attention out → MLP half, R→L)
box(ax, 8.0, 2.55, 1.55, 0.7, "attn out proj", ORANGEF, ORANGE, 8.5,
    sub="dequant-matvec")
box(ax, 6.3, 2.55, 1.25, 0.7, "residual +", GRAYF, GRAY, 8.5)
box(ax, 4.55, 2.55, 1.5, 0.7, "RMSNorm", GREENF, GREEN, 8.5, sub="fp32 accum")
box(ax, 2.3, 2.55, 1.95, 0.7, "SwiGLU MLP", ORANGEF, ORANGE, 8.5,
    sub="gate/up/down: 3 ×\ndequant-matvec")
box(ax, 0.6, 2.55, 1.4, 0.7, "residual +", GRAYF, GRAY, 8.5)
arrow(ax, 8.82, 4.45, 8.82, 3.25)                            # SDPA → attn out proj
arrow(ax, 8.0, 2.9, 7.55, 2.9); arrow(ax, 6.3, 2.9, 6.05, 2.9)
arrow(ax, 4.55, 2.9, 4.25, 2.9); arrow(ax, 2.3, 2.9, 2.0, 2.9)

# exit after N layers → final head
ax.text(0.45, 1.9, "after\nlayer N", fontsize=6.8, color=GRAY, ha="left")
arrow(ax, 1.25, 2.35, 1.25, 1.2)
ax.text(5.5, 1.75, "orange = quantized-weight kernels (bandwidth-critical)     green = hand-rolled logic     purple = KV cache state",
        ha="center", fontsize=7.6, color=GRAY)

box(ax, 0.35, 0.55, 1.75, 0.65, "final RMSNorm", GREENF, GREEN, 8.5)
box(ax, 2.4, 0.55, 2.2, 0.65, "logits head", ORANGEF, ORANGE, 8.5,
    sub="dequant-matvec, 152k vocab")
box(ax, 4.95, 0.55, 2.15, 0.65, "sample (greedy/temp)", GRAYF, GRAY, 8.5)
box(ax, 7.5, 0.55, 2.1, 0.65, "next token → repeat", BLUEF, BLUE, 8.5)
arrow(ax, 2.1, 0.87, 2.4, 0.87); arrow(ax, 4.6, 0.87, 4.95, 0.87)
arrow(ax, 7.1, 0.87, 7.5, 0.87)
plt.savefig(f"{OUT}/d2_dataflow.png", dpi=200, bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------------
# D3: Memory budget under the iOS ceiling
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 3.4))
segs = [("Weights (4-bit, mmap, file-backed)", 1.30, BLUE, BLUEF),
        ("Scales + biases", 0.10, "#1d4ed8", "#bfdbfe"),
        ("KV cache @4K (fp16)", 0.15, PURPLE, PURPLEF),
        ("Activations + scratch", 0.10, GREEN, GREENF),
        ("App + runtime + tokenizer", 0.15, GRAY, GRAYF)]
total = sum(s[1] for s in segs)
ceiling = 3.5
x = 0
ax.set_xlim(0, ceiling + 0.35); ax.set_ylim(0, 2.4); ax.axis("off")
for name, sz, ec, fc in segs:
    ax.add_patch(Rectangle((x, 0.8), sz, 0.8, fc=fc, ec=ec, lw=1.5))
    x += sz
ax.add_patch(Rectangle((x, 0.8), ceiling - x, 0.8, fc="white", ec=GRAY, lw=1.2, ls="--"))
ax.text(x + (ceiling-x)/2, 1.2, f"headroom  ≈ {ceiling-total:.1f} GB\n(margin of safety vs. jetsam)",
        ha="center", va="center", fontsize=8, color=GRAY)
# labels fanned out to fixed anchors with leader lines (segments 2-5 are thin
# and clustered — labels at segment centers would pile on top of each other)
labels_x = [0.65, 1.30, 1.95, 2.60, 3.10]
labels_y = [2.18, 1.88, 2.18, 1.88, 2.18]
x = 0
for (name, sz, ec, fc), lx, ly in zip(segs, labels_x, labels_y):
    cx = x + sz/2
    if lx - cx > 1.2:
        # long reach: elbow (shallow diagonal under the other labels, then up)
        ax.plot([cx, lx, lx], [1.62, 1.70, ly - 0.16], color=ec, lw=0.9)
    else:
        ax.plot([cx, lx], [1.62, ly - 0.16], color=ec, lw=0.9)
    ax.text(lx, ly, f"{name}\n≈ {sz:.2f} GB", ha="center", fontsize=7.6, color=ec)
    x += sz
ax.plot([ceiling, ceiling], [0.62, 2.4], color=RED, lw=2)
ax.text(ceiling+0.05, 1.2, "practical iOS\nmemory ceiling\n(jetsam; with\nIncreased Memory\nLimit entitlement)",
        fontsize=7.6, color=RED, va="center")
ax.text(0, 0.45, "0 GB", fontsize=8, color=GRAY)
ax.text(ceiling-0.12, 0.45, f"{ceiling} GB", fontsize=8, color=GRAY)
ax.text(0, 0.1, "Budget rule: every allocation is accounted here before it is written. Weights are file-backed (mmap) — cheaper under iOS memory\naccounting than dirty heap pages. KV cache is preallocated at load; nothing grows at runtime.",
        fontsize=7.8, color=INK)
plt.savefig(f"{OUT}/d3_memory.png", dpi=200, bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------------
# D4: Fused vs unfused dequant — DRAM traffic per weight
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

ax.text(2.5, 4.75, "UNFUSED  (forbidden — invariant #2)", fontsize=10, fontweight="bold",
        color=RED, ha="center")
box(ax, 0.4, 3.55, 1.7, 0.75, "DRAM\n4-bit weights", GRAYF, GRAY, 8.5)
box(ax, 2.7, 3.55, 1.7, 0.75, "Dequant kernel", REDF, RED, 8.5)
box(ax, 0.4, 2.35, 1.7, 0.75, "DRAM\nfp16 scratch\n(~2.6 GB!)", REDF, RED, 8)
box(ax, 2.7, 2.35, 1.7, 0.75, "Matmul kernel", REDF, RED, 8.5)
arrow(ax, 2.1, 3.92, 2.7, 3.92, color=GRAY)
ax.text(2.38, 4.14, "read 0.5 B", fontsize=6.5, color=GRAY, ha="center")
arrow(ax, 2.7, 3.55, 2.1, 3.0, color=RED)
ax.text(2.63, 3.28, "write 2 B", fontsize=6.5, color=RED, ha="left")
arrow(ax, 2.1, 2.72, 2.7, 2.72, color=RED)
ax.text(2.4, 2.9, "read 2 B", fontsize=6.5, color=RED, ha="center")
ax.add_patch(Rectangle((0.3, 1.15), 4.5, 0.8, fc=REDF, ec=RED, lw=1.5))
ax.text(2.55, 1.55, "≈ 4.5 bytes of DRAM traffic per weight, per token\n(~9× the fused path)  +  a disqualifying scratch buffer",
        ha="center", va="center", fontsize=7.8, color=RED, fontweight="bold")

ax.text(7.4, 4.75, "FUSED  dequant-matvec  (the design)", fontsize=10, fontweight="bold",
        color=GREEN, ha="center")
box(ax, 5.5, 3.55, 1.7, 0.75, "DRAM\n4-bit weights\n+ scales/biases", GRAYF, GRAY, 8)
ax.add_patch(FancyBboxPatch((7.7, 2.5), 2.0, 1.85, boxstyle="round,pad=0.02",
                            fc=GREENF, ec=GREEN, lw=1.6))
ax.text(8.7, 4.05, "One kernel, per thread:", fontsize=8, fontweight="bold", ha="center")
ax.text(8.7, 3.28, "unpack nibbles\n(shift + mask)\nw = scale·q + bias\n→ registers only\nfma into fp32 acc",
        fontsize=7.8, ha="center", va="center")
arrow(ax, 7.2, 3.92, 7.7, 3.75, color=GRAY)
ax.text(7.42, 3.42, "read 0.5 B", fontsize=7.2, color=GRAY, ha="center")
box(ax, 7.85, 1.45, 1.7, 0.7, "output vector\n(fp16, tiny)", GREENF, GREEN, 8)
arrow(ax, 8.7, 2.5, 8.7, 2.15, color=GREEN)
ax.add_patch(Rectangle((5.3, 1.15), 2.3, 1.0, fc=GREENF, ec=GREEN, lw=1.5))
ax.text(6.45, 1.65, "≈ 0.5 bytes per weight,\nper token — weights\ntouch DRAM exactly once",
        ha="center", va="center", fontsize=7.8, color=GREEN, fontweight="bold")
ax.text(5.0, 0.55, "Decode roofline:  tok/s  ≈  DRAM bandwidth  ÷  bytes read per token.   Bandwidth is fixed by hardware;\nthe denominator is the only lever. Fusion is not an optimization on top of the design — it is the design.",
        ha="center", fontsize=8.4, color=INK)
plt.savefig(f"{OUT}/d4_fused.png", dpi=200, bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------------
# D5: Roofline chart
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 4.6))
bpt = np.linspace(0.4, 3.4, 300)  # GB per token
for bw, c, ls in [(50, "#93c5fd", "--"), (60, BLUE, "-"), (70, "#1e40af", "--")]:
    ax.plot(bpt, bw / bpt, color=c, ls=ls, lw=1.8, label=f"{bw} GB/s bandwidth")
# markers
ax.scatter([1.25], [60/1.25], s=70, color=ORANGE, zorder=5)
ax.annotate("~2B @ 4-bit (this project)\nceiling ≈ 48–56 tok/s", (1.25, 48),
            xytext=(1.5, 62), fontsize=8.5, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.scatter([3.1], [60/3.1], s=70, color=GRAY, zorder=5)
ax.annotate("same model @ fp16\n≈ 19 tok/s ceiling", (3.1, 19.4), xytext=(2.45, 33),
            fontsize=8.5, color=GRAY, arrowprops=dict(arrowstyle="->", color=GRAY))
ax.scatter([1.1], [61], s=90, color=GREEN, marker="*", zorder=6)
ax.annotate("MLX (published): ~61 tok/s\n(iPhone 17 Pro; provisional until\nPhase 0 rows land)", (1.1, 61),
            xytext=(1.45, 84), fontsize=8.5, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN))
ax.set_xlabel("Bytes read per generated token (GB)  ≈  packed weights + scales + KV reads", fontsize=9)
ax.set_ylabel("Decode tokens / second (ceiling)", fontsize=9)
ax.set_ylim(0, 130); ax.set_xlim(0.4, 3.4)
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Decode roofline — why quantization and fusion set the speed limit", fontsize=10.5,
             fontweight="bold", color=INK)
plt.tight_layout(); plt.savefig(f"{OUT}/d5_roofline.png", dpi=200, bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------------
# D6: Oracle chain / correctness DAG
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 5.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.3); ax.axis("off")

box(ax, 0.3, 5.3, 2.5, 0.8, "HF transformers, CPU\ntorch_dtype=float32", GREENF, GREEN, 8.5,
    sub="pinned revision — root oracle")
box(ax, 3.3, 5.3, 2.5, 0.8, "Slim fixtures (~13 MB)", GREENF, GREEN, 8.5,
    sub="fp32 checkpoints {0,1,mid,last},\ntop-64/step, fingerprints, margins")
box(ax, 6.3, 5.3, 2.5, 0.8, "mlx-lm (fp16)", GRAYF, GRAY, 8.5,
    sub="loose-tolerance ecosystem\nsanity check only")
arrow(ax, 2.8, 5.7, 3.3, 5.7, color=GREEN)
box(ax, 3.3, 3.95, 2.5, 0.85, "CPU fp32 reference (P1)", BLUEF, BLUE, 8.8,
    sub="plain Swift logic +\nAccelerate sgemm wrapper")
arrow(ax, 4.55, 5.3, 4.55, 4.8, color=GREEN)
ax.text(4.68, 5.03, "max |Δ| ≤ 1e-3\nno loosening", fontsize=7, color=GREEN)
arrow(ax, 7.5, 5.3, 5.5, 4.75, color=GRAY, ls=":")

box(ax, 0.3, 3.95, 2.4, 0.85, "sgemm wrapper test", GRAYF, GRAY, 8.3,
    sub="vs naive triple-loop,\nodd shapes (lives in test only)")
arrow(ax, 2.7, 4.37, 3.3, 4.37, color=GRAY)

box(ax, 6.3, 3.95, 2.6, 0.85, "Packing script (offline)", PURPLEF, PURPLE, 8.5,
    sub="4-bit grouped affine +\nadversarial round-trip fixtures")
box(ax, 3.3, 2.6, 2.5, 0.85, "CPU-quant reference (P3)", BLUEF, BLUE, 8.8,
    sub="same code path, weights\ndequantized from packed fmt")
arrow(ax, 4.55, 3.95, 4.55, 3.45, color=BLUE)
arrow(ax, 6.9, 3.95, 5.6, 3.3, color=PURPLE)
box(ax, 6.3, 2.6, 2.6, 0.85, "Quality gate", PURPLEF, PURPLE, 8.5,
    sub="top-1 agreement + KL vs mlx-lm\n4-bit; perplexity recorded")
arrow(ax, 6.3, 3.03, 5.8, 3.03, color=PURPLE, ls="--")

box(ax, 0.3, 1.3, 2.7, 0.85, "GPU dequant tile test", ORANGEF, ORANGE, 8.5,
    sub="bit-for-bit vs CPU dequant\n(deterministic — exact required)")
box(ax, 3.45, 1.3, 2.6, 0.85, "GPU fused matvec test", ORANGEF, ORANGE, 8.5,
    sub="tol ~1e-3/elem; fp32 accum both\nsides; only reduction order differs")
box(ax, 6.6, 1.3, 2.7, 0.85, "Full model on GPU", ORANGEF, ORANGE, 8.5,
    sub="end-to-end diff vs CPU-quant;\ncached path ≡ uncached (Phase 2)")
arrow(ax, 3.9, 2.6, 1.8, 2.15, color=BLUE)
arrow(ax, 4.6, 2.6, 4.7, 2.15, color=BLUE)
arrow(ax, 5.4, 2.6, 7.6, 2.15, color=BLUE)
arrow(ax, 3.0, 1.72, 3.45, 1.72, color=ORANGE)
arrow(ax, 6.05, 1.72, 6.6, 1.72, color=ORANGE)
ax.text(5.0, 0.6, "Design rule: no GPU result is ever diffed against an oracle that legitimately differs from it.\nEvery arrow above is a test that exists in the suite; the suite runs in minutes (Accelerate), so it actually runs.",
        ha="center", fontsize=8.6, color=INK)
plt.savefig(f"{OUT}/d6_oracle.png", dpi=200, bbox_inches="tight"); plt.close()

# ----------------------------------------------------------------------------
# D7: Phase roadmap
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.4, 4.9))
phases = [
    ("0", "Baselines + toy kernels", "MLX & llama.cpp measured on-phone; saxpy + naive matmul; profiling workflow", 0, 1.5, GRAY, GRAYF),
    ("1", "CPU fp32 reference (macOS)", "full Qwen forward pass; ≤1e-3 vs HF fp32 oracle; Accelerate matmuls", 1.0, 2.5, GREEN, GREENF),
    ("2", "Naive Metal port + minimal KV cache", "incremental decode on-device; cached ≡ uncached logits; 'before' row", 3.0, 2.0, BLUE, BLUEF),
    ("3", "4-bit quant + fused dequant-matvec", "CPU-quant oracle; tile test exact; matvec GB/s microbench; quality gate; roofline check", 4.5, 2.5, ORANGE, ORANGEF),
    ("4", "Fused attention + kernel fusion", "GQA SDPA kernel; fold norm/RoPE; dispatches/token down", 6.5, 2.5, ORANGE, ORANGEF),
    ("5", "Tiled prefill GEMM", "threadgroup memory + simdgroup_matrix; prefill vs MLX", 8.5, 2.0, PURPLE, PURPLEF),
    ("6", "Benchmark writeup", "full cross-engine table; thermal + J/tok (battery-delta); roofline analysis", 10.0, 1.5, INK, "#e2e8f0"),
]
ax.set_xlim(-0.2, 12.4); ax.set_ylim(-0.6, len(phases)*1.02); ax.axis("off")
for i, (num, name, desc, start, dur, ec, fc) in enumerate(reversed(phases)):
    y = i * 1.0
    ax.add_patch(FancyBboxPatch((start, y+0.18), dur, 0.62, boxstyle="round,pad=0.015",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(start+0.12, y+0.49, f"P{num}", fontsize=9, fontweight="bold", va="center", color=ec)
    ax.text(start+dur+0.15, y+0.62, name, fontsize=8.8, fontweight="bold", va="center", color=INK)
    ax.text(start+dur+0.15, y+0.28, desc, fontsize=7.2, va="center", color=GRAY)
for m in range(0, 13, 2):
    ax.axvline(m, color=GRAYF, lw=1, zorder=0)
    ax.text(m, -0.42, f"wk {m*2}", fontsize=7.5, color=GRAY, ha="center")
ax.text(-0.2, len(phases)*1.02 - 0.15,
        "Phase roadmap  (~6–10 h/week; bars ≈ elapsed weekends; specs written just-in-time from prior phase's results)",
        fontsize=9.5, fontweight="bold", color=INK)
plt.tight_layout(); plt.savefig(f"{OUT}/d7_roadmap.png", dpi=200, bbox_inches="tight"); plt.close()

print("diagrams done:", sorted(os.listdir(OUT)))
