"""Figures for the Phase 6 analyzer (tools/phase6_analyze.py; task P6-3).

Thermal chart (D4: per-generation window rate vs elapsed, all cycles, median
cycle bold) and energy bars (D4: net J/token with statistical and systematic
error bars). Colors = the dataviz reference palette's first three categorical
slots (validated all-pairs in both modes); text stays in ink tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ENGINES = ("ours", "mlx", "llama")
ENGINE_LABELS = {"ours": "qwen-metal", "mlx": "MLX LLMEval", "llama": "llama.cpp"}
ENGINE_COLORS = {"ours": "#2a78d6", "mlx": "#eb6834", "llama": "#1baf7a"}


@dataclass
class ThermalSeries:
    engine: str
    file: str
    elapsed_s: list[float]
    window_tok_s: list[float]
    source: str          # json | text
    basis: str           # loopClock | sumOfGenerationWall
    bold: bool = False


@dataclass
class EnergyBar:
    engine: str
    mean: float
    statistical_halfwidth: float
    systematic_halfwidth: float


def render_thermal_chart(series: list[ThermalSeries], out: Path) -> Path:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    for s in series:
        if not s.window_tok_s:
            continue
        ax.plot([e / 60 for e in s.elapsed_s], s.window_tok_s, color=ENGINE_COLORS[s.engine],
                linewidth=2.0 if s.bold else 1.2, alpha=1.0 if s.bold else 0.35,
                label=ENGINE_LABELS[s.engine] if s.bold else None)
        if s.bold:
            ax.annotate(ENGINE_LABELS[s.engine], (s.elapsed_s[-1] / 60, s.window_tok_s[-1]),
                        textcoords="offset points", xytext=(6, 0), fontsize=9, color="#52514e", va="center")
    ax.set_xlabel("elapsed in the energy cycle (min)")
    ax.set_ylabel("decode tok/s, canonical window (128–512)")
    ax.set_title("Sustained-thermal: per-generation window rate, all cycles (median cycle bold)")
    _style_axes(ax)
    if any(s.bold for s in series):
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def energy_bar_data(energy: dict) -> list[EnergyBar]:
    bars = []
    for engine in ENGINES:
        h = energy[engine].headline
        if h is None:
            continue
        bars.append(EnergyBar(engine, h.mean_j_per_token, (h.spread[1] - h.spread[0]) / 2,
                              h.mean_j_per_token * h.systematic_fraction))
    return bars


def render_energy_bars(energy: dict, out: Path) -> Path:
    plt = _pyplot()
    bars = energy_bar_data(energy)
    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=150)
    xs = list(range(len(bars)))
    for x, b in zip(xs, bars):
        ax.bar(x, b.mean, width=0.55, color=ENGINE_COLORS[b.engine], linewidth=0)
        ax.errorbar(x - 0.08, b.mean, yerr=b.statistical_halfwidth, fmt="none", ecolor="#0b0b0b", capsize=4,
                    linewidth=1.2, label="statistical (min–max over n cycles)" if x == 0 else None)
        ax.errorbar(x + 0.08, b.mean, yerr=b.systematic_halfwidth, fmt="none", ecolor="#52514e", capsize=4,
                    linewidth=1.2, linestyle="--", label="systematic (SoC-reading quantization)" if x == 0 else None)
        ax.annotate(f"{b.mean:.3f} J", (x - 0.28, b.mean), textcoords="offset points", xytext=(0, 6),
                    ha="left", fontsize=9)
    ax.set_xticks(xs, [ENGINE_LABELS[b.engine] for b in bars])
    ax.set_ylabel("net J per token")
    ax.set_title("Energy per token — sustained, idle-subtracted")
    if bars:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    refused = [ENGINE_LABELS[e] for e in ENGINES if energy[e].headline is None]
    if refused:
        fig.text(0.01, 0.01, "no headline (below the ≥3 valid-cycle pin or refused): " + ", ".join(refused),
                 fontsize=8, color="#52514e", ha="left", va="bottom")
    _style_axes(ax)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out)
    plt.close(fig)
    return out


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_facecolor("#fcfcfb")
