"""Generate the figures for the README / GitHub Pages site.

Renders four panels into docs/figures/ from measured, deterministic study
outputs:

  banner.png                 hero banner (title + headline stat)
  headline_cost_exchange.png PX: EconomicMDP vs GreedyMyopic
  loadout_cost_lever.png     PM: cost-per-intercept by effector loadout
  coevolution_arc.png        PM: default -> adapted Blue -> Red counter-evolve

The numeric inputs are the deterministic results recorded in the ADRs; the PM
arc is read from a study JSON if present (tools/make_figures.py --pm <path>).
This script only renders — it runs no simulations.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

FIGDIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "figures"

# Palette (dark, high-contrast)
BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
CYAN = "#39d0d8"
ORANGE = "#f0883e"
GREEN = "#3fb950"
RED = "#f85149"
MUTED = "#8b949e"


def _style(ax, title=None):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG)
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)
    if title:
        ax.set_title(title, color=FG, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)


def _fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig


# --------------------------------------------------------------------------- #

def make_banner():
    fig = _fig(12, 3.2)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.text(0.04, 0.66, "AEGISNET", color=FG, fontsize=58, fontweight="bold",
            family="monospace", va="center")
    ax.text(0.045, 0.30, "counter-swarm air-defense simulation & research testbed",
            color=CYAN, fontsize=17, va="center")
    ax.text(0.045, 0.12, "sense → fuse → classify → predict → allocate → engage  ·  "
                         "human-on-the-loop C2  ·  adversarial co-evolution",
            color=MUTED, fontsize=11.5, va="center")
    # right-side stat
    ax.text(0.97, 0.62, "−22%", color=GREEN, fontsize=46, fontweight="bold",
            ha="right", va="center")
    ax.text(0.97, 0.27, "cost per intercepted threat\nvs greedy baseline",
            color=MUTED, fontsize=10.5, ha="right", va="center")
    fig.savefig(FIGDIR / "banner.png", dpi=140, facecolor=BG)
    plt.close(fig)


def make_headline():
    """PX headline: EconomicMDP vs GreedyMyopic (ADR-010)."""
    fig = _fig(7.6, 4.4)
    ax = fig.add_subplot(111)
    _style(ax, "Cost-exchange: economic allocator vs greedy baseline")
    labels = ["EconomicMDP", "GreedyMyopic"]
    cer = [24.8, 32.0]
    bars = ax.bar(labels, cer, color=[CYAN, MUTED], width=0.55,
                  edgecolor=BG, linewidth=2)
    ax.set_ylabel("cost-exchange ratio  ($ spent / $ intercepted)")
    ax.bar_label(bars, fmt="%.1f", color=FG, padding=4, fontweight="bold")
    ax.text(0.5, 0.92, "−22% cost per intercept", transform=ax.transAxes,
            ha="center", color=GREEN, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(cer) * 1.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / "headline_cost_exchange.png", dpi=150, facecolor=BG)
    plt.close(fig)


def make_loadout_lever():
    """PM cost lever: CER by loadout at ~constant intercept rate (measured)."""
    fig = _fig(8.4, 4.4)
    ax = fig.add_subplot(111)
    _style(ax, "Effector loadout is a pure cost lever (intercept rate ~constant)")
    labels = ["all-kinetic", "default\n(k,n,n)", "all-net", "all-collision"]
    cer = [90.0, 32.0, 3.0, 0.8]
    ir = [0.25, 0.22, 0.23, 0.22]
    colors = [RED, ORANGE, CYAN, GREEN]
    bars = ax.bar(labels, cer, color=colors, width=0.6, edgecolor=BG, linewidth=2)
    ax.set_yscale("log")
    ax.set_ylabel("cost-exchange ratio (log scale)")
    ax.bar_label(bars, fmt="%.1f", color=FG, padding=4, fontweight="bold")
    # intercept-rate annotation row
    for i, r in enumerate(ir):
        ax.text(i, 0.18, f"IR {r:.0%}", ha="center", color=MUTED, fontsize=9)
    ax.set_ylim(0.15, 200)
    fig.tight_layout()
    fig.savefig(FIGDIR / "loadout_cost_lever.png", dpi=150, facecolor=BG)
    plt.close(fig)


def make_coevolution(pm: dict | None):
    fig = _fig(8.0, 4.4)
    ax = fig.add_subplot(111)
    _style(ax, "Mutual co-evolution: Blue adapts, Red counter-evolves")

    if pm:
        base = pm["baseline_blue"]["mean_cost_exchange_ratio"]
        adapt = pm["adapted_blue"]["mean_cost_exchange_ratio"]
        counter = pm.get("counter_evolved", {}).get(
            "mean_cost_exchange_ratio", adapt)
    else:  # fallback to recorded ADR-012 numbers
        base, adapt, counter = 32.0, 0.8, 0.8

    labels = ["default Blue\n(vs PL tactics)", "adapted Blue\n(loadout + λ)",
              "adapted Blue\n(vs counter-evolved Red)"]
    vals = [base, adapt, counter]
    colors = [MUTED, GREEN, CYAN]
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor=BG, linewidth=2)
    ax.set_ylabel("cost-exchange ratio")
    ax.bar_label(bars, fmt="%.1f", color=FG, padding=4, fontweight="bold")
    if base > 0:
        ax.text(0.5, 0.9, f"−{(1 - adapt / base) * 100:.0f}% Blue cost; "
                          "Red cannot claw back the cost axis",
                transform=ax.transAxes, ha="center", color=GREEN,
                fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    fig.savefig(FIGDIR / "coevolution_arc.png", dpi=150, facecolor=BG)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pm", default=None, help="path to a PM study JSON")
    args = parser.parse_args(argv)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    pm = None
    if args.pm and pathlib.Path(args.pm).exists():
        pm = json.loads(pathlib.Path(args.pm).read_text())

    make_banner()
    make_headline()
    make_loadout_lever()
    make_coevolution(pm)
    print(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
