"""Swarm-for-swarm demo animation — the investor money shot.

Runs the SAME incoming attack swarm against two defenses, side by side:

  LEFT  "Legacy WTA doctrine"  — all-kinetic interceptors, engage-everything
  RIGHT "GRECKO"             — economic allocator + cost-adapted loadout

Both stop the swarm; the live $-spent and cost-per-kill counters tell the real
story — counter-swarm is a cost-exchange problem, and GRECKO wins it.

Renders docs/figures/swarm_demo.gif. This is a SIMULATION: effectors are
parameter sets (cost, Pk); nothing here is a weapon or hardware action.

    python -m tools.make_demo_gif
"""
from __future__ import annotations

import math
import pathlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from league.policy import SwarmPolicy
from sim.bridge.scenario import BridgeScenario
from sim.effectors.catalogue import CATALOGUE

FIGDIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "figures"

BG = "#0d1117"; FG = "#e6edf3"; GRID = "#30363d"; MUTED = "#8b949e"
CYAN = "#39d0d8"; GREEN = "#3fb950"; ORANGE = "#f0883e"; RED = "#f85149"; GOLD = "#f2cc60"

SIM_SECONDS = 26.0
DT = 0.02
FRAME_EVERY = 11          # sim ticks per rendered frame
N_ATTACK = 11
N_DEFENSE = 8
ARENA = 620.0

# A saturating raid: 11 UAS, mostly a main axis from the north with a small feint.
_ATTACK_THETA = np.array([
    float(N_ATTACK),   # n_total
    0.18,              # feint_frac
    0.0,               # screen_frac
    math.pi / 2,       # main_angle (north)
    math.pi / 2.5,     # feint_offset
    24.0,              # main_speed
    16.0,              # feint_speed
    560.0,             # main_range
    470.0,             # feint_range
    9.0,               # t_feint_turn
    120.0,             # main_spread
    90.0,              # feint_spread
    0.06,              # weave_amp
    0.0,               # timing_offset
])


def _attack_policy() -> SwarmPolicy:
    return SwarmPolicy(theta=_ATTACK_THETA.copy())


def _run(loadout, lambda_cost, seed=7):
    """Run one scenario, recording per-frame state for the animation."""
    sc = BridgeScenario(
        seed=seed, auto_authorize=True, policy=_attack_policy(),
        loadout=loadout,
    )
    sc.c2_state.lambda_cost = lambda_cost

    frames = []
    n_ticks = int(SIM_SECONDS / DT)
    seen_intercepts = 0
    spend = 0.0
    flashes = []  # (x, y, age)

    for tick in range(n_ticks):
        if sc.world.is_engagement_over():
            # keep last state to pad the animation tail
            pass
        else:
            sc.tick()

        # tally new intercepts from the event log
        ints = [e for e in sc.world.log.events if e["type"] == "INTERCEPT"]
        while seen_intercepts < len(ints):
            e = ints[seen_intercepts]
            iv = sc.world.interceptors.get(e["data"]["interceptor_id"])
            if iv is not None and iv.effector_type in CATALOGUE:
                spend += CATALOGUE[iv.effector_type].cost_usd
            flashes.append([float(e["position"][0]), float(e["position"][1]), 0])
            seen_intercepts += 1

        if tick % FRAME_EVERY == 0:
            hostiles = np.array([h.pos for h in sc.world.hostiles.values() if h.alive]) \
                if any(h.alive for h in sc.world.hostiles.values()) else np.empty((0, 2))
            ivs = np.array([iv.pos for iv in sc.world.interceptors.values() if iv.alive]) \
                if any(iv.alive for iv in sc.world.interceptors.values()) else np.empty((0, 2))
            frames.append({
                "t": sc.world.t,
                "hostiles": hostiles,
                "interceptors": ivs,
                "flashes": [f[:] for f in flashes if f[2] < 6],
                "intercepts": seen_intercepts,
                "spend": spend,
                "hp": sc.world.asset.hp,
            })
            for f in flashes:
                f[2] += 1

    return frames


def _draw_panel(ax, fr, title, accent, n_threats):
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_xlim(-ARENA, ARENA); ax.set_ylim(-ARENA, ARENA)
    ax.set_aspect("equal"); ax.axis("off")

    # range rings
    for r in (200, 400, 600):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ec=GRID, lw=0.6, alpha=0.5))

    # asset
    hp = fr["hp"]
    ax.scatter([0], [0], marker="*", s=420,
               c=GREEN if hp > 6 else (GOLD if hp > 2 else RED),
               edgecolors=FG, linewidths=1.2, zorder=5)

    # interceptors (defense swarm)
    iv = fr["interceptors"]
    if len(iv):
        ax.scatter(iv[:, 0], iv[:, 1], marker="o", s=70, c=accent,
                   edgecolors=FG, linewidths=0.6, zorder=4)
    # hostiles (attack swarm)
    h = fr["hostiles"]
    if len(h):
        ax.scatter(h[:, 0], h[:, 1], marker="^", s=64, c=RED,
                   edgecolors="#7a1d18", linewidths=0.6, zorder=4)
    # intercept flashes
    for fx, fy, age in fr["flashes"]:
        ax.scatter([fx], [fy], marker="x", s=200 - age * 26, c=GOLD,
                   linewidths=2.4, zorder=6, alpha=max(0.15, 1 - age / 6))

    # title + counters
    ax.set_title(title, color=FG, fontsize=14, fontweight="bold", pad=10)
    spend = fr["spend"]; stopped = fr["intercepts"]
    cpk = spend / stopped if stopped else 0.0
    ax.text(-ARENA * 0.94, ARENA * 0.9, f"threats stopped  {stopped}/{n_threats}",
            color=FG, fontsize=11, fontweight="bold")
    ax.text(-ARENA * 0.94, ARENA * 0.78, f"spent  ${spend:,.0f}",
            color=accent, fontsize=13, fontweight="bold")
    ax.text(-ARENA * 0.94, ARENA * 0.67, f"cost / kill  ${cpk:,.0f}",
            color=MUTED, fontsize=10.5)
    ax.text(ARENA * 0.94, -ARENA * 0.93, f"t = {fr['t']:4.1f}s",
            color=MUTED, fontsize=9.5, ha="right")


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("running legacy doctrine (all-kinetic, greedy)...")
    legacy = _run(loadout=["kinetic_interceptor"] * N_DEFENSE, lambda_cost=0.0)
    print("running GRECKO (economic + adapted loadout)...")
    # cost-adapted: cheap collision drones, economic rationing
    aegis = _run(loadout=["collision_drone"] * N_DEFENSE, lambda_cost=0.1)

    n = min(len(legacy), len(aegis))
    legacy, aegis = legacy[:n], aegis[:n]

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.6, 6.6), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.06, wspace=0.04)
    fig.suptitle("Same swarm, two doctrines — counter-swarm is a cost-exchange problem",
                 color=FG, fontsize=15, fontweight="bold", y=0.985)

    def update(i):
        _draw_panel(axl, legacy[i], "Legacy WTA doctrine", ORANGE, N_ATTACK)
        _draw_panel(axr, aegis[i], "GRECKO", CYAN, N_ATTACK)
        # verdict banner once both finish engaging
        if i == n - 1:
            ls, as_ = legacy[i]["spend"], aegis[i]["spend"]
            if as_ > 0:
                fig.text(0.5, 0.5, f"{ls / max(as_,1):.0f}× cheaper",
                         color=GREEN, fontsize=30, fontweight="bold",
                         ha="center", va="center", alpha=0.92,
                         bbox=dict(boxstyle="round,pad=0.5", fc=BG, ec=GREEN, lw=2))
        return []

    print(f"rendering {n} frames...")
    anim = FuncAnimation(fig, update, frames=n, interval=90, blit=False)
    out = FIGDIR / "swarm_demo.gif"
    anim.save(out, writer=PillowWriter(fps=11), dpi=66,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)

    ls, as_ = legacy[-1], aegis[-1]
    print(f"legacy : stopped {ls['intercepts']}/{N_ATTACK}, spent ${ls['spend']:,.0f}")
    print(f"aegis  : stopped {as_['intercepts']}/{N_ATTACK}, spent ${as_['spend']:,.0f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
