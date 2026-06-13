"""P1 demo: overlay truth tracks vs raw sensor reports for one engagement.

Usage: python -m sim.sensing.demo [--out docs/figures/p1_sensing_demo.png]
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from sim.core.events import EventLog
from sim.core.scenario import load_scenario
from sim.sensing.sensors import AcousticSensor, EOIRSensor, RadarSensor, SensorMesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/figures/p1_sensing_demo.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    log = EventLog()
    world = load_scenario("eval/scenarios/baseline.yaml", rng, log)

    mesh = SensorMesh([
        RadarSensor("radar_0", np.array([0.0, 0.0])),
        EOIRSensor("eoir_0", np.array([100.0, 0.0]), fov_center=math.pi),
        AcousticSensor("ac_0", np.array([0.0, 100.0])),
    ])

    truth_paths: dict[str, list] = {h: [] for h in world.hostiles}
    cart_reports = []
    bearing_reports = []

    for _ in range(int(30.0 / world.dt)):
        for hid, h in world.hostiles.items():
            if h.alive:
                truth_paths[hid].append(h.pos.copy())
        positions = [h.pos for h in world.hostiles.values() if h.alive]
        for rep in mesh.scan_all(world.t, positions, rng):
            if rep.kind == "cartesian":
                cart_reports.append(rep.position)
            else:
                bearing_reports.append((rep.sensor_pos, rep.bearing))
        world.step()
        if world.is_engagement_over():
            break

    fig, ax = plt.subplots(figsize=(9, 9))
    for hid, path in truth_paths.items():
        if path:
            arr = np.array(path)
            ax.plot(arr[:, 0], arr[:, 1], "-", lw=1.5, label=f"truth {hid}")
    if cart_reports:
        arr = np.array(cart_reports)
        ax.scatter(arr[:, 0], arr[:, 1], s=6, c="gray", alpha=0.5, label="cartesian reports")
    for spos, b in bearing_reports[:200]:
        end = spos + 300.0 * np.array([math.cos(b), math.sin(b)])
        ax.plot([spos[0], end[0]], [spos[1], end[1]], "r-", lw=0.3, alpha=0.3)
    ax.scatter([0], [0], marker="*", s=200, c="green", label="asset")
    ax.set_title("P1 — truth tracks vs raw sensor reports")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_aspect("equal")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}: {len(cart_reports)} cartesian + {len(bearing_reports)} bearing reports")


if __name__ == "__main__":
    main()
