"""PC degradation sweep: allocation quality vs comm radius.

Quality proxy (pre-Pillar-A): interceptors coordinate greedy assignments only
within their comms partition. Partitions cannot deconflict with each other, so
shrinking the comm radius produces duplicate assignments and uncovered
targets. Quality = fraction of targets covered by at least one interceptor.

This is the EVAL channel for the comms layer; it produces the degradation
curve plot required by the PC acceptance criterion.

Usage: python -m sim.comms.sweep [--out docs/figures/pc_degradation_curve.png]
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np

from sim.comms.links import CommsConfig
from sim.comms.network import CommsNetwork


def partition_greedy_quality(
    comm_radius: float,
    n_interceptors: int = 12,
    n_targets: int = 12,
    field: float = 2000.0,
    seed: int = 0,
) -> float:
    """One snapshot: place nodes/targets uniformly, allocate greedily per
    partition, return covered-target fraction."""
    rng = np.random.default_rng(seed)
    net = CommsNetwork(CommsConfig(comm_radius=comm_radius), rng)

    ipos = {f"i{k}": rng.uniform(-field, field, 2) for k in range(n_interceptors)}
    targets = [rng.uniform(-field, field, 2) for _ in range(n_targets)]
    for nid, p in ipos.items():
        net.set_position(nid, p)

    covered: set = set()
    for part in net.topology(0.0).partitions:
        taken: set = set()  # deconfliction is only possible inside a partition
        for nid in part:
            best, best_d = None, np.inf
            for j, tp in enumerate(targets):
                if j in taken:
                    continue
                d = float(np.linalg.norm(ipos[nid] - tp))
                if d < best_d:
                    best, best_d = j, d
            if best is not None:
                taken.add(best)
                covered.add(best)
    return len(covered) / n_targets


def run_sweep(radii: List[float], n_seeds: int = 30) -> Dict[float, float]:
    return {
        r: float(np.mean([partition_greedy_quality(r, seed=s) for s in range(n_seeds)]))
        for r in radii
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/figures/pc_degradation_curve.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    radii = [4000, 3000, 2000, 1500, 1000, 700, 500, 300, 100]
    results = run_sweep(radii)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = sorted(results)
    ax.plot(xs, [results[r] for r in xs], "o-", color="navy")
    ax.set_xlabel("comm radius (m)")
    ax.set_ylabel("target coverage (fraction)")
    ax.set_title("PC — allocation quality vs comms degradation (greedy, per-partition)")
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)

    for r in sorted(results, reverse=True):
        print(f"radius {r:>6.0f} m -> coverage {results[r]:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
