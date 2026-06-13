"""P2 saturation benchmark: sweep 10 -> 100 simultaneous targets.

Reports association runtime and track purity per cell. Purity uses the
truth sidecar — this file is part of the EVAL channel, not the fusion path
(fusion itself never sees the sidecar).

Usage: python -m sim.fusion.benchmark
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import numpy as np

from sim.fusion.tracker import Tracker
from sim.sensing.sensors import RadarSensor, SensorMesh


def make_ring_targets(n: int, radius: float, speed: float, rng: np.random.Generator):
    """n inbound targets evenly spaced on a ring, heading to origin."""
    targets = []
    for i in range(n):
        theta = 2 * math.pi * i / n + rng.uniform(-0.05, 0.05)
        pos = np.array([radius * math.cos(theta), radius * math.sin(theta)])
        vel = -pos / np.linalg.norm(pos) * speed
        targets.append({"pos": pos, "vel": vel})
    return targets


def run_cell(
    n_targets: int,
    duration: float = 20.0,
    dt: float = 0.02,
    seed: int = 42,
) -> Dict:
    rng = np.random.default_rng(seed)
    targets = make_ring_targets(n_targets, radius=2000.0, speed=20.0, rng=rng)

    mesh = SensorMesh([
        RadarSensor("radar_0", np.array([0.0, 0.0]), update_rate=4.0),
        RadarSensor("radar_1", np.array([500.0, 500.0]), update_rate=4.0),
    ])
    tracker = Tracker()

    # map (sensor_id, report_id) -> truth index, built from the eval sidecar
    report_truth: Dict[Tuple[str, int], int] = {}

    t = 0.0
    tracker_cpu = 0.0
    n_ticks = int(duration / dt)
    for _ in range(n_ticks):
        for tgt in targets:
            tgt["pos"] = tgt["pos"] + tgt["vel"] * dt
        positions = [tgt["pos"] for tgt in targets]
        reports, sidecar = mesh.scan_all_with_truth_sidecar(t, positions, rng)
        for rep, truth_idx in zip(reports, sidecar):
            report_truth[(rep.sensor_id, rep.report_id)] = truth_idx

        t0 = time.perf_counter()
        tracker.update(t, reports)
        tracker_cpu += time.perf_counter() - t0
        t += dt

    # purity: fraction of each confirmed track's associations from its dominant truth id
    purities: List[float] = []
    for trk in tracker.confirmed_tracks():
        truth_ids = [report_truth.get(k, -2) for k in trk.assoc_keys]
        truth_ids = [tid for tid in truth_ids if tid >= 0]  # ignore clutter assoc
        if not truth_ids:
            continue
        dominant = max(set(truth_ids), key=truth_ids.count)
        purities.append(truth_ids.count(dominant) / len(truth_ids))

    return {
        "n_targets": n_targets,
        "n_confirmed": len(tracker.confirmed_tracks()),
        "mean_purity": float(np.mean(purities)) if purities else 0.0,
        "tracker_cpu_s": tracker_cpu,
        "sim_duration_s": duration,
        "realtime_factor": duration / tracker_cpu if tracker_cpu > 0 else float("inf"),
    }


def main() -> None:
    print(f"{'targets':>8} {'confirmed':>10} {'purity':>8} {'cpu (s)':>9} {'RT factor':>10}")
    for n in [10, 20, 40, 60, 80, 100]:
        r = run_cell(n)
        print(
            f"{r['n_targets']:>8} {r['n_confirmed']:>10} {r['mean_purity']:>8.3f} "
            f"{r['tracker_cpu_s']:>9.2f} {r['realtime_factor']:>10.1f}x"
        )


if __name__ == "__main__":
    main()
