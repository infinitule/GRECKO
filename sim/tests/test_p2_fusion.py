"""P2 acceptance tests.

Criteria from the plan:
1. Single-target RMS position error below threshold.
2. Track ID stability under a crossing-target scenario.
3. Clutter does not create confirmed tracks at baseline false-alarm rates.
4. 40-target scenario maintains >90% track purity at real-time rate.
5. POSG: fusion never imports the truth sidecar (grep invariant).
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from sim.fusion.benchmark import run_cell
from sim.fusion.tracker import Tracker
from sim.sensing.sensors import RadarSensor, SensorMesh


# ---------------------------------------------------------------------------
# 1. Single-target accuracy
# ---------------------------------------------------------------------------

class TestSingleTargetAccuracy:
    def test_rms_position_error_below_threshold(self):
        """Track a single constant-velocity target; converged RMS error must
        beat the raw measurement noise (25 m) by a clear margin."""
        rng = np.random.default_rng(1)
        radar = RadarSensor(
            "r0", np.array([0.0, 0.0]),
            update_rate=10.0, false_alarm_rate=0.0, pd_max=1.0, pd_knee_frac=1.0,
            latency=0.0,
        )
        mesh = SensorMesh([radar])
        tracker = Tracker()

        pos = np.array([1500.0, 500.0])
        vel = np.array([-20.0, -8.0])
        dt = 0.02
        errors = []
        t = 0.0
        for i in range(int(30.0 / dt)):
            pos = pos + vel * dt
            reports = mesh.scan_all(t, [pos], rng)
            tracks = tracker.update(t, reports)
            if t > 5.0:  # after convergence
                confirmed = [trk for trk in tracks if trk.status == "confirmed"]
                if confirmed:
                    est = confirmed[0].state[:2]
                    errors.append(np.linalg.norm(est - pos))
            t += dt

        assert errors, "no confirmed track ever produced"
        rms = float(np.sqrt(np.mean(np.square(errors))))
        assert rms < 15.0, f"RMS position error {rms:.1f} m exceeds 15 m threshold"

    def test_velocity_estimated(self):
        """Filter must converge to the true velocity within tolerance."""
        rng = np.random.default_rng(2)
        radar = RadarSensor(
            "r0", np.array([0.0, 0.0]),
            update_rate=10.0, false_alarm_rate=0.0, pd_max=1.0, pd_knee_frac=1.0,
            latency=0.0,
        )
        mesh = SensorMesh([radar])
        tracker = Tracker()

        pos = np.array([1000.0, 0.0])
        vel = np.array([-25.0, 5.0])
        dt = 0.02
        t = 0.0
        last_vel_est = None
        for _ in range(int(20.0 / dt)):
            pos = pos + vel * dt
            reports = mesh.scan_all(t, [pos], rng)
            tracks = tracker.update(t, reports)
            confirmed = [trk for trk in tracks if trk.status == "confirmed"]
            if confirmed:
                last_vel_est = confirmed[0].state[2:4]
            t += dt

        assert last_vel_est is not None
        err = np.linalg.norm(last_vel_est - vel)
        assert err < 5.0, f"velocity error {err:.1f} m/s exceeds 5 m/s"


# ---------------------------------------------------------------------------
# 2. Crossing targets — ID stability
# ---------------------------------------------------------------------------

class TestCrossingTargets:
    def test_track_ids_stable_through_crossing(self):
        """Two targets cross paths; each confirmed track's dominant truth
        association before the crossing must match after the crossing."""
        rng = np.random.default_rng(3)
        radar = RadarSensor(
            "r0", np.array([0.0, -2000.0]),
            update_rate=10.0, false_alarm_rate=0.0, pd_max=1.0, pd_knee_frac=1.0,
            noise_std=np.array([10.0, 10.0]), latency=0.0,
        )
        mesh = SensorMesh([radar])
        tracker = Tracker()

        # Crossing at origin around t=10: distinct velocities -> separable in state space
        targets = [
            {"pos": np.array([-400.0, 200.0]), "vel": np.array([40.0, -20.0])},
            {"pos": np.array([400.0, 200.0]), "vel": np.array([-40.0, -20.0])},
        ]
        dt = 0.02
        t = 0.0
        report_truth = {}
        pre_assoc: dict = {}
        post_assoc: dict = {}

        for i in range(int(20.0 / dt)):
            for tgt in targets:
                tgt["pos"] = tgt["pos"] + tgt["vel"] * dt
            positions = [tgt["pos"] for tgt in targets]
            reports, sidecar = mesh.scan_all_with_truth_sidecar(t, positions, rng)
            for rep, ti in zip(reports, sidecar):
                report_truth[(rep.sensor_id, rep.report_id)] = ti
            tracker.update(t, reports)

            phase = "pre" if t < 8.0 else ("post" if t > 12.0 else None)
            if phase:
                for trk in tracker.confirmed_tracks():
                    truth_ids = [report_truth.get(k, -2) for k in trk.assoc_keys if report_truth.get(k, -2) >= 0]
                    if truth_ids:
                        dominant = max(set(truth_ids), key=truth_ids.count)
                        (pre_assoc if phase == "pre" else post_assoc)[trk.track_id] = dominant
            t += dt

        # every track alive in both phases must keep its dominant truth id
        common = set(pre_assoc) & set(post_assoc)
        assert len(common) >= 2, f"expected 2 persistent tracks, got {len(common)}"
        for tid in common:
            assert pre_assoc[tid] == post_assoc[tid], (
                f"track {tid} swapped identity through the crossing: "
                f"{pre_assoc[tid]} -> {post_assoc[tid]}"
            )


# ---------------------------------------------------------------------------
# 3. Clutter rejection
# ---------------------------------------------------------------------------

class TestClutterRejection:
    def test_clutter_does_not_confirm_tracks(self):
        """With no real targets and baseline clutter, no confirmed tracks emerge."""
        rng = np.random.default_rng(4)
        radar = RadarSensor("r0", np.array([0.0, 0.0]), false_alarm_rate=0.5, update_rate=4.0)
        mesh = SensorMesh([radar])
        tracker = Tracker()

        dt = 0.02
        t = 0.0
        max_confirmed = 0
        for _ in range(int(30.0 / dt)):
            reports = mesh.scan_all(t, [], rng)
            tracker.update(t, reports)
            max_confirmed = max(max_confirmed, len(tracker.confirmed_tracks()))
            t += dt

        assert max_confirmed == 0, f"clutter created {max_confirmed} confirmed tracks"


# ---------------------------------------------------------------------------
# 4. Saturation — the headline P2 criterion
# ---------------------------------------------------------------------------

class TestSaturation:
    def test_40_targets_purity_above_90_at_realtime(self):
        r = run_cell(40, duration=20.0, seed=42)
        assert r["n_confirmed"] >= 36, f"only {r['n_confirmed']}/40 targets confirmed"
        assert r["mean_purity"] > 0.90, f"purity {r['mean_purity']:.3f} <= 0.90"
        assert r["realtime_factor"] > 1.0, (
            f"tracker slower than real time: {r['realtime_factor']:.2f}x"
        )


# ---------------------------------------------------------------------------
# 5. POSG isolation — grep invariant
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_fusion_never_touches_truth_sidecar(self):
        """No file in /sim/fusion may reference the truth-sidecar API except
        benchmark.py, which is the sanctioned eval channel."""
        fusion_dir = pathlib.Path(__file__).parent.parent / "fusion"
        for py in fusion_dir.glob("*.py"):
            if py.name == "benchmark.py":
                continue
            text = py.read_text()
            assert "scan_all_with_truth_sidecar" not in text, (
                f"{py.name} references the truth sidecar — POSG violation"
            )
            assert "truth" not in text.lower() or py.name == "tracker.py", (
                f"{py.name} mentions truth — review for leakage"
            )
