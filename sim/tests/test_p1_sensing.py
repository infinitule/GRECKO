"""P1 acceptance tests.

Criteria from the plan:
1. Empirical Pd matches the configured curve over 10k trials.
2. No report field exposes a true entity id (POSG no-truth-leakage).
3. False alarms occur at the configured Poisson rate.
4. FOV and max-range geometry respected.
5. Bearing-only sensors emit bearings, not positions.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from sim.sensing.sensors import (
    AcousticSensor,
    EOIRSensor,
    RadarSensor,
    SensorMesh,
    SensorNode,
    SensorReport,
)


# ---------------------------------------------------------------------------
# 1. Detection probability matches configuration
# ---------------------------------------------------------------------------

class TestDetectionProbability:
    def _trial_pd(self, sensor: SensorNode, r: float, n: int, seed: int) -> float:
        rng = np.random.default_rng(seed)
        target = sensor.pos + np.array([r, 0.0])
        detections = 0
        for _ in range(n):
            sensor._last_scan_t = -1e9   # force scan eligibility each trial
            reports, sidecar = sensor.scan(0.0, [target], rng)
            detections += sum(1 for s in sidecar if s == 0)
        return detections / n

    def test_pd_inside_knee_matches_pd_max(self):
        radar = RadarSensor("r0", np.array([0.0, 0.0]), false_alarm_rate=0.0)
        r = 0.5 * radar.max_range  # inside knee (knee at 0.6)
        emp = self._trial_pd(radar, r, 10_000, seed=1)
        assert abs(emp - radar.pd_max) < 0.02, f"empirical Pd {emp:.3f} vs configured {radar.pd_max}"

    def test_pd_falloff_region(self):
        radar = RadarSensor("r0", np.array([0.0, 0.0]), false_alarm_rate=0.0)
        r = 0.8 * radar.max_range  # halfway through falloff (knee 0.6 -> 1.0)
        expected = radar.detection_probability(r)
        emp = self._trial_pd(radar, r, 10_000, seed=2)
        assert abs(emp - expected) < 0.02, f"empirical Pd {emp:.3f} vs curve {expected:.3f}"

    def test_pd_zero_beyond_max_range(self):
        radar = RadarSensor("r0", np.array([0.0, 0.0]), false_alarm_rate=0.0)
        emp = self._trial_pd(radar, radar.max_range * 1.1, 1_000, seed=3)
        assert emp == 0.0


# ---------------------------------------------------------------------------
# 2. POSG no-truth-leakage invariant
# ---------------------------------------------------------------------------

class TestNoTruthLeakage:
    def test_report_has_no_entity_id_field(self):
        fields = {f.name for f in dataclasses.fields(SensorReport)}
        forbidden = {"entity_id", "true_id", "truth_id", "target_id", "true_pos", "true_position", "truth"}
        assert not (fields & forbidden), f"SensorReport leaks truth fields: {fields & forbidden}"

    def test_report_dict_has_no_truth_keys(self):
        rng = np.random.default_rng(0)
        radar = RadarSensor("r0", np.array([0.0, 0.0]), false_alarm_rate=0.0)
        reports, _ = radar.scan(0.0, [np.array([500.0, 0.0])], rng)
        assert reports, "expected at least one detection at close range over retries"
        d = reports[0].to_dict()
        flat_keys = set(d.keys()) | set(d["measurement"].keys())
        forbidden = {"entity_id", "true_id", "truth", "target_id"}
        assert not (flat_keys & forbidden)

    def test_measurement_differs_from_truth(self):
        """The reported position must be the noisy measurement, not the exact truth."""
        rng = np.random.default_rng(0)
        radar = RadarSensor("r0", np.array([0.0, 0.0]), false_alarm_rate=0.0)
        truth = np.array([500.0, 300.0])
        exact_matches = 0
        n = 0
        for _ in range(100):
            radar._last_scan_t = -1e9
            reports, sidecar = radar.scan(0.0, [truth], rng)
            for rep, s in zip(reports, sidecar):
                if s == 0:
                    n += 1
                    if np.allclose(rep.position, truth):
                        exact_matches += 1
        assert n > 50
        assert exact_matches == 0, "reports reproduce exact truth positions — noise not applied"

    def test_fusion_facing_api_returns_reports_only(self):
        rng = np.random.default_rng(0)
        mesh = SensorMesh([RadarSensor("r0", np.array([0.0, 0.0]))])
        out = mesh.scan_all(0.0, [np.array([400.0, 0.0])], rng)
        assert isinstance(out, list)
        for rep in out:
            assert isinstance(rep, SensorReport)


# ---------------------------------------------------------------------------
# 3. False alarms
# ---------------------------------------------------------------------------

class TestFalseAlarms:
    def test_clutter_rate_matches_poisson_mean(self):
        rng = np.random.default_rng(7)
        radar = RadarSensor("r0", np.array([0.0, 0.0]), false_alarm_rate=0.5)
        total_clutter = 0
        n_scans = 10_000
        for _ in range(n_scans):
            radar._last_scan_t = -1e9
            _, sidecar = radar.scan(0.0, [], rng)
            total_clutter += sum(1 for s in sidecar if s == -1)
        emp_rate = total_clutter / n_scans
        assert abs(emp_rate - 0.5) < 0.03, f"empirical clutter rate {emp_rate:.3f} vs configured 0.5"

    def test_clutter_within_sensor_coverage(self):
        rng = np.random.default_rng(8)
        radar = RadarSensor("r0", np.array([100.0, -50.0]), false_alarm_rate=2.0)
        for _ in range(500):
            radar._last_scan_t = -1e9
            reports, sidecar = radar.scan(0.0, [], rng)
            for rep, s in zip(reports, sidecar):
                if s == -1 and rep.kind == "cartesian":
                    r = np.linalg.norm(rep.position - radar.pos)
                    assert r <= radar.max_range + 1e-6


# ---------------------------------------------------------------------------
# 4. Geometry
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_fov_excludes_targets_behind(self):
        rng = np.random.default_rng(9)
        eoir = EOIRSensor("e0", np.array([0.0, 0.0]), fov_center=0.0, false_alarm_rate=0.0)
        behind = np.array([-500.0, 0.0])   # 180 deg off a 120-deg FOV
        for _ in range(200):
            eoir._last_scan_t = -1e9
            reports, _ = eoir.scan(0.0, [behind], rng)
            assert reports == []

    def test_update_rate_respected(self):
        rng = np.random.default_rng(10)
        radar = RadarSensor("r0", np.array([0.0, 0.0]), update_rate=2.0, false_alarm_rate=0.0, pd_max=1.0, pd_knee_frac=1.0)
        target = [np.array([100.0, 0.0])]
        scans_emitting = 0
        # 50 Hz ticks for 1 simulated second -> at 2 Hz expect 2 scans
        for i in range(50):
            t = i * 0.02
            reports, _ = radar.scan(t, target, rng)
            if reports:
                scans_emitting += 1
        assert scans_emitting == 2, f"expected 2 scans in 1 s at 2 Hz, got {scans_emitting}"

    def test_latency_applied_to_timestamp(self):
        rng = np.random.default_rng(11)
        radar = RadarSensor("r0", np.array([0.0, 0.0]), latency=0.1, false_alarm_rate=0.0, pd_max=1.0, pd_knee_frac=1.0)
        reports, _ = radar.scan(5.0, [np.array([100.0, 0.0])], rng)
        assert reports
        assert abs(reports[0].t - 5.1) < 1e-9


# ---------------------------------------------------------------------------
# 5. Bearing-only sensors
# ---------------------------------------------------------------------------

class TestBearingOnly:
    def test_acoustic_emits_bearing_not_position(self):
        rng = np.random.default_rng(12)
        ac = AcousticSensor("a0", np.array([0.0, 0.0]), false_alarm_rate=0.0, pd_max=1.0, pd_knee_frac=1.0)
        reports, _ = ac.scan(0.0, [np.array([200.0, 0.0])], rng)
        assert reports
        rep = reports[0]
        assert rep.kind == "bearing_only"
        assert rep.position is None
        assert rep.bearing is not None
        assert rep.covariance.shape == (1, 1)

    def test_bearing_roughly_correct(self):
        rng = np.random.default_rng(13)
        ac = AcousticSensor("a0", np.array([0.0, 0.0]), false_alarm_rate=0.0, pd_max=1.0, pd_knee_frac=1.0)
        target = np.array([0.0, 200.0])  # due north -> bearing pi/2
        bearings = []
        for _ in range(200):
            ac._last_scan_t = -1e9
            reports, _ = ac.scan(0.0, [target], rng)
            bearings.append(reports[0].bearing)
        mean_b = np.mean(bearings)
        assert abs(mean_b - math.pi / 2) < 0.05
