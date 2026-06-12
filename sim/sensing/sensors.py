"""Heterogeneous sensor mesh — imperfect observers of the true world.

POSG invariant (the architectural point of this module): SensorReport carries
NO ground-truth identity and no true state. Fusion sees only noisy measurements
plus covariances. The only place truth appears is the eval sidecar
(`truth_sidecar`), which fusion must never import.

Sensor types:
- RADAR    : long range, good range accuracy, weaker angular accuracy, poor ID.
- EO/IR    : short range, excellent angular accuracy; optional bearing-only mode.
- ACOUSTIC : very short range, noisy bearing-only, immune to RF conditions.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from sim.core.vec import norm


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SensorReport:
    """Fusion-facing measurement. Mirrors /proto/sensor_report.schema.json.

    Deliberately contains NO entity id, NO truth position, NO truth velocity.
    """
    t: float
    sensor_id: str
    sensor_type: str           # "RADAR" | "EOIR" | "ACOUSTIC"
    report_id: int             # per-sensor sequence number, not an identity
    kind: str                  # "cartesian" | "bearing_only"
    position: Optional[np.ndarray]   # shape (2,) if cartesian, else None
    bearing: Optional[float]         # radians from sensor, if bearing_only
    covariance: np.ndarray           # (2,2) cartesian, (1,1) bearing-only
    sensor_pos: np.ndarray           # sensor's own location (public knowledge)

    def to_dict(self) -> dict:
        m: dict = {"kind": self.kind}
        if self.kind == "cartesian":
            m["position"] = self.position.tolist()
        else:
            m["bearing"] = self.bearing
        return {
            "t": self.t,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "report_id": self.report_id,
            "measurement": m,
            "covariance": self.covariance.tolist(),
        }


# ---------------------------------------------------------------------------
# Sensor base
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SensorNode:
    """Base sensor. Subclasses set error characters via the dataclass fields.

    detection probability vs range: Pd(r) = pd_max for r <= r_knee, then
    linear falloff to 0 at max_range. Simple, configurable, testable.
    """
    sensor_id: str
    sensor_type: str
    pos: np.ndarray                 # shape (2,)
    fov_center: float               # radians; boresight direction
    fov_width: float                # radians; 2*pi = omnidirectional
    max_range: float                # metres
    pd_max: float                   # peak detection probability
    pd_knee_frac: float             # fraction of max_range where falloff starts
    noise_std: np.ndarray           # per-axis measurement sigma (native space)
    update_rate: float              # Hz
    latency: float                  # seconds added to report timestamp
    false_alarm_rate: float         # expected clutter reports per scan (Poisson)
    bearing_only: bool = False

    _next_report_id: int = dataclasses.field(default=0, repr=False)
    _last_scan_t: float = dataclasses.field(default=-1e9, repr=False)

    # -- geometry ----------------------------------------------------------

    def detection_probability(self, r: float) -> float:
        if r > self.max_range:
            return 0.0
        knee = self.pd_knee_frac * self.max_range
        if r <= knee:
            return self.pd_max
        # linear falloff knee -> max_range
        frac = (r - knee) / (self.max_range - knee)
        return self.pd_max * (1.0 - frac)

    def in_fov(self, target_pos: np.ndarray) -> bool:
        if self.fov_width >= 2 * math.pi - 1e-9:
            return True
        diff = target_pos - self.pos
        bearing = math.atan2(diff[1], diff[0])
        delta = (bearing - self.fov_center + math.pi) % (2 * math.pi) - math.pi
        return abs(delta) <= self.fov_width / 2

    def due_for_scan(self, t: float) -> bool:
        period = 1.0 / self.update_rate
        return (t - self._last_scan_t) >= period - 1e-9

    # -- scan --------------------------------------------------------------

    def scan(
        self,
        t: float,
        true_positions: List[np.ndarray],
        rng: np.random.Generator,
    ) -> Tuple[List[SensorReport], List[int]]:
        """One sensor scan.

        `true_positions` is the list of all live target positions; truth is
        consumed HERE and only here — the returned reports carry none of it.

        Returns (reports, truth_index_sidecar): the sidecar maps each report
        to the truth index that generated it (-1 = clutter). The sidecar is
        for EVAL ONLY; the fusion-facing API (SensorMesh.scan_all) discards it
        unless explicitly asked for the eval channel.
        """
        if not self.due_for_scan(t):
            return [], []
        self._last_scan_t = t

        reports: List[SensorReport] = []
        sidecar: List[int] = []
        report_t = t + self.latency

        for idx, p in enumerate(true_positions):
            r = norm(p - self.pos)
            if not self.in_fov(p):
                continue
            pd = self.detection_probability(r)
            if rng.random() >= pd:
                continue
            reports.append(self._measure(report_t, p, rng))
            sidecar.append(idx)

        # Poisson clutter
        n_fa = rng.poisson(self.false_alarm_rate)
        for _ in range(n_fa):
            reports.append(self._clutter(report_t, rng))
            sidecar.append(-1)

        return reports, sidecar

    def _measure(self, t: float, true_pos: np.ndarray, rng: np.random.Generator) -> SensorReport:
        rid = self._next_report_id
        self._next_report_id += 1

        if self.bearing_only:
            diff = true_pos - self.pos
            true_bearing = math.atan2(diff[1], diff[0])
            sigma = float(self.noise_std[0])
            meas_bearing = true_bearing + rng.normal(0.0, sigma)
            return SensorReport(
                t=t, sensor_id=self.sensor_id, sensor_type=self.sensor_type,
                report_id=rid, kind="bearing_only",
                position=None, bearing=float(meas_bearing),
                covariance=np.array([[sigma ** 2]]),
                sensor_pos=self.pos.copy(),
            )

        noise = rng.normal(0.0, 1.0, size=2) * self.noise_std[:2]
        meas = true_pos + noise
        cov = np.diag(self.noise_std[:2] ** 2)
        return SensorReport(
            t=t, sensor_id=self.sensor_id, sensor_type=self.sensor_type,
            report_id=rid, kind="cartesian",
            position=meas, bearing=None,
            covariance=cov,
            sensor_pos=self.pos.copy(),
        )

    def _clutter(self, t: float, rng: np.random.Generator) -> SensorReport:
        rid = self._next_report_id
        self._next_report_id += 1

        if self.bearing_only:
            sigma = float(self.noise_std[0])
            bearing = rng.uniform(-math.pi, math.pi)
            return SensorReport(
                t=t, sensor_id=self.sensor_id, sensor_type=self.sensor_type,
                report_id=rid, kind="bearing_only",
                position=None, bearing=float(bearing),
                covariance=np.array([[sigma ** 2]]),
                sensor_pos=self.pos.copy(),
            )

        # uniform clutter inside the sensor's coverage disc
        r = self.max_range * math.sqrt(rng.random())
        theta = rng.uniform(-math.pi, math.pi)
        pos = self.pos + np.array([r * math.cos(theta), r * math.sin(theta)])
        cov = np.diag(self.noise_std[:2] ** 2)
        return SensorReport(
            t=t, sensor_id=self.sensor_id, sensor_type=self.sensor_type,
            report_id=rid, kind="cartesian",
            position=pos, bearing=None,
            covariance=cov,
            sensor_pos=self.pos.copy(),
        )


# ---------------------------------------------------------------------------
# Concrete sensor types
# ---------------------------------------------------------------------------

def RadarSensor(sensor_id: str, pos: np.ndarray, **over) -> SensorNode:
    """Long range, good range accuracy, weaker angular accuracy, clutter-prone."""
    cfg = dict(
        sensor_type="RADAR",
        fov_center=0.0, fov_width=2 * math.pi,
        max_range=3000.0, pd_max=0.9, pd_knee_frac=0.6,
        noise_std=np.array([25.0, 25.0]),
        update_rate=2.0, latency=0.1,
        false_alarm_rate=0.5,
        bearing_only=False,
    )
    cfg.update(over)
    return SensorNode(sensor_id=sensor_id, pos=pos, **cfg)


def EOIRSensor(sensor_id: str, pos: np.ndarray, bearing_only: bool = False, **over) -> SensorNode:
    """Short range, excellent angular accuracy; bearing-only mode available."""
    cfg = dict(
        sensor_type="EOIR",
        fov_center=0.0, fov_width=math.radians(120.0),
        max_range=1200.0, pd_max=0.95, pd_knee_frac=0.7,
        noise_std=np.array([0.005, 0.005]) if bearing_only else np.array([8.0, 8.0]),
        update_rate=10.0, latency=0.05,
        false_alarm_rate=0.1,
        bearing_only=bearing_only,
    )
    cfg.update(over)
    return SensorNode(sensor_id=sensor_id, pos=pos, **cfg)


def AcousticSensor(sensor_id: str, pos: np.ndarray, **over) -> SensorNode:
    """Very short range, noisy bearing-only, RF-immune (works under jamming)."""
    cfg = dict(
        sensor_type="ACOUSTIC",
        fov_center=0.0, fov_width=2 * math.pi,
        max_range=400.0, pd_max=0.8, pd_knee_frac=0.4,
        noise_std=np.array([0.15]),
        update_rate=4.0, latency=0.3,
        false_alarm_rate=0.3,
        bearing_only=True,
    )
    cfg.update(over)
    return SensorNode(sensor_id=sensor_id, pos=pos, **cfg)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

class SensorMesh:
    """The fusion-facing API. Exposes only SensorReports — never truth.

    scan_all() returns the reports. scan_all_with_truth_sidecar() additionally
    returns the eval sidecar; it exists for tests/eval and must never be called
    from /sim/fusion (CI greps for this).
    """

    def __init__(self, sensors: List[SensorNode]):
        self.sensors = sensors

    def scan_all(
        self,
        t: float,
        true_positions: List[np.ndarray],
        rng: np.random.Generator,
    ) -> List[SensorReport]:
        reports, _ = self.scan_all_with_truth_sidecar(t, true_positions, rng)
        return reports

    def scan_all_with_truth_sidecar(
        self,
        t: float,
        true_positions: List[np.ndarray],
        rng: np.random.Generator,
    ) -> Tuple[List[SensorReport], List[int]]:
        all_reports: List[SensorReport] = []
        all_sidecar: List[int] = []
        for s in self.sensors:
            reports, sidecar = s.scan(t, true_positions, rng)
            all_reports.extend(reports)
            all_sidecar.extend(sidecar)
        return all_reports, all_sidecar
