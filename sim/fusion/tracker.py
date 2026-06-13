"""Multi-target tracker: SensorReports in, Track messages out.

Track lifecycle: tentative -> confirmed (M-of-N) -> coasted -> dropped.
The tracker consumes ONLY SensorReports (POSG invariant — it never imports
the truth sidecar). Track ids are fusion-assigned and unrelated to truth ids.

Scan-aligned bookkeeping: the tracker is ticked at sim rate (50 Hz) but
sensors report at 2-10 Hz. Hits/misses are counted only on cycles that carry
reports — otherwise empty ticks would starve every track before confirmation.
Association runs per sensor group so simultaneous reports of one target from
two sensors update one track instead of spawning a duplicate.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from sim.fusion.associator import Associator, GNNAssociator
from sim.fusion.kalman import KalmanFilterCV
from sim.sensing.sensors import SensorReport


@dataclasses.dataclass
class TrackMessage:
    """Mirrors /proto/track.schema.json."""
    t: float
    track_id: str
    status: str            # tentative | confirmed | coasted
    state: np.ndarray      # [x, y, vx, vy]
    covariance: np.ndarray
    quality: float
    n_updates: int
    age: float

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "track_id": self.track_id,
            "status": self.status,
            "state": self.state.tolist(),
            "covariance": self.covariance.tolist(),
            "quality": self.quality,
            "n_updates": self.n_updates,
            "age": self.age,
        }


@dataclasses.dataclass
class InternalTrack:
    track_id: str
    kf: KalmanFilterCV
    status: str
    t_created: float
    t_last_update: float
    hit_history: List[bool] = dataclasses.field(default_factory=list)  # per report-carrying cycle
    n_updates: int = 0
    assoc_keys: List[tuple] = dataclasses.field(default_factory=list)  # (sensor_id, report_id) — EVAL purity only

    def quality(self) -> float:
        if not self.hit_history:
            return 0.0
        window = self.hit_history[-10:]
        return sum(window) / len(window)


class Tracker:
    def __init__(
        self,
        associator: Optional[Associator] = None,
        q_process: float = 4.0,         # white-accel intensity (m^2/s^3)
        confirm_m: int = 3,
        confirm_n: int = 5,
        coast_time: float = 1.5,        # s without update before a confirmed track coasts
        drop_coast_time: float = 6.0,   # s without update before a coasted track drops
        init_pos_var: float = 50.0 ** 2,
        init_vel_var: float = 40.0 ** 2,
    ):
        self.associator = associator or GNNAssociator()
        self.q_process = q_process
        self.confirm_m = confirm_m
        self.confirm_n = confirm_n
        self.coast_time = coast_time
        self.drop_coast_time = drop_coast_time
        self.init_pos_var = init_pos_var
        self.init_vel_var = init_vel_var

        self.tracks: List[InternalTrack] = []
        self._next_id = 0
        self._last_t: Optional[float] = None

    # ------------------------------------------------------------------

    def update(self, t: float, reports: List[SensorReport]) -> List[TrackMessage]:
        """One fusion cycle: predict to t; if reports arrived, associate per
        sensor group, update, and run lifecycle bookkeeping."""
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        if dt > 0:
            for trk in self.tracks:
                trk.kf.predict(dt)

        if not reports:
            self._coast_and_drop(t)
            return self.emit(t)

        by_sensor: Dict[str, List[SensorReport]] = defaultdict(list)
        for rep in reports:
            by_sensor[rep.sensor_id].append(rep)

        hit_this_cycle: set = set()
        new_tracks: List[InternalTrack] = []

        def _associate_pass(candidates, group, t):
            """Run one association pass; returns indices of unused reports."""
            assignments, unassociated = self.associator.associate(candidates, group)
            for ti, ri in assignments.items():
                trk = candidates[ti]
                rep = group[ri]
                if rep.kind == "cartesian":
                    trk.kf.update_cart(rep.position, rep.covariance)
                else:
                    trk.kf.update_bearing(rep.bearing, rep.covariance, rep.sensor_pos)
                trk.n_updates += 1
                trk.t_last_update = t
                trk.assoc_keys.append((rep.sensor_id, rep.report_id))
                hit_this_cycle.add(id(trk))
            return unassociated

        for sensor_id in sorted(by_sensor):  # sorted: determinism
            group = by_sensor[sensor_id]
            # Two-pass association: established tracks claim reports first, so
            # an outlier-spawned tentative duplicate cannot steal measurements
            # from the real track — it starves and is dropped by M-of-N.
            established = [trk for trk in self.tracks if trk.status != "tentative"]
            leftover_idx = _associate_pass(established, group, t)
            leftovers = [group[i] for i in leftover_idx]

            tentative = [trk for trk in self.tracks if trk.status == "tentative"] + new_tracks
            unassociated = _associate_pass(tentative, leftovers, t)
            unassoc_reports = [leftovers[i] for i in unassociated]

            # Initiate tentative tracks from unassociated CARTESIAN reports.
            # Bearing-only reports cannot initialise a position estimate alone
            # (design decision, ADR-002) — they only update existing tracks.
            for rep in unassoc_reports:
                if rep.kind != "cartesian":
                    continue
                x0 = np.array([rep.position[0], rep.position[1], 0.0, 0.0])
                P0 = np.diag([
                    max(self.init_pos_var, rep.covariance[0, 0]),
                    max(self.init_pos_var, rep.covariance[1, 1]),
                    self.init_vel_var, self.init_vel_var,
                ])
                trk = InternalTrack(
                    track_id=f"T{self._next_id:04d}",
                    kf=KalmanFilterCV(x0, P0, self.q_process),
                    status="tentative",
                    t_created=t,
                    t_last_update=t,
                    n_updates=1,
                    assoc_keys=[(rep.sensor_id, rep.report_id)],
                )
                self._next_id += 1
                new_tracks.append(trk)
                hit_this_cycle.add(id(trk))

        self.tracks.extend(new_tracks)

        # hit/miss bookkeeping — once per report-carrying cycle
        for trk in self.tracks:
            trk.hit_history.append(id(trk) in hit_this_cycle)

        self._lifecycle(t)
        return self.emit(t)

    # ------------------------------------------------------------------

    def _lifecycle(self, t: float) -> None:
        survivors: List[InternalTrack] = []
        for trk in self.tracks:
            window = trk.hit_history[-self.confirm_n:]

            if trk.status == "tentative":
                if sum(window) >= self.confirm_m:
                    trk.status = "confirmed"
                elif len(window) >= self.confirm_n and sum(window) == 0:
                    continue  # drop: all-miss window, never confirmed
            if trk.status == "confirmed" and (t - trk.t_last_update) > self.coast_time:
                trk.status = "coasted"
            if trk.status == "coasted":
                if trk.hit_history and trk.hit_history[-1]:
                    trk.status = "confirmed"
                elif (t - trk.t_last_update) > self.drop_coast_time:
                    continue  # drop
            survivors.append(trk)
        self.tracks = survivors

    def _coast_and_drop(self, t: float) -> None:
        """Time-based transitions on report-free ticks (no hit/miss appended)."""
        survivors: List[InternalTrack] = []
        for trk in self.tracks:
            if trk.status == "confirmed" and (t - trk.t_last_update) > self.coast_time:
                trk.status = "coasted"
            if trk.status == "coasted" and (t - trk.t_last_update) > self.drop_coast_time:
                continue
            survivors.append(trk)
        self.tracks = survivors

    # ------------------------------------------------------------------

    def emit(self, t: float) -> List[TrackMessage]:
        return [
            TrackMessage(
                t=t,
                track_id=trk.track_id,
                status=trk.status,
                state=trk.kf.x.copy(),
                covariance=trk.kf.P.copy(),
                quality=trk.quality(),
                n_updates=trk.n_updates,
                age=t - trk.t_created,
            )
            for trk in self.tracks
        ]

    def confirmed_tracks(self) -> List[InternalTrack]:
        return [trk for trk in self.tracks if trk.status == "confirmed"]
