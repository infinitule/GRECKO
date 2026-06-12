"""Feature extraction from track history.

Consumes TrackMessage objects (the /proto Track bus) — never truth.
Outputs a plain FeatureVector dataclass that the classifier consumes.
"""
from __future__ import annotations

import dataclasses
import math
from typing import List

import numpy as np

from sim.fusion.tracker import TrackMessage


@dataclasses.dataclass
class FeatureVector:
    track_id: str
    t: float
    speed: float              # m/s, current estimate
    heading_to_asset: float   # radians; 0 = pointing straight at asset
    approach_rate: float      # m/s, positive = closing on asset
    weave_energy: float       # variance of recent turn-rate samples (rad^2/s^2)
    altitude_band: int        # stub for 3-D: always 0 in 2-D sim
    rf_emitter: bool          # True if any RADAR report drove this track (from sensor_type)
    track_age: float          # seconds
    n_updates: int


def extract(
    track: TrackMessage,
    history: List[TrackMessage],
    asset_pos: np.ndarray,
    sensor_type_hint: str = "",
) -> FeatureVector:
    """Extract features from the current track state and its recent history.

    `history` should be the last N TrackMessages for this track (oldest first).
    `sensor_type_hint` carries the dominant sensor type that confirmed this track
    (set by the classifier pipeline; not a truth field).
    """
    x, y, vx, vy = track.state[:4]
    pos = np.array([x, y])
    vel = np.array([vx, vy])
    speed = float(np.linalg.norm(vel))

    # heading_to_asset: angle between current velocity and direction to asset
    diff = asset_pos - pos
    asset_dist = float(np.linalg.norm(diff))
    if asset_dist < 1e-6 or speed < 1e-6:
        heading_to_asset = 0.0
        approach_rate = speed
    else:
        asset_dir = diff / asset_dist
        cos_a = float(np.clip(np.dot(vel / speed, asset_dir), -1.0, 1.0))
        heading_to_asset = math.acos(cos_a)
        approach_rate = speed * cos_a

    # weave_energy: variance of per-tick turn-rate over recent history
    weave_energy = _weave_energy(history)

    return FeatureVector(
        track_id=track.track_id,
        t=track.t,
        speed=speed,
        heading_to_asset=heading_to_asset,
        approach_rate=approach_rate,
        weave_energy=weave_energy,
        altitude_band=0,
        rf_emitter=(sensor_type_hint == "RADAR"),
        track_age=track.age,
        n_updates=track.n_updates,
    )


def _weave_energy(history: List[TrackMessage]) -> float:
    """Variance of successive heading changes across the history window."""
    if len(history) < 3:
        return 0.0
    headings = []
    for msg in history:
        vx, vy = msg.state[2], msg.state[3]
        if abs(vx) + abs(vy) > 1e-6:
            headings.append(math.atan2(vy, vx))
    if len(headings) < 3:
        return 0.0
    deltas = []
    for i in range(1, len(headings)):
        d = (headings[i] - headings[i - 1] + math.pi) % (2 * math.pi) - math.pi
        deltas.append(d)
    return float(np.var(deltas))
