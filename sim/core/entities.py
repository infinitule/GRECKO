"""Plain-data entity definitions (ECS component style).

All entities are dataclasses — no behaviour lives here.
Systems in world.py and kinematics.py operate on these structs.
"""
from __future__ import annotations

import dataclasses
import numpy as np
from typing import List, Optional


@dataclasses.dataclass
class HostileUAS:
    id: str
    pos: np.ndarray          # shape (2,), metres
    vel: np.ndarray          # shape (2,), m/s
    heading: float           # radians
    speed: float             # m/s (constant magnitude)
    max_turn_rate: float     # rad/s
    weave_amplitude: float   # radians; 0 = straight ingress
    weave_period: float      # seconds per full weave cycle
    waypoints: List[np.ndarray]   # remaining ingress waypoints
    alive: bool = True
    t_spawned: float = 0.0


@dataclasses.dataclass
class Interceptor:
    id: str
    pos: np.ndarray          # shape (2,), metres
    vel: np.ndarray          # shape (2,), m/s
    heading: float           # radians
    speed: float             # m/s
    max_turn_rate: float     # rad/s
    endurance: float         # seconds of flight remaining
    effector_type: str       # e.g. "kinetic", "net", "ew"
    alive: bool = True
    t_spawned: float = 0.0


@dataclasses.dataclass
class Asset:
    id: str
    pos: np.ndarray          # shape (2,), metres (default origin)
    hp: float                # hit-points remaining
    value: float             # economic value; used by allocator
    alive: bool = True
