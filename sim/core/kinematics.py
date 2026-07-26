"""Turn-rate-limited constant-speed kinematics.

Pure functions: (entity, dt, rng) -> (new_pos, new_vel, new_heading).
Works for both HostileUAS and Interceptor; caller supplies the parameters.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from sim.core.vec import angle_of, clamp, normalise


def _shortest_turn(current: float, desired: float) -> float:
    """Signed angle delta (radians) on [-π, π] from current to desired heading."""
    delta = (desired - current + np.pi) % (2 * np.pi) - np.pi
    return delta


def step_entity(
    pos: np.ndarray,
    vel: np.ndarray,
    heading: float,
    speed: float,
    max_turn_rate: float,
    desired_heading: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Advance one fixed timestep.

    Returns (new_pos, new_vel, new_heading).
    Turn rate is clamped; speed magnitude is constant.
    """
    delta = _shortest_turn(heading, desired_heading)
    max_delta = max_turn_rate * dt
    actual_delta = clamp(delta, -max_delta, max_delta)
    new_heading = heading + actual_delta

    new_vel = np.array([np.cos(new_heading), np.sin(new_heading)]) * speed
    new_pos = pos + new_vel * dt
    return new_pos, new_vel, new_heading


def hostile_desired_heading(
    pos: np.ndarray,
    waypoints: list,
    heading: float,
    weave_amplitude: float,
    weave_period: float,
    t: float,
) -> tuple[float, list]:
    """Compute next desired heading for a HostileUAS.

    Returns (desired_heading_rad, remaining_waypoints).
    Pops waypoints when within arrival_radius.
    """
    arrival_radius = 20.0  # metres
    remaining = list(waypoints)

    while remaining:
        diff = remaining[0] - pos
        if np.linalg.norm(diff) < arrival_radius:
            remaining.pop(0)
            continue
        break

    if not remaining:
        # No waypoints left — continue on current heading (already past asset)
        base = heading
    else:
        diff = remaining[0] - pos
        base = float(np.arctan2(diff[1], diff[0]))

    # Sinusoidal weave overlay
    if weave_amplitude > 0 and weave_period > 0:
        weave = weave_amplitude * np.sin(2 * np.pi * t / weave_period)
        base += weave

    return base, remaining


def interceptor_desired_heading(
    pos: np.ndarray,
    target_pos: Optional[np.ndarray],
    target_vel: Optional[np.ndarray],
    speed: float,
    dt: float,
) -> float:
    """Pure pursuit guidance toward target (upgradeable to PN).

    If no target, return current heading (caller must supply it separately).
    """
    if target_pos is None:
        return 0.0
    diff = target_pos - pos
    return float(np.arctan2(diff[1], diff[0]))
