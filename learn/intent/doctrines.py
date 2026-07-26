"""Scripted swarm doctrines — the bootstrap training data for Pillar B.

Each generator produces a Scenario: agent trajectories sampled at SAMPLE_DT,
with a ground-truth intent label per agent. Doctrines: frontal saturation,
feint+main-axis, pincer, ISR loiter, leader-follower (per the plan).

The league (Pillar C) will later export novel doctrines in this same format
to retrain the model — keep the Scenario container stable.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List

import numpy as np

SAMPLE_DT = 0.5          # s between trajectory samples
DURATION = 60.0          # s per scenario
N_STEPS = int(DURATION / SAMPLE_DT)

INTENT_CLASSES = ["main_axis", "feint", "screen", "isr", "reserve"]


@dataclasses.dataclass
class Scenario:
    doctrine: str
    # trajectories[agent_idx] -> (N_STEPS, 4) array of [x, y, vx, vy]
    trajectories: np.ndarray         # (n_agents, N_STEPS, 4)
    labels: List[str]                # per-agent intent label
    asset_pos: np.ndarray

    @property
    def n_agents(self) -> int:
        return self.trajectories.shape[0]


# ---------------------------------------------------------------------------
# Group trajectory primitives
# ---------------------------------------------------------------------------

def _ingress_group(
    rng, n: int, axis_angle: float, r0: float, speed: float,
    spacing: float, t_turn_away: float = None,
) -> np.ndarray:
    """Group ingressing toward origin from bearing axis_angle at range r0.
    If t_turn_away is set, the group breaks off at that time and orbits out.
    Returns (n, N_STEPS, 4)."""
    out = np.zeros((n, N_STEPS, 4))
    base = np.array([math.cos(axis_angle), math.sin(axis_angle)]) * r0
    for i in range(n):
        offset = rng.normal(0.0, spacing, 2)
        pos = base + offset
        # aim at origin with small noise
        for k in range(N_STEPS):
            t = k * SAMPLE_DT
            if t_turn_away is not None and t >= t_turn_away:
                # turn away: head tangentially (orbit out)
                radial = pos / max(np.linalg.norm(pos), 1.0)
                tangent = np.array([-radial[1], radial[0]])
                vel = (tangent * 0.7 + radial * 0.7) * speed / math.sqrt(2)
            else:
                aim = -pos / max(np.linalg.norm(pos), 1.0)
                jitter = rng.normal(0.0, 0.03, 2)
                vel = (aim + jitter) * speed
            out[i, k, :2] = pos
            out[i, k, 2:] = vel
            pos = pos + vel * SAMPLE_DT
    return out


def _orbit_group(rng, n: int, axis_angle: float, r0: float, speed: float) -> np.ndarray:
    """ISR loiter: orbit at standoff range."""
    out = np.zeros((n, N_STEPS, 4))
    for i in range(n):
        phase = rng.uniform(0, 2 * math.pi)
        radius = r0 + rng.normal(0, 50)
        omega = speed / radius
        for k in range(N_STEPS):
            t = k * SAMPLE_DT
            a = axis_angle + phase + omega * t
            out[i, k, 0] = radius * math.cos(a)
            out[i, k, 1] = radius * math.sin(a)
            out[i, k, 2] = -radius * omega * math.sin(a)
            out[i, k, 3] = radius * omega * math.cos(a)
    return out


def _hold_group(rng, n: int, axis_angle: float, r0: float, drift_speed: float) -> np.ndarray:
    """Reserve: hold at long range with slow inbound drift."""
    out = np.zeros((n, N_STEPS, 4))
    base = np.array([math.cos(axis_angle), math.sin(axis_angle)]) * r0
    for i in range(n):
        pos = base + rng.normal(0, 200, 2)
        for k in range(N_STEPS):
            aim = -pos / max(np.linalg.norm(pos), 1.0)
            vel = aim * drift_speed + rng.normal(0, 0.5, 2)
            out[i, k, :2] = pos
            out[i, k, 2:] = vel
            pos = pos + vel * SAMPLE_DT
    return out


def _screen_group(rng, n: int, axis_angle: float, r0: float, speed: float) -> np.ndarray:
    """Screen: lateral weaving ahead of the main axis."""
    out = np.zeros((n, N_STEPS, 4))
    base = np.array([math.cos(axis_angle), math.sin(axis_angle)]) * r0
    for i in range(n):
        pos = base + rng.normal(0, 100, 2)
        for k in range(N_STEPS):
            t = k * SAMPLE_DT
            aim = -pos / max(np.linalg.norm(pos), 1.0)
            tangent = np.array([-aim[1], aim[0]])
            weave = math.sin(2 * math.pi * t / 8.0 + i)
            vel = (aim * 0.5 + tangent * weave) * speed
            out[i, k, :2] = pos
            out[i, k, 2:] = vel
            pos = pos + vel * SAMPLE_DT
    return out


# ---------------------------------------------------------------------------
# Doctrines
# ---------------------------------------------------------------------------

def frontal_saturation(rng) -> Scenario:
    n = rng.integers(8, 14)
    angle = rng.uniform(0, 2 * math.pi)
    traj = _ingress_group(rng, n, angle, rng.uniform(2500, 3000),
                          rng.uniform(22, 28), 60.0)
    return Scenario("frontal_saturation", traj, ["main_axis"] * n, np.zeros(2))


def feint_main_axis(rng) -> Scenario:
    """The doctrine that matters: a small loud feint + the real thrust.

    Signature differences the model can learn (and a heading-only heuristic
    cannot): the feint is smaller, slower, looser; it overcommits its heading
    (perfectly direct) until t_turn, then breaks off. The main axis is larger,
    faster, tighter.
    """
    angle_f = rng.uniform(0, 2 * math.pi)
    angle_m = angle_f + rng.uniform(math.pi / 2, 3 * math.pi / 2)
    n_f = int(rng.integers(3, 6))
    n_m = int(rng.integers(7, 11))
    t_turn = rng.uniform(20.0, 30.0)
    feint = _ingress_group(rng, n_f, angle_f, rng.uniform(2200, 2700),
                           rng.uniform(11, 15), 150.0, t_turn_away=t_turn)
    main = _ingress_group(rng, n_m, angle_m, rng.uniform(2500, 3000),
                          rng.uniform(23, 28), 50.0)
    traj = np.concatenate([feint, main], axis=0)
    labels = ["feint"] * n_f + ["main_axis"] * n_m
    return Scenario("feint_main_axis", traj, labels, np.zeros(2))


def pincer(rng) -> Scenario:
    angle = rng.uniform(0, 2 * math.pi)
    n1, n2 = int(rng.integers(5, 8)), int(rng.integers(5, 8))
    g1 = _ingress_group(rng, n1, angle, rng.uniform(2500, 2900), rng.uniform(22, 27), 60.0)
    g2 = _ingress_group(rng, n2, angle + math.pi, rng.uniform(2500, 2900), rng.uniform(22, 27), 60.0)
    traj = np.concatenate([g1, g2], axis=0)
    return Scenario("pincer", traj, ["main_axis"] * (n1 + n2), np.zeros(2))


def isr_loiter(rng) -> Scenario:
    angle = rng.uniform(0, 2 * math.pi)
    n_isr = int(rng.integers(2, 5))
    n_res = int(rng.integers(3, 6))
    isr = _orbit_group(rng, n_isr, angle, rng.uniform(1400, 1800), rng.uniform(15, 20))
    res = _hold_group(rng, n_res, angle + rng.uniform(-1, 1), rng.uniform(3200, 3800),
                      rng.uniform(1, 3))
    traj = np.concatenate([isr, res], axis=0)
    labels = ["isr"] * n_isr + ["reserve"] * n_res
    return Scenario("isr_loiter", traj, labels, np.zeros(2))


def leader_follower(rng) -> Scenario:
    angle = rng.uniform(0, 2 * math.pi)
    n_m = int(rng.integers(6, 10))
    n_s = int(rng.integers(2, 5))
    main = _ingress_group(rng, n_m, angle, rng.uniform(2600, 3000), rng.uniform(22, 26), 50.0)
    screen = _screen_group(rng, n_s, angle, rng.uniform(2000, 2300), rng.uniform(20, 25))
    traj = np.concatenate([main, screen], axis=0)
    labels = ["main_axis"] * n_m + ["screen"] * n_s
    return Scenario("leader_follower", traj, labels, np.zeros(2))


DOCTRINES = {
    "frontal_saturation": frontal_saturation,
    "feint_main_axis": feint_main_axis,
    "pincer": pincer,
    "isr_loiter": isr_loiter,
    "leader_follower": leader_follower,
}


def generate_dataset(n_per_doctrine: int, seed: int) -> List[Scenario]:
    rng = np.random.default_rng(seed)
    scenarios: List[Scenario] = []
    for name in sorted(DOCTRINES):
        for _ in range(n_per_doctrine):
            scenarios.append(DOCTRINES[name](rng))
    return scenarios
