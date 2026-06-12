"""Episode fitness evaluation.

Runs a SwarmPolicy against the Blue defense stack (BridgeScenario with
auto_authorize=True) and returns a scalar fitness score for the ES.

Red fitness:
  leakers  × 10   — successful penetrations are the primary objective
  damage   × 5    — partial asset HP loss counts
  mag_used × 0.5  — forcing Blue to spend magazine is a secondary goal
  intercepts × -1 — losing drones is mildly costly for red
"""
from __future__ import annotations

import dataclasses

import numpy as np

from league.policy import SwarmPolicy
from sim.bridge.scenario import BridgeScenario

MAX_EPISODE_TIME = 25.0   # seconds — agents at 300–900 m arrive within this window
_MAG_FULL = 50            # kinetic=8 + net=12 + ew=20 + collision=10


@dataclasses.dataclass
class EpisodeResult:
    leakers: int
    intercepts: int
    asset_hp: float
    time_s: float
    magazine_used: int
    fitness: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def evaluate_policy(
    policy: SwarmPolicy,
    seed: int = 0,
    n_episodes: int = 2,
) -> EpisodeResult:
    """Evaluate policy over n_episodes; return mean-fitness result."""
    all_results = []
    for ep in range(n_episodes):
        r = _run_episode(policy, seed=seed + ep)
        all_results.append(r)

    mean_fit = float(np.mean([r.fitness for r in all_results]))
    # Return the episode with the best fitness for representative metrics
    best = max(all_results, key=lambda r: r.fitness)
    return dataclasses.replace(best, fitness=mean_fit)


def _run_episode(policy: SwarmPolicy, seed: int) -> EpisodeResult:
    sc = BridgeScenario(seed=seed, auto_authorize=True, policy=policy)
    while (not sc.world.is_engagement_over()
           and sc.world.t < MAX_EPISODE_TIME):
        sc.tick()
    summary = sc.world.summary()
    mag_remaining = sum(sc.magazine.rounds.values())
    mag_used = _MAG_FULL - mag_remaining
    fit = _fitness(summary["leakers"], summary["intercepts"],
                   summary["asset_hp"], mag_used)
    return EpisodeResult(
        leakers=int(summary["leakers"]),
        intercepts=int(summary["intercepts"]),
        asset_hp=float(summary["asset_hp"]),
        time_s=float(sc.world.t),
        magazine_used=int(mag_used),
        fitness=fit,
    )


def _fitness(leakers: int, intercepts: int,
             asset_hp: float, mag_used: int) -> float:
    damage = 10.0 - asset_hp          # asset starts at HP=10
    return (leakers * 10.0
            + damage * 5.0
            - intercepts * 1.0
            + mag_used * 0.5)
