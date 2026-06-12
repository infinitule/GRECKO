"""SwarmPolicy — the parameter vector the league evolves.

A policy is a 14-dimensional real-valued vector that fully specifies a
red-team attack formation. It generates HostileUAS initial conditions for
BridgeScenario (live physics) and Scenario trajectories (for PB retraining).

Parameter space:
  [n_total, feint_frac, screen_frac, main_angle, feint_offset,
   main_speed, feint_speed, main_range, feint_range, t_feint_turn,
   main_spread, feint_spread, weave_amp, timing_offset]
"""
from __future__ import annotations

import dataclasses
import math
from typing import List

import numpy as np

from learn.intent.doctrines import Scenario, _ingress_group, _screen_group

N_POLICY_PARAMS = 14

# (min, max) bounds for each parameter dimension
PARAM_BOUNDS = np.array([
    [5.0,  14.0],               # 0  n_total  (mapped to int)
    [0.0,   0.50],              # 1  feint_frac
    [0.0,   0.30],              # 2  screen_frac
    [0.0,  2 * math.pi],        # 3  main_angle  (radians)
    [math.pi / 4, 3 * math.pi / 2],  # 4  feint_offset relative to main
    [18.0, 35.0],               # 5  main_speed  m/s (fast enough to arrive)
    [10.0,  22.0],              # 6  feint_speed m/s
    [300.0, 900.0],             # 7  main_range  m — arrives within episode window
    [200.0, 700.0],             # 8  feint_range m
    [5.0,   25.0],              # 9  t_feint_turn  s
    [20.0,  150.0],             # 10 main_spread  m
    [30.0,  200.0],             # 11 feint_spread m
    [0.0,   0.25],              # 12 weave_amp    rad
    [0.0,  10.0],               # 13 timing_offset s (unused in live sim)
], dtype=float)


def _clip(theta: np.ndarray) -> np.ndarray:
    return np.clip(theta, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])


@dataclasses.dataclass
class SwarmPolicy:
    theta: np.ndarray   # shape (N_POLICY_PARAMS,)
    policy_id: int = 0
    generation: int = 0
    fitness: float = -1.0

    # ------------------------------------------------------------------ #
    # Decoded parameters                                                   #
    # ------------------------------------------------------------------ #

    @property
    def n_total(self) -> int:
        return max(5, int(round(float(self.theta[0]))))

    @property
    def feint_frac(self) -> float:
        return float(self.theta[1])

    @property
    def screen_frac(self) -> float:
        return float(self.theta[2])

    @property
    def n_feint(self) -> int:
        return max(0, int(round(self.feint_frac * self.n_total)))

    @property
    def n_screen(self) -> int:
        return max(0, int(round(self.screen_frac * self.n_total)))

    @property
    def n_main(self) -> int:
        return max(1, self.n_total - self.n_feint - self.n_screen)

    @property
    def main_angle(self) -> float:
        return float(self.theta[3])

    @property
    def feint_angle(self) -> float:
        return float((float(self.theta[3]) + float(self.theta[4])) % (2 * math.pi))

    @property
    def main_speed(self) -> float:
        return float(self.theta[5])

    @property
    def feint_speed(self) -> float:
        return float(self.theta[6])

    @property
    def main_range(self) -> float:
        return float(self.theta[7])

    @property
    def feint_range(self) -> float:
        return float(self.theta[8])

    @property
    def t_feint_turn(self) -> float:
        return float(self.theta[9])

    @property
    def main_spread(self) -> float:
        return float(self.theta[10])

    @property
    def feint_spread(self) -> float:
        return float(self.theta[11])

    @property
    def weave_amp(self) -> float:
        return float(self.theta[12])

    # ------------------------------------------------------------------ #
    # Scenario generation (for PB retraining export)                      #
    # ------------------------------------------------------------------ #

    def to_scenario(self, rng: np.random.Generator) -> Scenario:
        """Generate a Scenario from this policy (pre-computed trajectories)."""
        trajs = []
        labels: List[str] = []

        if self.n_main > 0:
            g = _ingress_group(rng, self.n_main, self.main_angle,
                               self.main_range, self.main_speed, self.main_spread)
            trajs.append(g)
            labels += ["main_axis"] * self.n_main

        if self.n_feint > 0:
            g = _ingress_group(rng, self.n_feint, self.feint_angle,
                               self.feint_range, self.feint_speed, self.feint_spread,
                               t_turn_away=self.t_feint_turn)
            trajs.append(g)
            labels += ["feint"] * self.n_feint

        if self.n_screen > 0:
            g = _screen_group(rng, self.n_screen, self.main_angle,
                               self.main_range * 0.7, self.main_speed * 0.9)
            trajs.append(g)
            labels += ["screen"] * self.n_screen

        traj = np.concatenate(trajs, axis=0)
        return Scenario(
            doctrine=f"league_gen{self.generation}",
            trajectories=traj,
            labels=labels,
            asset_pos=np.zeros(2),
        )

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "generation": self.generation,
            "fitness": float(self.fitness),
            "theta": self.theta.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SwarmPolicy":
        return cls(
            theta=np.array(d["theta"], dtype=float),
            policy_id=int(d.get("policy_id", 0)),
            generation=int(d.get("generation", 0)),
            fitness=float(d.get("fitness", -1.0)),
        )

    @classmethod
    def random(cls, rng: np.random.Generator, policy_id: int = 0) -> "SwarmPolicy":
        theta = rng.uniform(PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
        return cls(theta=theta, policy_id=policy_id)

    def mutate(self, rng: np.random.Generator, sigma: float = 0.08) -> "SwarmPolicy":
        """Gaussian mutation — sigma is a fraction of each parameter's range."""
        scale = PARAM_BOUNDS[:, 1] - PARAM_BOUNDS[:, 0]
        noise = rng.normal(0.0, sigma, N_POLICY_PARAMS) * scale
        new_theta = _clip(self.theta + noise)
        return SwarmPolicy(theta=new_theta, policy_id=self.policy_id,
                           generation=self.generation)

    def crossover(self, other: "SwarmPolicy",
                  rng: np.random.Generator) -> "SwarmPolicy":
        """Uniform crossover with the other parent."""
        mask = rng.random(N_POLICY_PARAMS) < 0.5
        new_theta = np.where(mask, self.theta, other.theta)
        return SwarmPolicy(theta=new_theta.copy(), policy_id=self.policy_id,
                           generation=self.generation)
