"""PX Monte Carlo evaluation — episode runner.

Runs N seeded episodes of BridgeScenario under a given configuration and
collects per-episode cost-exchange metrics. Configurations differ in:
  - attack formation (SwarmPolicy from PL tactic library or default)
  - allocator (EconomicMDP vs GreedyMyopic — same interface)
  - reality gap (RealityGap from PS)

SCOPE: simulation and analysis only; no hardware, no RF design.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np

from s2r.gap import RealityGap
from s2r.episodes import MAX_PROBE_TIME
from sim.alloc.interface import Allocator
from sim.bridge.scenario import BridgeScenario

# Nominal cost estimate per hostile UAS (quadrotor-class commercial drone).
# Used only to set the exchange-ratio denominator; not a weapon parameter.
THREAT_COST_USD = 1_000.0

# Episode time cap — same window used for the PL fitness runner.
MC_MAX_TIME = MAX_PROBE_TIME   # 25 s


@dataclasses.dataclass
class EpisodeMetrics:
    seed: int
    allocator_name: str
    scenario_label: str
    n_threats: int
    intercepts: int
    leakers: int
    asset_hp: float
    defense_spend_usd: float    # effector costs actually expended
    threat_value_at_risk: float  # n_threats × THREAT_COST_USD
    intercepted_value: float     # n_intercepted × THREAT_COST_USD
    asset_damage_usd: float      # (10 - hp) / 10 × asset_value
    time_s: float

    @property
    def intercept_rate(self) -> float:
        return self.intercepts / max(self.n_threats, 1)

    @property
    def cost_exchange_ratio(self) -> float:
        """Defense USD spent per intercepted USD of threat value.

        Ratio < 1 → defense is cost-positive (cheaper to intercept than the
        threat costs). Ratio > 1 → Red wins the cost exchange.
        """
        return self.defense_spend_usd / max(self.intercepted_value, 1.0)

    def to_dict(self) -> dict:
        return {
            **dataclasses.asdict(self),
            "intercept_rate": round(self.intercept_rate, 4),
            "cost_exchange_ratio": round(self.cost_exchange_ratio, 4),
        }


def run_mc_episode(
    seed: int,
    allocator: Allocator,
    allocator_name: str,
    scenario_label: str,
    policy=None,
    gap: Optional[RealityGap] = None,
    lambda_cost: float = 0.05,
    asset_value: float = 1_000_000.0,
) -> EpisodeMetrics:
    """Run one episode and return cost-exchange metrics."""
    if gap is None:
        gap = RealityGap.nominal()

    sc = BridgeScenario(
        seed=seed,
        auto_authorize=True,
        policy=policy,
        sensor_mesh=gap.apply_to_mesh(),
        comms_cfg=gap.make_comms_cfg(),
    )
    sc.allocator = allocator
    sc.c2_state.lambda_cost = lambda_cost

    while not sc.world.is_engagement_over() and sc.world.t < MC_MAX_TIME:
        sc.tick()

    summary = sc.world.summary()
    intercepts = int(summary["intercepts"])
    n_threats = len(sc.world.hostiles)

    # Defense spend = effector cost of each PHYSICALLY consumed interceptor.
    # One intercept event → one interceptor consumed → one effector expended.
    # This grounds the cost in what actually happened, not the alloc-cycle
    # re-targeting bookkeeping (which re-expends magazine for geometry updates).
    from sim.effectors.catalogue import CATALOGUE
    defense_spend = 0.0
    for e in sc.world.log.events:
        if e["type"] == "INTERCEPT":
            iv_id = e["data"]["interceptor_id"]
            iv = sc.world.interceptors.get(iv_id)
            if iv is not None:
                eff = CATALOGUE.get(iv.effector_type)
                if eff is not None:
                    defense_spend += eff.cost_usd

    threat_at_risk = n_threats * THREAT_COST_USD
    intercepted_val = intercepts * THREAT_COST_USD
    asset_damage = (10.0 - float(summary["asset_hp"])) / 10.0 * asset_value

    return EpisodeMetrics(
        seed=seed,
        allocator_name=allocator_name,
        scenario_label=scenario_label,
        n_threats=n_threats,
        intercepts=intercepts,
        leakers=int(summary["leakers"]),
        asset_hp=float(summary["asset_hp"]),
        defense_spend_usd=round(defense_spend, 2),
        threat_value_at_risk=round(threat_at_risk, 2),
        intercepted_value=round(intercepted_val, 2),
        asset_damage_usd=round(asset_damage, 2),
        time_s=round(float(sc.world.t), 3),
    )
