"""Cost-exchange headline figure and aggregate statistics.

Aggregates a batch of EpisodeMetrics into the headline cost-exchange ratio
and supporting confidence interval, plus per-allocator / per-scenario splits.
"""
from __future__ import annotations

import dataclasses
import statistics
from typing import Dict, List

import numpy as np

from eval.monte_carlo import EpisodeMetrics


@dataclasses.dataclass
class AggregateStats:
    label: str
    n: int
    mean_intercept_rate: float
    mean_cost_exchange_ratio: float
    p10_cost_exchange: float      # low-cost (favourable) tail
    p90_cost_exchange: float      # high-cost (unfavourable) tail
    mean_leakers: float
    mean_asset_damage_usd: float
    mean_defense_spend_usd: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def aggregate(episodes: List[EpisodeMetrics], label: str = "") -> AggregateStats:
    if not episodes:
        raise ValueError("cannot aggregate an empty episode list")
    irs = [e.intercept_rate for e in episodes]
    cers = [e.cost_exchange_ratio for e in episodes]
    sorted_cers = sorted(cers)
    n = len(episodes)
    p10 = sorted_cers[max(0, int(0.10 * n) - 1)]
    p90 = sorted_cers[min(n - 1, int(0.90 * n))]
    return AggregateStats(
        label=label,
        n=n,
        mean_intercept_rate=round(float(np.mean(irs)), 4),
        mean_cost_exchange_ratio=round(float(np.mean(cers)), 4),
        p10_cost_exchange=round(p10, 4),
        p90_cost_exchange=round(p90, 4),
        mean_leakers=round(float(np.mean([e.leakers for e in episodes])), 4),
        mean_asset_damage_usd=round(
            float(np.mean([e.asset_damage_usd for e in episodes])), 2),
        mean_defense_spend_usd=round(
            float(np.mean([e.defense_spend_usd for e in episodes])), 2),
    )


def split_by(episodes: List[EpisodeMetrics],
             key: str) -> Dict[str, List[EpisodeMetrics]]:
    """Group episodes by a field name ('allocator_name', 'scenario_label')."""
    groups: Dict[str, List[EpisodeMetrics]] = {}
    for ep in episodes:
        val = str(getattr(ep, key))
        groups.setdefault(val, []).append(ep)
    return groups


def headline_figure(
    economic_episodes: List[EpisodeMetrics],
    greedy_episodes: List[EpisodeMetrics],
) -> dict:
    """Top-level headline: EconomicMDP vs Greedy across all scenarios.

    The cost-exchange improvement is the fractional reduction in defense
    spend per intercepted threat when the EconomicMDP replaces the Greedy
    allocator. Positive values mean EconomicMDP is cheaper to operate.
    """
    eco = aggregate(economic_episodes, "EconomicMDP")
    grd = aggregate(greedy_episodes, "GreedyMyopic")

    cer_improvement = 0.0
    if grd.mean_cost_exchange_ratio > 0:
        cer_improvement = round(
            (grd.mean_cost_exchange_ratio - eco.mean_cost_exchange_ratio)
            / grd.mean_cost_exchange_ratio,
            4,
        )

    return {
        "economic_mdp": eco.to_dict(),
        "greedy_myopic": grd.to_dict(),
        "cer_improvement_frac": cer_improvement,
        "intercept_rate_delta": round(
            eco.mean_intercept_rate - grd.mean_intercept_rate, 4),
    }
