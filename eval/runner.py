"""PX evaluation harness — the full MC study driver.

Pulls attack patterns from the PL tactic library and runs paired allocator
trials (EconomicMDP vs GreedyMyopic) under each, returning the headline
cost-exchange figure and per-scenario breakdown.

Scenarios:
  "default"   — BridgeScenario's built-in formation (known-good baseline)
  "pl_N"      — N-th policy from the PL tactic library (adversarial attacks)
"""
from __future__ import annotations

import json
import pathlib
from typing import List, Optional

import numpy as np

from eval.metrics import aggregate, headline_figure, split_by
from eval.monte_carlo import EpisodeMetrics, run_mc_episode
from league.league import LeagueRunner
from league.policy import SwarmPolicy
from s2r.gap import RealityGap
from sim.alloc.economic_mdp import EconomicMDP
from sim.alloc.greedy import GreedyMyopic

# Small league run to populate the tactic library for PX evaluation.
# 4 gens × 6 pop × 1 ep — fast, deterministic, seeded apart from PL tests.
_PX_LEAGUE_N_GEN = 4
_PX_LEAGUE_POP = 6
_PX_LEAGUE_SEED = 42


def _build_tactic_library(top_n: int = 3) -> List[Optional[SwarmPolicy]]:
    """Return the top-N discovered tactics plus None (default formation)."""
    runner = LeagueRunner(
        n_generations=_PX_LEAGUE_N_GEN,
        pop_size=_PX_LEAGUE_POP,
        n_eval_episodes=1,
        seed=_PX_LEAGUE_SEED,
    )
    runner.run()
    tactics = runner.tactic_library(top_n=top_n)
    return tactics


def run_px_study(
    n_seeds: int = 6,
    n_tactics: int = 3,
    gap: Optional[RealityGap] = None,
    lambda_cost: float = 0.05,
) -> dict:
    """Full PX Monte Carlo study.

    For each scenario (default + n_tactics PL policies), run n_seeds episodes
    under both EconomicMDP and GreedyMyopic. Return the headline figure and
    per-scenario breakdown.
    """
    if gap is None:
        gap = RealityGap.nominal()

    tactics = _build_tactic_library(top_n=n_tactics)

    # Scenarios: None = default formation, else SwarmPolicy
    scenario_configs = [(None, "default")] + [
        (p, f"pl_{i}") for i, p in enumerate(tactics)
    ]

    eco_alloc = EconomicMDP()
    grd_alloc = GreedyMyopic()

    all_eco: List[EpisodeMetrics] = []
    all_grd: List[EpisodeMetrics] = []

    for policy, label in scenario_configs:
        for seed in range(n_seeds):
            eco_ep = run_mc_episode(
                seed=seed, allocator=eco_alloc, allocator_name="EconomicMDP",
                scenario_label=label, policy=policy, gap=gap,
                lambda_cost=lambda_cost,
            )
            grd_ep = run_mc_episode(
                seed=seed, allocator=grd_alloc, allocator_name="GreedyMyopic",
                scenario_label=label, policy=policy, gap=gap,
                lambda_cost=lambda_cost,
            )
            all_eco.append(eco_ep)
            all_grd.append(grd_ep)

    headline = headline_figure(all_eco, all_grd)

    eco_by_scenario = {
        sc: aggregate(eps, f"EconomicMDP/{sc}").to_dict()
        for sc, eps in split_by(all_eco, "scenario_label").items()
    }
    grd_by_scenario = {
        sc: aggregate(eps, f"GreedyMyopic/{sc}").to_dict()
        for sc, eps in split_by(all_grd, "scenario_label").items()
    }

    return {
        "config": {
            "n_seeds": n_seeds,
            "n_tactics": n_tactics,
            "lambda_cost": lambda_cost,
            "gap": gap.to_dict(),
            "league_seed": _PX_LEAGUE_SEED,
        },
        "headline": headline,
        "by_scenario": {
            "EconomicMDP": eco_by_scenario,
            "GreedyMyopic": grd_by_scenario,
        },
        "all_episodes": {
            "EconomicMDP": [e.to_dict() for e in all_eco],
            "GreedyMyopic": [e.to_dict() for e in all_grd],
        },
    }
