"""PD study — centralized vs decentralized coordination under comms denial.

Answers the question ADR-013 poses: on a clean mesh the central allocator wins
on raw cost-exchange (a global optimum dominates a local one); the value of the
decentralized coordinator is the *shape of its degradation curve* as comms
denial rises. This study runs both arms across a denial sweep and reports the
trade directly.

The sweep dial is `base_drop_rate` on the comms config — the fraction of
otherwise-up peer messages that fail. At 0.0 the mesh is clean; as it climbs,
the central path loses its ability to reach the fleet coherently while the
decentralized path keeps coordinating per surviving partition.

SCOPE: simulation and analysis only.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from eval.monte_carlo import EpisodeMetrics, run_mc_episode
from s2r.gap import RealityGap
from sim.alloc.economic_mdp import EconomicMDP
from sim.swarm.defense import DecentralizedDefense


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _summarise(eps: List[EpisodeMetrics]) -> dict:
    return {
        "mean_cost_exchange_ratio": round(_mean([e.cost_exchange_ratio for e in eps]), 4),
        "mean_intercept_rate": round(_mean([e.intercept_rate for e in eps]), 4),
        "mean_defense_spend_usd": round(_mean([e.defense_spend_usd for e in eps]), 2),
        "mean_leakers": round(_mean([float(e.leakers) for e in eps]), 3),
        "n_episodes": len(eps),
    }


def run_pd_study(
    n_seeds: int = 6,
    drop_rates: Optional[List[float]] = None,
    lambda_cost: float = 0.05,
) -> dict:
    """Run centralized (EconomicMDP) vs decentralized (DecentralizedDefense)
    across a comms-denial sweep and return per-arm, per-denial summaries plus
    the resilience delta.

    Both arms are the SAME Allocator interface into the SAME scenario, so the
    only difference is whether the decision is made centrally or negotiated
    peer-to-peer over the mesh.
    """
    if drop_rates is None:
        drop_rates = [0.0, 0.15, 0.30, 0.50]

    sweep: List[dict] = []
    for dr in drop_rates:
        central = EconomicMDP()
        decentral = DecentralizedDefense(drop_rate=dr, seed=1)

        gap = RealityGap.nominal()
        cen_eps: List[EpisodeMetrics] = []
        dec_eps: List[EpisodeMetrics] = []
        for seed in range(n_seeds):
            cen_eps.append(run_mc_episode(
                seed=seed, allocator=central, allocator_name="EconomicMDP",
                scenario_label=f"drop_{dr}", gap=gap, lambda_cost=lambda_cost,
            ))
            dec_eps.append(run_mc_episode(
                seed=seed, allocator=decentral, allocator_name="DecentralizedDefense",
                scenario_label=f"drop_{dr}", gap=gap, lambda_cost=lambda_cost,
            ))

        cen = _summarise(cen_eps)
        dec = _summarise(dec_eps)
        cer_gap = round(dec["mean_cost_exchange_ratio"] - cen["mean_cost_exchange_ratio"], 4)
        sweep.append({
            "drop_rate": dr,
            "centralized": cen,
            "decentralized": dec,
            "cer_gap_decentral_minus_central": cer_gap,
        })

    clean = sweep[0]
    worst = sweep[-1]
    return {
        "config": {
            "n_seeds": n_seeds,
            "drop_rates": drop_rates,
            "lambda_cost": lambda_cost,
        },
        "sweep": sweep,
        "headline": {
            "clean_mesh_cer_gap": clean["cer_gap_decentral_minus_central"],
            "worst_mesh_cer_gap": worst["cer_gap_decentral_minus_central"],
            "interpretation": (
                "Positive gap = decentralized costs more per intercept. The gap "
                "is expected to be largest on a clean mesh (central optimum wins) "
                "and to shrink or reverse as denial rises (central coherence "
                "degrades while peer coordination survives per partition)."
            ),
        },
    }
