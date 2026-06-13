"""OracleLP — full-information LP relaxation. EVAL UPPER BOUND ONLY.

This solver has access to perfect kill probabilities and ground-truth threat
values. It is NOT deployed; it is the theoretical ceiling against which
EconomicMDP's optimality gap is measured.

Uses scipy.optimize.linear_sum_assignment (Hungarian) on the full
interceptor × track benefit matrix with no comms or partition constraints.
Effectively: what is the best possible assignment if we had perfect comms
and perfect information?
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

from sim.alloc.interface import AllocInput, Allocator, InterceptorState
from sim.alloc.types import Assignment, MagazineState, Provenance
from sim.classify.classifier import ThreatAssessment
from sim.effectors.catalogue import EffectorType


class OracleLP(Allocator):
    """Eval upper bound: centralised, full-information, no comms constraint."""

    def allocate(self, inp: AllocInput) -> List[Assignment]:
        mag = inp.magazine.copy()
        lam = inp.lambda_cost
        ivs = inp.interceptors
        tracks = inp.assessments

        if not tracks:
            return [_hold(inp.t, iv, mag, "No targets") for iv in ivs]

        nI, nT = len(ivs), len(tracks)
        benefit = np.full((nI, nT), -1e9)
        eff_choice: Dict[tuple, Optional[EffectorType]] = {}

        for i, iv in enumerate(ivs):
            for j, assess in enumerate(tracks):
                r = float(np.linalg.norm(assess.features.pos - iv.pos))
                threat_type = getattr(assess, "threat_type", "quadrotor")
                cands = [
                    e for e in inp.effector_catalogue.values()
                    if mag.can_fire(e.effector_id) and e.geometry_valid(r, math.pi)
                ]
                if not cands:
                    eff_choice[(i, j)] = None
                    continue
                best = max(cands, key=lambda e: e.p_k(threat_type) / max(e.cost_usd, 1.0))
                kv = assess.priority_score * best.p_k(threat_type)
                benefit[i, j] = kv - lam * (best.cost_usd / max(inp.asset_value, 1.0))
                eff_choice[(i, j)] = best

        row_ind, col_ind = linear_sum_assignment(-benefit)
        assignments: List[Assignment] = []
        assigned_ivs: Set[int] = set()

        for i, j in zip(row_ind, col_ind):
            if benefit[i, j] <= 0.0:
                continue
            eff = eff_choice.get((i, j))
            if eff is None or not mag.can_fire(eff.effector_id):
                continue
            iv, assess = ivs[i], tracks[j]
            assignments.append(Assignment(
                t=inp.t, interceptor_id=iv.interceptor_id,
                action="ASSIGN",
                track_id=assess.track_id,
                effector_id=eff.effector_id,
                provenance=Provenance(
                    solver="OracleLP",
                    bid_value=float(benefit[i, j]),
                    track_value_estimate=assess.priority_score,
                    magazine_state=mag.to_dict(),
                    round=0,
                ),
            ))
            assigned_ivs.add(i)
            mag.expend(eff.effector_id)

        for i, iv in enumerate(ivs):
            if i not in assigned_ivs:
                assignments.append(_hold(inp.t, iv, mag,
                                         "No positive-benefit assignment available"))
        return assignments


def _hold(t, iv, mag, reason) -> Assignment:
    return Assignment(
        t=t, interceptor_id=iv.interceptor_id,
        action="HOLD_FIRE", track_id=None, effector_id=None,
        provenance=Provenance(
            solver="OracleLP", bid_value=0.0,
            track_value_estimate=0.0,
            magazine_state=mag.to_dict(), round=0,
            hold_reason=reason,
        ),
    )
