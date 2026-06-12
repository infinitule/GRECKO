"""GreedyMyopic allocator — baseline; matches current best practice.

Assigns interceptors to the highest-priority unengaged track by nearest
TTA (time-to-intercept). Picks the cheapest feasible effector. No horizon,
no magazine reservation, no rationing: this is the allocator that wins the
first wave and loses the war.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

import numpy as np

from sim.alloc.interface import AllocInput, Allocator, InterceptorState
from sim.alloc.types import Assignment, Provenance
from sim.classify.classifier import ThreatAssessment
from sim.effectors.catalogue import EffectorType


def _tti(iv: InterceptorState, track_pos: np.ndarray) -> float:
    dist = float(np.linalg.norm(track_pos - iv.pos))
    return dist / max(iv.speed_mps, 1.0)


def _best_effector(
    iv: InterceptorState,
    track_pos: np.ndarray,
    magazine_state: dict,
    catalogue: Dict[str, EffectorType],
) -> Optional[EffectorType]:
    """Cheapest feasible effector with rounds remaining."""
    r = float(np.linalg.norm(track_pos - iv.pos))
    eff = catalogue.get(iv.effector_type)
    if eff is not None and magazine_state.get(eff.effector_id, 0) > 0 \
            and eff.geometry_valid(r, math.pi):
        return eff
    # Fall back to any affordable effector in catalogue
    candidates = [
        e for e in catalogue.values()
        if magazine_state.get(e.effector_id, 0) > 0 and e.geometry_valid(r, math.pi)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: e.cost_usd)


class GreedyMyopic(Allocator):
    def allocate(self, inp: AllocInput) -> List[Assignment]:
        assignments: List[Assignment] = []
        engaged: Set[str] = set()
        mag_remaining = inp.magazine.copy()

        for iv in inp.interceptors:
            best_assess: Optional[ThreatAssessment] = None
            best_tti = float("inf")
            for assess in inp.assessments:
                if assess.track_id in engaged:
                    continue
                tti = _tti(iv, assess.features.pos)
                if tti < best_tti:
                    best_tti = tti
                    best_assess = assess

            if best_assess is None:
                assignments.append(_hold(inp.t, iv, mag_remaining, "No targets available"))
                continue

            eff = _best_effector(iv, best_assess.features.pos,
                                 mag_remaining.rounds, inp.effector_catalogue)
            if eff is None:
                assignments.append(_hold(inp.t, iv, mag_remaining,
                                         "No feasible effector or magazine empty"))
                continue

            engaged.add(best_assess.track_id)
            mag_remaining.expend(eff.effector_id)
            assignments.append(Assignment(
                t=inp.t, interceptor_id=iv.interceptor_id,
                action="ASSIGN",
                track_id=best_assess.track_id,
                effector_id=eff.effector_id,
                provenance=Provenance(
                    solver="GreedyMyopic",
                    bid_value=best_assess.priority_score,
                    track_value_estimate=best_assess.priority_score,
                    magazine_state=mag_remaining.to_dict(),
                    round=0,
                ),
            ))

        return assignments


def _hold(t, iv, mag, reason) -> Assignment:
    return Assignment(
        t=t, interceptor_id=iv.interceptor_id,
        action="HOLD_FIRE", track_id=None, effector_id=None,
        provenance=Provenance(
            solver="GreedyMyopic", bid_value=0.0,
            track_value_estimate=0.0,
            magazine_state=mag.to_dict(), round=0,
            hold_reason=reason,
        ),
    )
