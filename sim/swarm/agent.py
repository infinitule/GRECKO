"""DroneAgent — one autonomous node's local reasoning.

An agent wraps a single interceptor. It does three things, all from local
information only:

  1. assess()   — score every visible threat by severity, and compute how well
                  this node in particular could neutralize each (capability).
  2. pick()     — choose the highest-severity threat it can still win, given the
                  set of tracks already conceded to stronger neighbours.
  3. claim()    — emit the Claim message its neighbours will arbitrate against.

The agent never sees the global picture; it only ever reasons over the tracks
and neighbour claims that reach it through the comms mesh. That locality is the
whole point of Pillar D.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Set

import numpy as np

from sim.alloc.interface import InterceptorState
from sim.classify.classifier import ThreatAssessment
from sim.effectors.catalogue import EffectorType
from sim.swarm.messages import Claim


@dataclasses.dataclass(frozen=True)
class LocalPick:
    """An agent's current best intended engagement (or None target = hold)."""
    track_id: Optional[str]
    effector_id: Optional[str]
    severity: float
    capability: float
    cost: float
    reason: str


def _time_to_reach(iv: InterceptorState, track_pos: np.ndarray) -> float:
    dist = float(np.linalg.norm(track_pos - iv.pos))
    return dist / max(iv.speed_mps, 1.0)


class DroneAgent:
    def __init__(
        self,
        iv: InterceptorState,
        catalogue: Dict[str, EffectorType],
    ) -> None:
        self.id = iv.interceptor_id
        self.iv = iv
        self.catalogue = catalogue
        # Populated by assess(): track_id -> (severity, capability, cost, eff_id)
        self._table: Dict[str, tuple] = {}

    # -- 1. local threat assessment ------------------------------------------

    def _best_effector(
        self, assess: ThreatAssessment, mag_rounds: Dict[str, int]
    ) -> Optional[EffectorType]:
        """Highest-P_k feasible effector this node still has rounds for.

        Threat-primary: we pick for kill probability against THIS threat first,
        and let cost act only as the downstream tiebreak in consensus — so the
        selection here maximises P_k, not P_k/cost.
        """
        r = float(np.linalg.norm(assess.features.pos - self.iv.pos))
        threat_type = getattr(assess, "threat_type", "quadrotor")
        candidates = [
            e for e in self.catalogue.values()
            if mag_rounds.get(e.effector_id, 0) > 0 and e.geometry_valid(r, math.pi)
        ]
        if not candidates:
            return None
        # Tie-break equal-P_k effectors toward the cheaper one for a stable pick.
        return max(candidates, key=lambda e: (e.p_k(threat_type), -e.cost_usd))

    def assess(
        self,
        assessments: List[ThreatAssessment],
        mag_rounds: Dict[str, int],
    ) -> None:
        """Build this node's local threat/capability table."""
        self._table = {}
        for a in assessments:
            eff = self._best_effector(a, mag_rounds)
            if eff is None:
                continue
            threat_type = getattr(a, "threat_type", "quadrotor")
            pk = eff.p_k(threat_type)
            if pk <= 0.0:
                continue
            # capability = how well AND how soon this node can neutralize it.
            # urgency factor 1/(1+ttr) rewards nodes that can reach it sooner;
            # bounded so it never dominates the P_k term.
            ttr = _time_to_reach(self.iv, a.features.pos)
            capability = pk * (1.0 / (1.0 + 0.02 * ttr))
            self._table[a.track_id] = (
                float(a.priority_score), float(capability),
                float(eff.cost_usd), eff.effector_id,
            )

    # -- 2. pick the best still-winnable target ------------------------------

    @property
    def engageable(self) -> Set[str]:
        return set(self._table)

    def severity(self, track_id: str) -> float:
        return self._table.get(track_id, (0.0,))[0]

    def capability(self, track_id: str) -> float:
        t = self._table.get(track_id)
        return t[1] if t else 0.0

    def pick(self, conceded: Set[str]) -> LocalPick:
        """Choose the highest-severity threat not yet conceded to a neighbour.

        `conceded` is the set of tracks this node has lost consensus on this
        round; it re-picks among the rest. Threat-primary: we sort by severity
        (the threat), and only among what we can actually engage.
        """
        options = [
            (tid, self._table[tid]) for tid in self._table if tid not in conceded
        ]
        if not options:
            return LocalPick(None, None, 0.0, 0.0, 0.0,
                             "no winnable threat in local picture — HOLD")
        # Highest severity first; capability then cost as internal tiebreaks so
        # the choice is deterministic and stable.
        tid, (sev, cap, cost, eff_id) = max(
            options, key=lambda kv: (kv[1][0], kv[1][1], -kv[1][2], kv[0])
        )
        return LocalPick(
            track_id=tid, effector_id=eff_id,
            severity=sev, capability=cap, cost=cost,
            reason=f"top local threat sev={sev:.4f}, capable P_k-based={cap:.3f}",
        )

    # -- 3. emit the claim message -------------------------------------------

    def claim(self, pick: LocalPick, t: float) -> Optional[Claim]:
        if pick.track_id is None:
            return None
        return Claim(
            src=self.id, track_id=pick.track_id,
            capability=pick.capability, cost=pick.cost, t=t,
        )
