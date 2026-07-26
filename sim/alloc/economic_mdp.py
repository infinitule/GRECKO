"""EconomicMDP allocator — Pillar A, the central IP.

Formalises counter-swarm as a magazine-constrained sequential allocation problem:

  OBJECTIVE over a rolling horizon H:
    minimise  E[ Σ_j v_j * P(leak_j) ]  +  λ * E[ magazine cost spent ]
    s.t.      per-type magazine budgets, reload/range/Pk feasibility.

Where:
  v_j   = leak damage value (confidence-weighted priority score from P3/PB)
  P(leak_j) = 1 - Σ_i x_ij * Pk(effector_i, threat_type_j)
  λ     = cost-exchange knob (AllocInput.lambda_cost); higher = more rationing

The MDP novelty: this allocator can output HOLD_FIRE on a low-value track
when the horizon objective is better served by preserving magazine for later
waves. Standard WTA cannot do this.

Decentralisation: each interceptor solves its local assignment problem given
only the tracks reachable via its comms partition (adjacency graph). When
partitioned, nodes optimise independently (graceful degradation). The
adjacency graph is injected per-round from the PC layer.

Implementation: receding-horizon greedy with explicit cost-exchange gating.
True MDP value-function computation is O(2^n * n) and deferred to v2 (with RL
from the league). This version solves the single-horizon assignment exactly
using a cost-exchange threshold derived from the lambda parameter.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from sim.alloc.interface import AllocInput, Allocator, InterceptorState
from sim.alloc.types import Assignment, MagazineState, Provenance
from sim.classify.classifier import ThreatAssessment
from sim.effectors.catalogue import EffectorType


def _expected_kill_value(
    iv: InterceptorState,
    assess: ThreatAssessment,
    eff: EffectorType,
) -> float:
    """E[damage_prevented] = v_j * Pk(effector, threat_type)."""
    threat_type = getattr(assess, "threat_type", "quadrotor")
    pk = eff.p_k(threat_type)
    return assess.priority_score * pk


def _cost_exchange_ratio(kill_value: float, cost: float, lam: float,
                          asset_value: float) -> float:
    """Net benefit of an engagement.

    benefit = kill_value - λ * (cost / asset_value)

    Normalising cost by asset_value makes λ a dimensionless knob in [0, 1]:
    λ=0 → engage everything; λ=1 → only engage if kill_value > cost/asset_value.
    """
    return kill_value - lam * (cost / max(asset_value, 1.0))


def _partition_of(interceptor_id: str, adjacency: Dict[str, List[str]]) -> Set[str]:
    """BFS to find all nodes reachable from this interceptor through the comms graph."""
    if interceptor_id not in adjacency:
        return {interceptor_id}
    visited = set()
    queue = [interceptor_id]
    while queue:
        n = queue.pop()
        if n in visited:
            continue
        visited.add(n)
        queue.extend(adjacency.get(n, []))
    return visited


class EconomicMDP(Allocator):
    def allocate(self, inp: AllocInput) -> List[Assignment]:
        assignments: List[Assignment] = []
        mag = inp.magazine.copy()
        lam = inp.lambda_cost
        engaged: Set[str] = set()

        # Group interceptors by their comms partition so each partition
        # optimises jointly and avoids duplicate assignment within the partition.
        partitions = self._build_partitions(inp)

        round_idx = 0
        for part_ids in partitions:
            part_ivs = [iv for iv in inp.interceptors if iv.interceptor_id in part_ids]
            part_assigns, round_idx = self._solve_partition(
                inp, part_ivs, mag, lam, engaged, round_idx
            )
            for a in part_assigns:
                if a.action == "ASSIGN" and a.track_id:
                    engaged.add(a.track_id)
                    if a.effector_id:
                        mag.expend(a.effector_id)
            assignments.extend(part_assigns)

        return assignments

    # ------------------------------------------------------------------ #

    def _build_partitions(self, inp: AllocInput) -> List[Set[str]]:
        """Connected components among interceptors only (C2 nodes ignored here)."""
        iv_ids = {iv.interceptor_id for iv in inp.interceptors}
        adj = {k: [v for v in vs if v in iv_ids]
               for k, vs in inp.adjacency.items() if k in iv_ids}
        # Add singletons for interceptors not in adj
        for iid in iv_ids:
            adj.setdefault(iid, [])

        seen, parts = set(), []
        for start in sorted(iv_ids):
            if start in seen:
                continue
            comp = _partition_of(start, adj) & iv_ids
            seen |= comp
            parts.append(comp)
        return parts

    def _solve_partition(
        self,
        inp: AllocInput,
        ivs: List[InterceptorState],
        mag: MagazineState,
        lam: float,
        already_engaged: Set[str],
        round_start: int,
    ) -> Tuple[List[Assignment], int]:
        """Solve the single-horizon assignment for one comms partition.

        Uses cost-benefit gating: if the best engagement for an interceptor
        has a negative net benefit (kill_value - λ*cost < 0), output HOLD_FIRE
        with the rationale — this is the magazine-rationing behaviour.

        Inside the partition, we run Hungarian assignment on the benefit matrix
        to avoid duplicate assignments (same mechanism as the OracleLP but
        limited to the partition's horizon).
        """
        avail_tracks = [a for a in inp.assessments
                        if a.track_id not in already_engaged]
        if not avail_tracks:
            return [_hold(inp.t, iv, mag, "No targets", "EconomicMDP", round_start)
                    for iv in ivs], round_start + 1

        # Build benefit matrix: rows=interceptors, cols=tracks
        nI, nT = len(ivs), len(avail_tracks)
        benefit = np.full((nI, nT), -1e9)

        for i, iv in enumerate(ivs):
            for j, assess in enumerate(avail_tracks):
                eff = self._select_effector(iv, assess, mag, inp.effector_catalogue)
                if eff is None:
                    continue
                kv = _expected_kill_value(iv, assess, eff)
                net = _cost_exchange_ratio(kv, eff.cost_usd, lam, inp.asset_value)
                benefit[i, j] = net

        # Hungarian on negated benefit (minimise cost = maximise benefit)
        row_ind, col_ind = linear_sum_assignment(-benefit)

        assignments: List[Assignment] = []
        assigned_ivs: Set[int] = set()

        for i, j in zip(row_ind, col_ind):
            if benefit[i, j] <= 0.0:
                # Net benefit non-positive → HOLD this interceptor
                continue
            iv = ivs[i]
            assess = avail_tracks[j]
            eff = self._select_effector(iv, assess, mag, inp.effector_catalogue)
            if eff is None or not mag.can_fire(eff.effector_id):
                continue
            kv = _expected_kill_value(iv, assess, eff)
            assignments.append(Assignment(
                t=inp.t, interceptor_id=iv.interceptor_id,
                action="ASSIGN",
                track_id=assess.track_id,
                effector_id=eff.effector_id,
                provenance=Provenance(
                    solver="EconomicMDP",
                    bid_value=benefit[i, j],
                    track_value_estimate=assess.priority_score,
                    magazine_state=mag.to_dict(),
                    round=round_start,
                ),
            ))
            assigned_ivs.add(i)
            mag.expend(eff.effector_id)

        # HOLDs for unassigned interceptors
        for i, iv in enumerate(ivs):
            if i not in assigned_ivs:
                best_net = max(benefit[i]) if nT > 0 else -1e9
                reason = (
                    "Cost-exchange negative: preserving magazine for later waves"
                    if best_net <= 0.0
                    else "Magazine empty or no feasible effector"
                )
                assignments.append(_hold(inp.t, iv, mag, reason, "EconomicMDP", round_start))

        return assignments, round_start + 1

    def _select_effector(
        self,
        iv: InterceptorState,
        assess: ThreatAssessment,
        mag: MagazineState,
        catalogue: Dict[str, EffectorType],
    ) -> Optional[EffectorType]:
        """Best effector by cost-adjusted Pk that is feasible and has ammo."""
        r = float(np.linalg.norm(assess.features.pos - iv.pos))
        threat_type = getattr(assess, "threat_type", "quadrotor")
        candidates = [
            e for e in catalogue.values()
            if mag.can_fire(e.effector_id) and e.geometry_valid(r, math.pi)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.p_k(threat_type) / max(e.cost_usd, 1.0))


def _hold(t, iv, mag, reason, solver="EconomicMDP", rnd=0) -> Assignment:
    return Assignment(
        t=t, interceptor_id=iv.interceptor_id,
        action="HOLD_FIRE", track_id=None, effector_id=None,
        provenance=Provenance(
            solver=solver, bid_value=0.0,
            track_value_estimate=0.0,
            magazine_state=mag.to_dict(),
            round=rnd,
            hold_reason=reason,
        ),
    )
