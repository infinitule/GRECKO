"""DecentralizedDefense — Pillar D defender coordinator, an Allocator drop-in.

Same `allocate(AllocInput) -> List[Assignment]` contract as EconomicMDP, so it
slots straight into the eval harness and the bridge for a like-for-like study.
The difference is entirely in *how* the assignment is reached:

  * EconomicMDP: one solver sees every interceptor and every track and runs a
    global Hungarian assignment.
  * DecentralizedDefense: every interceptor is an autonomous DroneAgent. Agents
    exchange Claim messages only along the live comms mesh (`AllocInput.adjacency`,
    the topology `sim/comms` publishes each tick) and converge on a conflict-free
    plan by synchronous gossip consensus. No node ever sees the global picture.

Convergence properties (verified in test_pd_swarm.py):
  * Within one connected mesh partition, consensus reaches a conflict-free
    assignment — no two agents engage the same track.
  * Across partitions, agents that cannot hear each other may both claim a
    commonly-visible track. That double-commit is the honest, measured cost of
    losing comms — it shows up as extra spend / worse cost-exchange under
    denial, which is exactly the trade Pillar D exists to characterise.

Determinism: synchronous (Jacobi) updates over sorted agents, integer-stable
tiebreaks, and a seeded generator for optional message loss. Same inputs →
byte-identical assignments.

SCOPE: coordination and assignment only. Produces the same advisory Assignment
objects the central allocator does; the C2 interlock downstream is unchanged.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from sim.alloc.interface import AllocInput, Allocator
from sim.alloc.types import Assignment, MagazineState, Provenance
from sim.swarm.agent import DroneAgent, LocalPick
from sim.swarm.consensus import best_claim
from sim.swarm.messages import Claim


class DecentralizedDefense(Allocator):
    def __init__(
        self,
        max_rounds: int = 0,
        drop_rate: float = 0.0,
        seed: int = 0,
    ) -> None:
        """max_rounds=0 → auto (derive a safe convergence bound from fleet size).
        drop_rate → per-message Bernoulli loss during gossip (comms stress).
        """
        self.max_rounds = max_rounds
        self.drop_rate = drop_rate
        self.seed = seed
        # Diagnostics from the last allocate() call (consumed by the CLI/eval).
        self.last_rounds_used = 0
        self.last_collisions = 0

    # ------------------------------------------------------------------ #

    def allocate(self, inp: AllocInput) -> List[Assignment]:
        agents = {
            iv.interceptor_id: DroneAgent(iv, inp.effector_catalogue)
            for iv in inp.interceptors
        }
        ids = sorted(agents)
        for a in agents.values():
            a.assess(inp.assessments, inp.magazine.rounds)

        neighbours = self._neighbour_map(ids, inp.adjacency)
        rng = np.random.default_rng((self.seed & 0xFFFFFFFF) ^ 0xD3CE)

        # Per-agent state: current pick, current claim, tracks conceded so far,
        # and the best-claim table each agent has heard (its local belief).
        picks: Dict[str, LocalPick] = {}
        claims: Dict[str, Optional[Claim]] = {}
        conceded: Dict[str, Set[str]] = {i: set() for i in ids}
        belief: Dict[str, Dict[str, Claim]] = {i: {} for i in ids}

        for i in ids:
            picks[i] = agents[i].pick(conceded[i])
            claims[i] = agents[i].claim(picks[i], inp.t)
            if claims[i] is not None:
                belief[i] = {claims[i].track_id: claims[i]}

        cap = self.max_rounds if self.max_rounds > 0 else max(4, 4 * len(ids))
        rounds_used = 0
        for _ in range(cap):
            rounds_used += 1
            new_belief, new_conceded, changed = self._gossip_round(
                ids, agents, neighbours, claims, belief, conceded, rng, inp.t
            )
            belief = new_belief
            conceded = new_conceded
            # Recompute each agent's pick/claim from its refreshed concessions.
            next_claims: Dict[str, Optional[Claim]] = {}
            for i in ids:
                picks[i] = agents[i].pick(conceded[i])
                next_claims[i] = agents[i].claim(picks[i], inp.t)
                if next_claims[i] is not None:
                    belief[i][next_claims[i].track_id] = best_claim(
                        [next_claims[i]] + (
                            [belief[i][next_claims[i].track_id]]
                            if next_claims[i].track_id in belief[i] else []
                        )
                    )
            claims = next_claims
            if not changed:
                break

        self.last_rounds_used = rounds_used
        return self._emit(inp, agents, ids, picks, conceded)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _neighbour_map(ids: List[str], adjacency: Dict[str, List[str]]) -> Dict[str, List[str]]:
        idset = set(ids)
        return {
            i: sorted(n for n in adjacency.get(i, []) if n in idset and n != i)
            for i in ids
        }

    def _gossip_round(
        self, ids, agents, neighbours, claims, belief, conceded, rng, t,
    ):
        """One synchronous gossip step.

        Each agent merges its neighbours' current beliefs (subject to optional
        message loss), then checks whether it still holds its own claimed track.
        If a stronger neighbour claim exists for that track, the agent concedes
        it. Returns the refreshed belief/concession state and whether anything
        changed (fixpoint detection).
        """
        new_belief: Dict[str, Dict[str, Claim]] = {i: dict(belief[i]) for i in ids}
        new_conceded: Dict[str, Set[str]] = {i: set(conceded[i]) for i in ids}
        changed = False

        for i in ids:
            merged = dict(belief[i])
            for j in neighbours[i]:
                if self.drop_rate > 0.0 and rng.random() < self.drop_rate:
                    continue  # message from j lost this round
                for tid, c in belief[j].items():
                    incumbent = merged.get(tid)
                    merged[tid] = best_claim(
                        [c] + ([incumbent] if incumbent else [])
                    )
            new_belief[i] = merged

            my = claims[i]
            if my is not None:
                winner = merged.get(my.track_id)
                if winner is not None and winner.src != i:
                    # A neighbour is more capable (or equal-capable, cheaper):
                    # concede this track and re-pick next round.
                    if my.track_id not in new_conceded[i]:
                        new_conceded[i].add(my.track_id)
                        changed = True

        return new_belief, new_conceded, changed

    def _emit(self, inp, agents, ids, picks, conceded) -> List[Assignment]:
        """Turn each agent's final pick into an Assignment, expending magazine.

        Collisions across comms partitions are preserved (both agents ASSIGN the
        same track) and counted for diagnostics, because suppressing them would
        hide the real cost of a partitioned mesh.
        """
        mag: MagazineState = inp.magazine.copy()
        out: List[Assignment] = []
        claimed_by: Dict[str, str] = {}
        collisions = 0

        for i in ids:
            pk = picks[i]
            agent = agents[i]
            if pk.track_id is None or pk.effector_id is None:
                out.append(self._hold(inp, agent, mag,
                                      "no winnable threat in local picture"))
                continue
            if not mag.can_fire(pk.effector_id):
                out.append(self._hold(inp, agent, mag,
                                      "magazine empty for chosen effector"))
                continue
            if pk.track_id in claimed_by:
                collisions += 1  # cross-partition double-commit
            claimed_by.setdefault(pk.track_id, i)
            mag.expend(pk.effector_id)
            out.append(Assignment(
                t=inp.t, interceptor_id=i, action="ASSIGN",
                track_id=pk.track_id, effector_id=pk.effector_id,
                provenance=Provenance(
                    solver="DecentralizedDefense",
                    bid_value=pk.capability,
                    track_value_estimate=pk.severity,
                    magazine_state=mag.to_dict(),
                    round=self.last_rounds_used,
                    hold_reason=None,
                ),
            ))

        self.last_collisions = collisions
        return out

    @staticmethod
    def _hold(inp, agent, mag, reason) -> Assignment:
        return Assignment(
            t=inp.t, interceptor_id=agent.id, action="HOLD_FIRE",
            track_id=None, effector_id=None,
            provenance=Provenance(
                solver="DecentralizedDefense", bid_value=0.0,
                track_value_estimate=0.0, magazine_state=mag.to_dict(),
                round=0, hold_reason=reason,
            ),
        )
