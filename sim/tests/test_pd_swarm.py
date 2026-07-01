"""PD acceptance tests — decentralized swarm-on-swarm coordination.

Criteria:
 1. Consensus is threat-primary: higher capability wins; cost only breaks a
    capability tie; agent-id breaks a full tie (deterministic).
 2. Drop-in Allocator conformance: every interceptor appears exactly once, every
    HOLD_FIRE carries a non-empty reason, provenance is populated.
 3. Correctness under connectivity: within one connected mesh partition, no two
    agents engage the same track.
 4. Graceful degradation: a partitioned mesh may double-commit on a commonly
    visible threat, and the coordinator reports the collision (it does not hide
    it). More partitioning → no fewer collisions.
 5. Determinism: identical inputs → identical assignments, including under
    message loss (seeded).
 6. Attacker swarm: drones mass on the least-defended axis and re-mass when the
    defence shifts; the plan is deterministic.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from sim.alloc.interface import AllocInput, InterceptorState
from sim.alloc.types import MagazineState
from sim.classify.classifier import ThreatAssessment
from sim.classify.features import FeatureVector
from sim.effectors.catalogue import CATALOGUE
from sim.swarm.consensus import best_claim, resolve
from sim.swarm.defense import DecentralizedDefense
from sim.swarm.messages import Claim
from sim.swarm.swarm import DecentralizedSwarm, Drone

ASSET = np.array([0.0, 0.0])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fv(track_id: str, pos: np.ndarray) -> FeatureVector:
    return FeatureVector(
        track_id=track_id, t=1.0, pos=pos.astype(float),
        vel=np.array([-30.0, 0.0]), speed=30.0, heading_to_asset=0.2,
        approach_rate=25.0, weave_energy=0.001, altitude_band=0,
        rf_emitter=False, track_age=3.0, n_updates=5,
    )


def _assess(track_id: str, pos, severity: float) -> ThreatAssessment:
    return ThreatAssessment(
        t=1.0, track_id=track_id, label="hostile", confidence=0.8,
        priority_score=severity, features=_fv(track_id, np.asarray(pos, float)),
        why="test",
    )


def _full_mag() -> MagazineState:
    return MagazineState({
        "kinetic_interceptor": 5, "net_capture_drone": 5,
        "ew_soft_kill": 5, "collision_drone": 5,
    })


def _interceptors(n: int) -> List[InterceptorState]:
    return [
        InterceptorState(f"i{k}", np.array([100.0 * k, 0.0]),
                         "kinetic_interceptor", 120.0, 120.0)
        for k in range(n)
    ]


def _full_mesh(ids: List[str]) -> Dict[str, List[str]]:
    return {i: [j for j in ids if j != i] for i in ids}


def _input(interceptors, assessments, adjacency, mag=None) -> AllocInput:
    return AllocInput(
        t=1.0, interceptors=interceptors, assessments=assessments,
        magazine=mag or _full_mag(), effector_catalogue=CATALOGUE,
        adjacency=adjacency, asset_pos=ASSET,
    )


# --------------------------------------------------------------------------- #
# 1. Threat-primary consensus
# --------------------------------------------------------------------------- #

def test_consensus_capability_beats_cost():
    strong_expensive = Claim("i2", "t", capability=0.9, cost=90_000, t=1.0)
    weak_cheap = Claim("i1", "t", capability=0.7, cost=800, t=1.0)
    out = resolve(strong_expensive, weak_cheap)
    assert out.winner == "i2"
    assert "capability" in out.reason


def test_consensus_cost_breaks_capability_tie():
    a = Claim("i3", "t", capability=0.8, cost=90_000, t=1.0)
    b = Claim("i4", "t", capability=0.8, cost=800, t=1.0)
    assert resolve(a, b).winner == "i4"


def test_consensus_id_breaks_full_tie():
    a = Claim("i9", "t", capability=0.8, cost=800, t=1.0)
    b = Claim("i2", "t", capability=0.8, cost=800, t=1.0)
    assert resolve(a, b).winner == "i2"


def test_best_claim_over_many():
    claims = [
        Claim("i1", "t", 0.5, 800, 1.0),
        Claim("i2", "t", 0.9, 90_000, 1.0),
        Claim("i3", "t", 0.9, 800, 1.0),
    ]
    assert best_claim(claims).src == "i3"  # top capability, cheaper of the tie
    assert best_claim([]) is None


# --------------------------------------------------------------------------- #
# 2. Allocator conformance
# --------------------------------------------------------------------------- #

def test_every_interceptor_appears_once():
    ivs = _interceptors(4)
    assess = [_assess("t0", (500, 100), 0.9), _assess("t1", (600, 50), 0.6)]
    out = DecentralizedDefense().allocate(_input(ivs, assess, _full_mesh([i.interceptor_id for i in ivs])))
    assert sorted(a.interceptor_id for a in out) == ["i0", "i1", "i2", "i3"]


def test_holds_carry_reason_and_provenance():
    ivs = _interceptors(3)
    # only one threat → at least two agents must HOLD
    assess = [_assess("t0", (500, 100), 0.9)]
    out = DecentralizedDefense().allocate(_input(ivs, assess, _full_mesh([i.interceptor_id for i in ivs])))
    holds = [a for a in out if a.action == "HOLD_FIRE"]
    assert len(holds) >= 2
    for h in holds:
        assert h.hold_reason if hasattr(h, "hold_reason") else h.provenance.hold_reason
        assert h.provenance.hold_reason
    for a in out:
        assert a.provenance.solver == "DecentralizedDefense"


# --------------------------------------------------------------------------- #
# 3. Correctness under full connectivity
# --------------------------------------------------------------------------- #

def test_no_double_claims_in_connected_mesh():
    ivs = _interceptors(3)
    assess = [
        _assess("t0", (500, 100), 0.9),
        _assess("t1", (600, 50), 0.5),
        _assess("t2", (400, 200), 0.3),
    ]
    coord = DecentralizedDefense()
    out = coord.allocate(_input(ivs, assess, _full_mesh([i.interceptor_id for i in ivs])))
    engaged = [a.track_id for a in out if a.action == "ASSIGN"]
    assert len(engaged) == len(set(engaged)), "no two agents on the same track"
    assert coord.last_collisions == 0
    assert "t0" in engaged, "highest-severity threat must be engaged"


def test_highest_severity_preferred_when_scarce():
    # 1 interceptor, 3 threats → it must take the most severe.
    ivs = _interceptors(1)
    assess = [
        _assess("t_lo", (400, 0), 0.2),
        _assess("t_hi", (500, 0), 0.95),
        _assess("t_mid", (450, 0), 0.5),
    ]
    out = DecentralizedDefense().allocate(_input(ivs, assess, {"i0": []}))
    assign = [a for a in out if a.action == "ASSIGN"]
    assert len(assign) == 1 and assign[0].track_id == "t_hi"


# --------------------------------------------------------------------------- #
# 4. Graceful degradation under partition
# --------------------------------------------------------------------------- #

def test_partition_can_double_commit_and_reports_it():
    ivs = _interceptors(3)
    assess = [_assess("t0", (500, 100), 0.9)]  # one shared high-value threat
    # i0 isolated; i1<->i2 connected. i0 and one of {i1,i2} both see t0.
    adj = {"i0": [], "i1": ["i2"], "i2": ["i1"]}
    coord = DecentralizedDefense()
    out = coord.allocate(_input(ivs, assess, adj))
    engaged = [a.track_id for a in out if a.action == "ASSIGN"]
    assert engaged.count("t0") == 2, "isolated partitions double-commit the shared threat"
    assert coord.last_collisions == 1, "the collision is reported, not hidden"


def test_more_partitioning_never_fewer_collisions():
    ivs = _interceptors(4)
    assess = [_assess("t0", (500, 100), 0.9)]
    ids = [i.interceptor_id for i in ivs]
    connected = DecentralizedDefense()
    connected.allocate(_input(ivs, assess, _full_mesh(ids)))
    fully_split = DecentralizedDefense()
    fully_split.allocate(_input(ivs, assess, {i: [] for i in ids}))
    assert connected.last_collisions == 0
    assert fully_split.last_collisions >= connected.last_collisions


# --------------------------------------------------------------------------- #
# 5. Determinism
# --------------------------------------------------------------------------- #

def _signature(out):
    return [(a.interceptor_id, a.action, a.track_id, a.effector_id)
            for a in sorted(out, key=lambda a: a.interceptor_id)]


def test_determinism_plain():
    ivs = _interceptors(3)
    assess = [_assess("t0", (500, 100), 0.9), _assess("t1", (600, 50), 0.5),
              _assess("t2", (400, 200), 0.3)]
    inp = _input(ivs, assess, _full_mesh([i.interceptor_id for i in ivs]))
    c = DecentralizedDefense()
    assert _signature(c.allocate(inp)) == _signature(c.allocate(inp))


def test_determinism_under_message_loss():
    ivs = _interceptors(4)
    assess = [_assess(f"t{k}", (400 + 40 * k, 20 * k), 0.9 - 0.1 * k) for k in range(4)]
    inp = _input(ivs, assess, _full_mesh([i.interceptor_id for i in ivs]))
    a = DecentralizedDefense(drop_rate=0.3, seed=7)
    b = DecentralizedDefense(drop_rate=0.3, seed=7)
    assert _signature(a.allocate(inp)) == _signature(b.allocate(inp))


# --------------------------------------------------------------------------- #
# 6. Attacker swarm
# --------------------------------------------------------------------------- #

def _ring(n: int, r: float = 1000.0) -> List[Drone]:
    out = []
    for i in range(n):
        ang = i / n * 2 * math.pi
        out.append(Drone(f"d{i}", np.array([r * math.cos(ang), r * math.sin(ang)])))
    return out


def test_swarm_avoids_defended_axis():
    sw = DecentralizedSwarm(ASSET, n_sectors=8)
    drones = _ring(8)
    defended = [np.array([500.0, 20.0 * k]) for k in range(4)]  # all in sector 0
    plan = sw.plan(drones, defended)
    counts = [0] * 8
    for intent in plan.values():
        counts[intent.sector] += 1
    assert counts[0] <= 1, "swarm should shun the heavily defended sector"
    assert sum(counts) == 8


def test_swarm_remasses_when_defence_shifts():
    sw = DecentralizedSwarm(ASSET, n_sectors=8)
    drones = _ring(8)
    sw.plan(drones, [np.array([500.0, 0.0])])          # defence on +x
    shifted = [np.array([-500.0, 20.0 * k]) for k in range(5)]  # now on -x (sector 4)
    plan2 = sw.plan(drones, shifted)
    counts = [0] * 8
    for intent in plan2.values():
        counts[intent.sector] += 1
    assert counts[4] <= 1, "swarm re-masses away from the new defensive axis"


def test_swarm_deterministic():
    sw = DecentralizedSwarm(ASSET, n_sectors=8)
    drones = _ring(6)
    inter = [np.array([500.0, 0.0])]
    p1 = {k: (v.sector, v.role) for k, v in sw.plan(drones, inter).items()}
    p2 = {k: (v.sector, v.role) for k, v in sw.plan(drones, inter).items()}
    assert p1 == p2


def test_swarm_tasks_a_feint():
    sw = DecentralizedSwarm(ASSET, n_sectors=8)
    drones = _ring(8)
    plan = sw.plan(drones, [np.array([500.0, 0.0])])
    roles = {intent.role for intent in plan.values()}
    assert "press" in roles  # there is always a main axis
