"""Pillar D — decentralized swarm-on-swarm coordination.

Where Pillar A (`sim/alloc`) solves the allocation problem with a single
central solver that sees the whole air picture, Pillar D distributes the same
decision across autonomous agents that coordinate peer-to-peer over the
contested comms mesh. Each agent assesses the threat locally, claims the target
it is best placed to neutralize, and reconciles conflicts with its neighbours
by consensus — no central node.

SCOPE: simulation and research only. Agents exchange abstract claim/report
messages over the `sim/comms` topology; nothing here designs RF, controls
hardware, or computes a launch solution. Effectors remain parameter sets, and
every engagement intent this layer produces is still subordinate to the
human-on-the-loop C2 interlock (`sim/bridge/state.py::C2State.can_engage`).

Public surface:
  DecentralizedDefense  — Allocator drop-in; defender interceptors coordinate.
  DecentralizedSwarm    — attacker-side coordinator; drones re-mass on the
                          least-defended axis peer-to-peer.
  DroneAgent            — a single autonomous node's local reasoning.
"""
from __future__ import annotations

from sim.swarm.agent import DroneAgent, LocalPick
from sim.swarm.consensus import ConsensusOutcome, resolve
from sim.swarm.defense import DecentralizedDefense
from sim.swarm.messages import Claim, Release, ThreatReport
from sim.swarm.swarm import DecentralizedSwarm, SwarmIntent

__all__ = [
    "DroneAgent",
    "LocalPick",
    "ConsensusOutcome",
    "resolve",
    "DecentralizedDefense",
    "Claim",
    "Release",
    "ThreatReport",
    "DecentralizedSwarm",
    "SwarmIntent",
]
