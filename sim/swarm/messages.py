"""Peer message types exchanged over the comms mesh during consensus.

These mirror the abstract envelope carried by `sim/comms/network.py`: the
coordinator serialises them into `Envelope.payload` dicts when it runs over the
live bus, and uses them directly in the fast in-process consensus path. They
carry no control information — only what a node believes and what it intends to
claim — so a dropped or delayed message degrades coordination quality, never
safety (the C2 interlock is downstream of everything here).
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ThreatReport:
    """`src` reports seeing `track_id` at the given threat severity.

    Severity is the classifier's priority_score (dominant term 1/tta): it is a
    threat-urgency estimate, not an economic quantity. Reports let an agent
    build a shared picture of threats it cannot yet see itself.
    """
    src: str
    track_id: str
    severity: float
    t: float


@dataclasses.dataclass(frozen=True)
class Claim:
    """`src` intends to engage `track_id`.

    The claim carries the two quantities consensus arbitrates on, in priority
    order:
      capability — P_k(best feasible effector) × urgency the agent can apply;
                   "how well, and how soon, can I neutralize this threat".
      cost       — dollar cost of that effector; the SECONDARY tiebreak.

    Threat-primary rule: when two agents claim the same track, the higher
    capability wins; cost only breaks a capability tie. Money is a tiebreak,
    not the driver.
    """
    src: str
    track_id: str
    capability: float
    cost: float
    t: float


@dataclasses.dataclass(frozen=True)
class Release:
    """`src` withdraws its claim on `track_id` (it lost the consensus)."""
    src: str
    track_id: str
    t: float
