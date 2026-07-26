"""Threat-primary conflict resolution — the arbitration rule.

When two agents claim the same track, exactly one must keep it. The order the
project chose:

  1. capability  (higher wins)  — who can best neutralize the threat, soonest.
  2. cost        (lower wins)   — economics, the SECONDARY tiebreak.
  3. agent id    (lower wins)   — a deterministic final tiebreak so replays are
                                  byte-identical regardless of message order.

This is deliberately *not* stock CBBA, where the money bid is the primary key.
Here the tactical question ("who stops this best") leads; the dollar figure only
separates agents who are equally capable. That keeps the coordinator's behaviour
aligned with threat response first, cost-exchange second.
"""
from __future__ import annotations

import dataclasses

from sim.swarm.messages import Claim

# Capability values within this fraction of each other are treated as a tie, so
# floating-point noise never decides an engagement — cost does.
_CAPABILITY_EPS = 1e-6


@dataclasses.dataclass(frozen=True)
class ConsensusOutcome:
    winner: str
    loser: str
    track_id: str
    reason: str


def _key(c: Claim):
    """Sort key implementing the threat-primary order.

    Negate capability so that Python's ascending sort puts the strongest claim
    first; cost ascends (cheaper preferred); id ascends (deterministic).
    """
    return (-c.capability, c.cost, c.src)


def resolve(a: Claim, b: Claim) -> ConsensusOutcome:
    """Decide which of two competing claims on the same track survives."""
    if a.track_id != b.track_id:
        raise ValueError("resolve() only arbitrates claims on the same track")

    cap_gap = abs(a.capability - b.capability)
    if cap_gap > _CAPABILITY_EPS:
        primary = "capability"
    elif abs(a.cost - b.cost) > 1e-9:
        primary = "cost (capability tied)"
    else:
        primary = "agent-id (capability & cost tied)"

    win, lose = (a, b) if _key(a) <= _key(b) else (b, a)
    return ConsensusOutcome(
        winner=win.src,
        loser=lose.src,
        track_id=win.track_id,
        reason=f"{win.src} beat {lose.src} on {primary}",
    )


def best_claim(claims):
    """Return the single surviving Claim among any number of claims on one track."""
    claims = list(claims)
    if not claims:
        return None
    return min(claims, key=_key)
