"""C2State — the human-authorised engagement envelope.

This is the ONLY path through which terminal engagements enter the World.
Nothing in the allocator, classifier, or fusion pipeline reads or writes it.
All C2 decisions generate audit events so the trail is complete.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Set


@dataclasses.dataclass
class AuditEntry:
    sim_t: float
    wall_t: float
    event: str
    interceptor_id: str = ""
    track_id: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class C2State:
    """Operator engagement envelope.

    Mutated ONLY by the bridge server in response to C2 commands from the
    console.  The allocator, classifier, and fusion pipeline never touch it.
    """

    def __init__(self, lambda_cost: float = 0.05) -> None:
        self.weapons_hold: bool = False
        self.authorized_tracks: Set[str] = set()
        self.held_tracks: Set[str] = set()
        self.friendly_marked: Set[str] = set()
        self.lambda_cost: float = lambda_cost
        self._audit: List[AuditEntry] = []

    # ------------------------------------------------------------------ #
    # Operator actions (called from bridge server on receipt of C2 cmd)   #
    # ------------------------------------------------------------------ #

    def authorize(self, track_id: str, sim_t: float = 0.0) -> None:
        self.authorized_tracks.add(track_id)
        self.held_tracks.discard(track_id)
        self._log(sim_t, "AUTHORIZE", track_id=track_id)

    def hold(self, track_id: str, sim_t: float = 0.0) -> None:
        self.held_tracks.add(track_id)
        self.authorized_tracks.discard(track_id)
        self._log(sim_t, "OPERATOR_HOLD", track_id=track_id)

    def mark_friendly(self, track_id: str, sim_t: float = 0.0) -> None:
        self.friendly_marked.add(track_id)
        self.authorized_tracks.discard(track_id)
        self.held_tracks.add(track_id)
        self._log(sim_t, "MARK_FRIENDLY", track_id=track_id)

    def lift_hold(self, track_id: str, sim_t: float = 0.0) -> None:
        self.held_tracks.discard(track_id)
        self._log(sim_t, "LIFT_HOLD", track_id=track_id)

    def set_weapons_hold(self, active: bool, sim_t: float = 0.0) -> None:
        changed = self.weapons_hold != active
        self.weapons_hold = active
        if changed:
            event = "WEAPONS_HOLD_ON" if active else "WEAPONS_HOLD_OFF"
            self._log(sim_t, event)

    def set_lambda(self, value: float) -> None:
        self.lambda_cost = float(max(0.0, min(1.0, value)))

    # ------------------------------------------------------------------ #
    # Engagement predicate (called by bridge scenario each tick)          #
    # ------------------------------------------------------------------ #

    def can_engage(self, track_id: str) -> bool:
        """True only if global weapons hold is off AND track is authorized."""
        if self.weapons_hold:
            return False
        if track_id in self.friendly_marked:
            return False
        if track_id in self.held_tracks:
            return False
        return track_id in self.authorized_tracks

    # ------------------------------------------------------------------ #
    # Audit logging helpers (called by bridge scenario on each decision)  #
    # ------------------------------------------------------------------ #

    def log_engage(self, sim_t: float, interceptor_id: str, track_id: str) -> None:
        self._log(sim_t, "AUTHORIZED_ENGAGE",
                  interceptor_id=interceptor_id, track_id=track_id)

    def log_hold_pending(self, sim_t: float, interceptor_id: str,
                         track_id: str) -> None:
        self._log(sim_t, "HOLD_PENDING_AUTH",
                  interceptor_id=interceptor_id, track_id=track_id)

    def log_weapons_hold_block(self, sim_t: float, interceptor_id: str,
                               track_id: str) -> None:
        self._log(sim_t, "WEAPONS_HOLD_ACTIVE",
                  interceptor_id=interceptor_id, track_id=track_id)

    def log_operator_hold_block(self, sim_t: float, interceptor_id: str,
                                track_id: str) -> None:
        self._log(sim_t, "OPERATOR_HOLD_BLOCK",
                  interceptor_id=interceptor_id, track_id=track_id)

    # ------------------------------------------------------------------ #
    # Apply a JSON command dict (from WebSocket client)                   #
    # ------------------------------------------------------------------ #

    def apply_command(self, cmd: dict, sim_t: float = 0.0) -> bool:
        """Dispatch a C2 command dict.  Returns True if recognised."""
        t = cmd.get("type", "")
        tid = cmd.get("track_id", "")
        if t == "AUTHORIZE":
            self.authorize(tid, sim_t)
        elif t == "HOLD":
            self.hold(tid, sim_t)
        elif t == "MARK_FRIENDLY":
            self.mark_friendly(tid, sim_t)
        elif t == "LIFT_HOLD":
            self.lift_hold(tid, sim_t)
        elif t == "WEAPONS_HOLD":
            self.set_weapons_hold(bool(cmd.get("active", True)), sim_t)
        elif t == "SET_LAMBDA":
            self.set_lambda(float(cmd.get("value", 0.05)))
        else:
            return False
        return True

    # ------------------------------------------------------------------ #

    @property
    def audit_trail(self) -> List[dict]:
        return [e.to_dict() for e in self._audit]

    def recent_audit(self, n: int = 50) -> List[dict]:
        return [e.to_dict() for e in self._audit[-n:]]

    def _log(self, sim_t: float, event: str, **kwargs) -> None:
        self._audit.append(AuditEntry(
            sim_t=sim_t,
            wall_t=time.time(),
            event=event,
            **kwargs,
        ))
