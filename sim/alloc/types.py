"""Shared data types for the allocation layer.

Assignment mirrors /proto/assignment.schema.json.
MagazineState tracks remaining rounds per effector type.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional


@dataclasses.dataclass
class Provenance:
    solver: str
    bid_value: float
    track_value_estimate: float
    magazine_state: Dict[str, int]
    round: int
    hold_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Assignment:
    t: float
    interceptor_id: str
    action: str                  # "ASSIGN" | "HOLD_FIRE" | "RTB"
    track_id: Optional[str]
    effector_id: Optional[str]
    provenance: Provenance

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "interceptor_id": self.interceptor_id,
            "action": self.action,
            "track_id": self.track_id,
            "effector_id": self.effector_id,
            "provenance": self.provenance.to_dict(),
        }


@dataclasses.dataclass
class MagazineState:
    rounds: Dict[str, int]       # effector_id -> remaining rounds

    def can_fire(self, effector_id: str) -> bool:
        return self.rounds.get(effector_id, 0) > 0

    def expend(self, effector_id: str) -> None:
        self.rounds[effector_id] = max(0, self.rounds.get(effector_id, 0) - 1)

    def copy(self) -> "MagazineState":
        return MagazineState(dict(self.rounds))

    def to_dict(self) -> dict:
        return dict(self.rounds)
