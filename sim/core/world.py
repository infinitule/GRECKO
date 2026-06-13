"""World: the simulation's state container and step loop.

Entity-component style: World holds plain data; systems (kinematics, intercept
detection) are called as pure-function passes each tick.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

import numpy as np

from sim.core.entities import Asset, HostileUAS, Interceptor
from sim.core.events import EventLog
from sim.core.kinematics import (
    hostile_desired_heading,
    interceptor_desired_heading,
    step_entity,
)
from sim.core.vec import norm


INTERCEPT_RADIUS = 15.0   # metres — within this, an interceptor kills a UAS
ASSET_IMPACT_RADIUS = 10.0  # metres — UAS reaching this damages the asset


class World:
    def __init__(
        self,
        dt: float,
        rng: np.random.Generator,
        asset: Asset,
        log: EventLog,
    ):
        self.dt = dt
        self.rng = rng
        self.asset = asset
        self.log = log
        self.t: float = 0.0
        self.hostiles: Dict[str, HostileUAS] = {}
        self.interceptors: Dict[str, Interceptor] = {}
        self._interceptor_assignments: Dict[str, Optional[str]] = {}

    def spawn_hostile(self, h: HostileUAS) -> None:
        self.hostiles[h.id] = h
        self.log.append(
            self.t, "ENTITY_SPAWN",
            entity_id=h.id, entity_type="HostileUAS",
            position=h.pos, velocity=h.vel, heading=h.heading,
            data={"speed": h.speed, "weave_amplitude": h.weave_amplitude},
        )

    def spawn_interceptor(self, iv: Interceptor) -> None:
        self.interceptors[iv.id] = iv
        self._interceptor_assignments[iv.id] = None
        self.log.append(
            self.t, "ENTITY_SPAWN",
            entity_id=iv.id, entity_type="Interceptor",
            position=iv.pos, velocity=iv.vel, heading=iv.heading,
            data={"speed": iv.speed, "effector_type": iv.effector_type},
        )

    def assign(self, interceptor_id: str, hostile_id: Optional[str]) -> None:
        """Simple pass-through assignment; upper layers (allocator) call this."""
        self._interceptor_assignments[interceptor_id] = hostile_id

    def step(self) -> None:
        """Advance the world by one dt."""
        self._step_hostiles()
        self._step_interceptors()
        self._check_intercepts()
        self._check_asset_impacts()
        self.t += self.dt

    def _step_hostiles(self) -> None:
        for h in list(self.hostiles.values()):
            if not h.alive:
                continue
            desired, h.waypoints = hostile_desired_heading(
                h.pos, h.waypoints, h.heading,
                h.weave_amplitude, h.weave_period, self.t,
            )
            h.pos, h.vel, h.heading = step_entity(
                h.pos, h.vel, h.heading, h.speed, h.max_turn_rate, desired, self.dt,
            )

    def _step_interceptors(self) -> None:
        for iv in list(self.interceptors.values()):
            if not iv.alive:
                continue
            iv.endurance -= self.dt
            if iv.endurance <= 0:
                iv.alive = False
                self.log.append(
                    self.t, "ENTITY_DESPAWN",
                    entity_id=iv.id, entity_type="Interceptor",
                    position=iv.pos, data={"reason": "endurance_exhausted"},
                )
                continue

            target_id = self._interceptor_assignments.get(iv.id)
            target = self.hostiles.get(target_id) if target_id else None
            t_pos = target.pos if (target and target.alive) else None
            t_vel = target.vel if (target and target.alive) else None

            desired = interceptor_desired_heading(iv.pos, t_pos, t_vel, iv.speed, self.dt)
            if t_pos is None:
                desired = iv.heading

            iv.pos, iv.vel, iv.heading = step_entity(
                iv.pos, iv.vel, iv.heading, iv.speed, iv.max_turn_rate, desired, self.dt,
            )

    def _check_intercepts(self) -> None:
        for iv in self.interceptors.values():
            if not iv.alive:
                continue
            for h in list(self.hostiles.values()):
                if not h.alive:
                    continue
                if norm(iv.pos - h.pos) <= INTERCEPT_RADIUS:
                    h.alive = False
                    self.log.append(
                        self.t, "INTERCEPT",
                        entity_id=h.id, entity_type="HostileUAS",
                        position=h.pos, velocity=h.vel, heading=h.heading,
                        data={"interceptor_id": iv.id},
                    )
                    self.log.append(
                        self.t, "ENTITY_DESPAWN",
                        entity_id=h.id, entity_type="HostileUAS",
                        position=h.pos, data={"reason": "intercepted"},
                    )
                    # interceptor is also consumed
                    iv.alive = False
                    self.log.append(
                        self.t, "ENTITY_DESPAWN",
                        entity_id=iv.id, entity_type="Interceptor",
                        position=iv.pos, data={"reason": "engagement"},
                    )
                    break

    def _check_asset_impacts(self) -> None:
        for h in list(self.hostiles.values()):
            if not h.alive:
                continue
            if norm(h.pos - self.asset.pos) <= ASSET_IMPACT_RADIUS:
                h.alive = False
                self.asset.hp -= 1.0
                self.log.append(
                    self.t, "LEAKER",
                    entity_id=h.id, entity_type="HostileUAS",
                    position=h.pos, data={"asset_id": self.asset.id, "asset_hp_remaining": self.asset.hp},
                )
                self.log.append(
                    self.t, "ASSET_DAMAGE",
                    entity_id=self.asset.id, entity_type="Asset",
                    position=self.asset.pos,
                    data={"hp_remaining": self.asset.hp, "leaker_id": h.id},
                )
                self.log.append(
                    self.t, "ENTITY_DESPAWN",
                    entity_id=h.id, entity_type="HostileUAS",
                    position=h.pos, data={"reason": "impact"},
                )
                if self.asset.hp <= 0:
                    self.asset.alive = False

    def is_engagement_over(self) -> bool:
        alive_hostiles = [h for h in self.hostiles.values() if h.alive]
        return len(alive_hostiles) == 0

    def summary(self) -> Dict:
        intercepts = sum(1 for e in self.log.events if e["type"] == "INTERCEPT")
        leakers = sum(1 for e in self.log.events if e["type"] == "LEAKER")
        return {
            "intercepts": intercepts,
            "leakers": leakers,
            "asset_hp": self.asset.hp,
            "time_to_clear": round(self.t, 3),
            "total_events": len(self.log.events),
        }

    def log_hash(self) -> str:
        """SHA-256 of the canonical JSONL — the determinism acceptance criterion."""
        jsonl = self.log.to_jsonl()
        return hashlib.sha256(jsonl.encode()).hexdigest()
