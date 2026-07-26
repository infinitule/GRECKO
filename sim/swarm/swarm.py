"""DecentralizedSwarm — Pillar D attacker-side coordinator.

The mirror of DecentralizedDefense on the Red team. Attacking drones have no
central controller either: each one senses where the defence is thin, and the
swarm coordinates peer-to-peer to concentrate mass on the least-defended axis —
re-massing on the fly as drones are intercepted.

This complements the co-evolution league (Pillar C). The league evolves attack
*formations* between episodes; DecentralizedSwarm adapts *within* an episode, so
the two compose: the league discovers a good opening, the swarm re-solves the
approach live as the defence commits.

Model (abstract, kinematics-free): the approach is discretised into K angular
sectors around the protected asset. Each drone estimates defensive pressure per
sector from the interceptor bearings it can see, shares that belief with comms
neighbours, and the swarm distributes itself across sectors by water-filling
toward the gaps. The heaviest sector becomes the main press; a thin sector is
tasked as a feint to split the defence.

SCOPE: tactical geometry and coordination only — no guidance law, no control
surface, no hardware. Sectors and bearings are analysis constructs.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class SwarmIntent:
    drone_id: str
    sector: int
    bearing: float           # radians, sector centre relative to asset
    role: str                # "press" (main axis) | "feint" (defence-splitter)


@dataclasses.dataclass
class Drone:
    drone_id: str
    pos: np.ndarray          # 2D position
    alive: bool = True


class DecentralizedSwarm:
    def __init__(self, asset_pos: Sequence[float], n_sectors: int = 8) -> None:
        self.asset = np.asarray(asset_pos, dtype=float)
        self.k = int(n_sectors)

    # -- geometry helpers ----------------------------------------------------

    def _bearing(self, pos: np.ndarray) -> float:
        d = pos[:2] - self.asset
        return math.atan2(d[1], d[0]) % (2 * math.pi)

    def _sector_of(self, bearing: float) -> int:
        return int(bearing / (2 * math.pi) * self.k) % self.k

    def _sector_centre(self, s: int) -> float:
        return (s + 0.5) * (2 * math.pi) / self.k

    # -- defensive pressure the swarm perceives ------------------------------

    def pressure(self, interceptor_pos: Sequence[np.ndarray]) -> List[int]:
        """Count interceptors whose bearing falls in each sector."""
        p = [0] * self.k
        for ip in interceptor_pos:
            p[self._sector_of(self._bearing(np.asarray(ip, dtype=float)))] += 1
        return p

    # -- the coordinated plan ------------------------------------------------

    def plan(
        self,
        drones: Sequence[Drone],
        interceptor_pos: Sequence[np.ndarray],
    ) -> Dict[str, SwarmIntent]:
        """Distribute living drones across sectors, massing on the gaps.

        Water-filling: sectors are filled in ascending order of defensive
        pressure, so the least-defended axis attracts the most drones. Within
        that, each drone is assigned to keep its bearing change small (drones
        drift to the nearest desirable sector, not an arbitrary one), which is
        the behaviour a real decentralized swarm would settle into.
        """
        living = [d for d in drones if d.alive]
        if not living:
            return {}

        pressure = self.pressure(interceptor_pos)
        # Target count per sector by water-filling toward low pressure.
        target = self._water_fill(len(living), pressure)

        # Assign each drone to a sector with remaining capacity, preferring the
        # sector nearest its current bearing (stable, low-churn).
        remaining = dict(enumerate(target))
        order = sorted(living, key=lambda d: d.drone_id)
        assign: Dict[str, int] = {}
        for d in order:
            b = self._bearing(d.pos)
            cand = [s for s in range(self.k) if remaining.get(s, 0) > 0]
            if not cand:
                cand = list(range(self.k))
            # nearest desirable sector by angular distance, then lowest pressure
            s = min(cand, key=lambda s: (self._ang_dist(b, self._sector_centre(s)),
                                         pressure[s], s))
            assign[d.drone_id] = s
            remaining[s] = remaining.get(s, 0) - 1

        # Role: the most-massed sector is the main press; a single thin,
        # low-pressure sector with at least one drone is tasked as a feint.
        counts = [0] * self.k
        for s in assign.values():
            counts[s] += 1
        main_sector = max(range(self.k), key=lambda s: (counts[s], -pressure[s]))
        feint_sector = self._pick_feint(counts, pressure, main_sector)

        out: Dict[str, SwarmIntent] = {}
        for did, s in assign.items():
            role = "press"
            if s == feint_sector and s != main_sector:
                role = "feint"
            out[did] = SwarmIntent(
                drone_id=did, sector=s, bearing=self._sector_centre(s), role=role
            )
        return out

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _ang_dist(a: float, b: float) -> float:
        d = abs(a - b) % (2 * math.pi)
        return min(d, 2 * math.pi - d)

    def _water_fill(self, n: int, pressure: List[int]) -> List[int]:
        """Allocate n drones across sectors, one at a time, always to the sector
        whose (pressure + drones already placed) is currently lowest."""
        placed = [0] * self.k
        for _ in range(n):
            s = min(range(self.k), key=lambda s: (pressure[s] + placed[s], s))
            placed[s] += 1
        return placed

    @staticmethod
    def _pick_feint(counts: List[int], pressure: List[int], main: int) -> Optional[int]:
        cand = [s for s in range(len(counts)) if counts[s] > 0 and s != main]
        if not cand:
            return None
        # thinnest, least-defended sector = the cheapest place to split defence
        return min(cand, key=lambda s: (counts[s], pressure[s], s))
