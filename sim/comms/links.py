"""Abstract link model — the contested EM environment as a first-class input.

SCOPE: this is a probabilistic message-delivery abstraction parameterised by
distance, zones, and time. It deliberately models NO RF physics: no waveforms,
no jammer design, no emitter modelling. Degradation scenarios are config, not
code (see CommsConfig.from_dict and eval/scenarios/comms/*.yaml).

Degradation modes, all composable:
- range collapse : comm_radius shrinks (config value).
- regional denial: circular zones; a link is down if either endpoint is inside
                   a zone OR the line-of-link crosses a zone.
- intermittency  : links flap on a deterministic duty cycle with a per-pair
                   phase offset (so the whole mesh doesn't blink in unison).
- full partition : kill switch — all links down.
- base drop rate : per-message Bernoulli loss on otherwise-up links.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class DenialZone:
    center: Tuple[float, float]
    radius: float

    def contains(self, p: np.ndarray) -> bool:
        return float(np.hypot(p[0] - self.center[0], p[1] - self.center[1])) <= self.radius

    def crosses_segment(self, a: np.ndarray, b: np.ndarray) -> bool:
        """True if segment a-b passes through the zone."""
        c = np.array(self.center, dtype=float)
        ab = b - a
        ab2 = float(ab @ ab)
        if ab2 < 1e-12:
            return self.contains(a)
        s = float(np.clip(((c - a) @ ab) / ab2, 0.0, 1.0))
        closest = a + s * ab
        return float(np.linalg.norm(c - closest)) <= self.radius


@dataclasses.dataclass
class CommsConfig:
    comm_radius: float = 1500.0
    base_drop_rate: float = 0.0          # Bernoulli loss on up-links
    latency: float = 0.05                # seconds, delivery delay
    denial_zones: List[DenialZone] = dataclasses.field(default_factory=list)
    intermittency_period: float = 0.0    # seconds; 0 disables flapping
    intermittency_duty: float = 1.0      # fraction of period the link is up
    full_partition: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "CommsConfig":
        zones = [DenialZone(tuple(z["center"]), float(z["radius"]))
                 for z in d.get("denial_zones", [])]
        return cls(
            comm_radius=float(d.get("comm_radius", 1500.0)),
            base_drop_rate=float(d.get("base_drop_rate", 0.0)),
            latency=float(d.get("latency", 0.05)),
            denial_zones=zones,
            intermittency_period=float(d.get("intermittency_period", 0.0)),
            intermittency_duty=float(d.get("intermittency_duty", 1.0)),
            full_partition=bool(d.get("full_partition", False)),
        )


def _pair_phase(a: str, b: str) -> float:
    """Deterministic per-pair phase in [0, 1) so flapping is desynchronised."""
    key = "|".join(sorted((a, b))).encode()
    h = hashlib.sha256(key).digest()
    return int.from_bytes(h[:4], "big") / 2 ** 32


class LinkModel:
    """Pure predicate: is the link between two named, positioned nodes up at t?"""

    def __init__(self, config: CommsConfig):
        self.config = config

    def link_up(self, t: float, id_a: str, pos_a: np.ndarray, id_b: str, pos_b: np.ndarray) -> bool:
        cfg = self.config
        if cfg.full_partition:
            return False
        if float(np.linalg.norm(pos_a - pos_b)) > cfg.comm_radius:
            return False
        for z in cfg.denial_zones:
            if z.contains(pos_a) or z.contains(pos_b) or z.crosses_segment(pos_a, pos_b):
                return False
        if cfg.intermittency_period > 0:
            phase = (t / cfg.intermittency_period + _pair_phase(id_a, id_b)) % 1.0
            if phase >= cfg.intermittency_duty:
                return False
        return True
