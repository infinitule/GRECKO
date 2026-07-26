"""RealityGap — the bounded perturbation envelope between sim and reality.

Each dimension is a single scalar that degrades one modelled assumption:
multiplicative scales sit at 1.0 when the sim is taken at face value,
additive offsets sit at 0.0. The envelope bounds are deliberately one-sided
toward pessimism — real sensors underperform their spec sheet, real links
drop more packets, real threats fly faster than the threat library assumed.

`apply_to_mesh` / `make_comms_cfg` build perturbed copies; nothing in /sim
is mutated. The POSG invariant is untouched: perturbation changes how truth
is *observed*, never who may read it.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import numpy as np

from sim.bridge.scenario import default_sensor_mesh
from sim.comms.links import CommsConfig
from sim.sensing.sensors import SensorMesh

# dimension -> (nominal, worst) ; nominal is "trust the sim as-is"
GAP_ENVELOPE: Dict[str, Tuple[float, float]] = {
    "pd_scale":           (1.0, 0.60),   # sensors detect worse than spec
    "noise_scale":        (1.0, 2.50),   # measurement scatter larger than spec
    "clutter_scale":      (1.0, 3.00),   # more false alarms in the real clutter env
    "latency_add_s":      (0.0, 0.30),   # extra report latency (processing, network)
    "update_rate_scale":  (1.0, 0.50),   # sustained scan rate below datasheet
    "comms_drop_add":     (0.0, 0.30),   # extra link loss probability
    "comms_radius_scale": (1.0, 0.50),   # usable link range below modelled radius
    "target_speed_scale": (1.0, 1.30),   # threats faster than the threat library
}

N_GAP_DIMS = len(GAP_ENVELOPE)
GAP_DIMS = tuple(GAP_ENVELOPE.keys())


def _bounds(dim: str) -> Tuple[float, float]:
    nominal, worst = GAP_ENVELOPE[dim]
    return (min(nominal, worst), max(nominal, worst))


@dataclasses.dataclass(frozen=True)
class RealityGap:
    """One point inside the perturbation envelope."""
    pd_scale: float = 1.0
    noise_scale: float = 1.0
    clutter_scale: float = 1.0
    latency_add_s: float = 0.0
    update_rate_scale: float = 1.0
    comms_drop_add: float = 0.0
    comms_radius_scale: float = 1.0
    target_speed_scale: float = 1.0

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def nominal(cls) -> "RealityGap":
        return cls()

    @classmethod
    def worst_case(cls) -> "RealityGap":
        return cls(**{dim: GAP_ENVELOPE[dim][1] for dim in GAP_DIMS})

    @classmethod
    def single(cls, dim: str, value: float) -> "RealityGap":
        """Perturb one dimension, all others nominal (for OAT sweeps)."""
        if dim not in GAP_ENVELOPE:
            raise KeyError(f"unknown gap dimension: {dim!r}")
        lo, hi = _bounds(dim)
        if not (lo - 1e-9 <= value <= hi + 1e-9):
            raise ValueError(f"{dim}={value} outside envelope [{lo}, {hi}]")
        return cls(**{dim: value})

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "RealityGap":
        """Domain randomization: uniform draw inside the envelope."""
        values = {}
        for dim in GAP_DIMS:
            lo, hi = _bounds(dim)
            values[dim] = float(rng.uniform(lo, hi))
        return cls(**values)

    # ------------------------------------------------------------------ #
    # Application                                                         #
    # ------------------------------------------------------------------ #

    def apply_to_mesh(self, mesh: SensorMesh | None = None) -> SensorMesh:
        """Return a perturbed copy of the (default) bridge sensor mesh."""
        if mesh is None:
            mesh = default_sensor_mesh()
        perturbed = []
        for s in mesh.sensors:
            perturbed.append(dataclasses.replace(
                s,
                pd_max=min(1.0, s.pd_max * self.pd_scale),
                noise_std=s.noise_std * self.noise_scale,
                false_alarm_rate=s.false_alarm_rate * self.clutter_scale,
                latency=s.latency + self.latency_add_s,
                update_rate=max(0.1, s.update_rate * self.update_rate_scale),
            ))
        return SensorMesh(perturbed)

    def make_comms_cfg(self) -> CommsConfig:
        """Perturbed copy of the bridge's nominal comms configuration."""
        return CommsConfig(
            comm_radius=2000.0 * self.comms_radius_scale,
            base_drop_rate=min(1.0, 0.0 + self.comms_drop_add),
        )

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {dim: float(getattr(self, dim)) for dim in GAP_DIMS}

    def is_within_envelope(self) -> bool:
        for dim in GAP_DIMS:
            lo, hi = _bounds(dim)
            v = float(getattr(self, dim))
            if not (lo - 1e-9 <= v <= hi + 1e-9):
                return False
        return True
