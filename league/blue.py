"""BluePolicy — the defense parameters mutual co-evolution adapts.

PILLAR C (PL) evolved Red against a *fixed* Blue. Mutual co-evolution
(PM) lets Blue adapt back: it chooses an interceptor effector loadout and a
cost-rationing knob, then Red counter-evolves against that Blue.

A Blue policy is deliberately small and interpretable:

  lambda_cost  cost-exchange rationing knob in [0, 0.5]; higher = ration more
  loadout      effector_type per interceptor (3 of them), chosen from a cheap
               -> expensive menu. This is the dominant cost lever: a kinetic
               interceptor costs $90k, a net drone $3k, a collision drone $800.

The optimisation question: can Blue cut its cost-per-intercept by fielding
cheaper effectors and rationing fire, without letting leakers blow up?

SCOPE: effectors are parameter sets (cost, Pk). Choosing a loadout is a
resource-allocation decision in the sim, not a hardware action.
"""
from __future__ import annotations

import dataclasses
import itertools
from typing import List, Tuple

import numpy as np

from sim.alloc.types import MagazineState

N_INTERCEPTORS = 3

# Effector menu Blue may field, cheapest first. ew_soft_kill is excluded: it is
# soft-kill RF-only (Pk=0 vs quadrotor) and useless against the kinetic probe.
LOADOUT_MENU: Tuple[str, ...] = (
    "collision_drone",      # $800
    "net_capture_drone",    # $3,000
    "kinetic_interceptor",  # $90,000
)

LAMBDA_CHOICES: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5)

# Per-effector magazine depth (generous; not the binding constraint for the
# small probe, but kept consistent so can_fire() never spuriously blocks).
_MAG_DEPTH = {
    "collision_drone": 12,
    "net_capture_drone": 12,
    "kinetic_interceptor": 12,
    "ew_soft_kill": 12,
}


@dataclasses.dataclass(frozen=True)
class BluePolicy:
    lambda_cost: float = 0.05
    loadout: Tuple[str, ...] = ("kinetic_interceptor", "net_capture_drone",
                                "net_capture_drone")

    def __post_init__(self) -> None:
        if len(self.loadout) != N_INTERCEPTORS:
            raise ValueError(
                f"loadout must have {N_INTERCEPTORS} entries, got {len(self.loadout)}")
        for eff in self.loadout:
            if eff not in LOADOUT_MENU:
                raise ValueError(f"effector not in menu: {eff!r}")

    @classmethod
    def default(cls) -> "BluePolicy":
        """The fixed Blue the PL league evolved against."""
        return cls()

    @classmethod
    def random(cls, rng: np.random.Generator) -> "BluePolicy":
        lam = float(rng.choice(LAMBDA_CHOICES))
        loadout = tuple(rng.choice(LOADOUT_MENU) for _ in range(N_INTERCEPTORS))
        return cls(lambda_cost=lam, loadout=loadout)

    def loadout_list(self) -> List[str]:
        return list(self.loadout)

    def magazine(self) -> MagazineState:
        """A magazine that covers every effector type (all 4) so the allocator
        can always fall back; depth is uniform and non-binding."""
        return MagazineState(dict(_MAG_DEPTH))

    def to_dict(self) -> dict:
        return {"lambda_cost": float(self.lambda_cost),
                "loadout": list(self.loadout)}

    @classmethod
    def from_dict(cls, d: dict) -> "BluePolicy":
        return cls(lambda_cost=float(d["lambda_cost"]),
                   loadout=tuple(d["loadout"]))


def enumerate_blue_policies(lambda_choices: Tuple[float, ...] = LAMBDA_CHOICES,
                            menu: Tuple[str, ...] = LOADOUT_MENU,
                            unordered_loadout: bool = True) -> List[BluePolicy]:
    """All Blue policies in the search space.

    With unordered_loadout, loadouts that are permutations of each other (same
    multiset of effectors over interchangeable interceptor slots) are collapsed
    to one representative — the interceptor positions are near-symmetric, so
    permutations are redundant and only inflate the search.
    """
    if unordered_loadout:
        loadouts = [tuple(sorted(c)) for c in
                    itertools.combinations_with_replacement(menu, N_INTERCEPTORS)]
    else:
        loadouts = list(itertools.product(menu, repeat=N_INTERCEPTORS))
    return [BluePolicy(lambda_cost=lam, loadout=lo)
            for lam in lambda_choices for lo in loadouts]
