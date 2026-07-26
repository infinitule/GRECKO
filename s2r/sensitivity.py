"""One-at-a-time (OAT) sensitivity sweep + tornado ranking.

For each RealityGap dimension, run the probe with that dimension at its
worst-case bound (all others nominal) and compare the engagement margin
against the shared nominal baseline. The tornado ranking orders dimensions
by impact — it tells the validation campaign which gates to fund first.

The margin statistic is the minimum over a small seed set, so the ranking
reflects worst-case behaviour rather than single-seed luck.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

from s2r.episodes import margin_over_seeds
from s2r.gap import GAP_DIMS, GAP_ENVELOPE, RealityGap

SWEEP_SEEDS: Tuple[int, ...] = (0, 1, 2)


@dataclasses.dataclass
class DimSensitivity:
    gap_dim: str
    worst_value: float
    nominal_margin_m: float      # engagement margin at nominal fidelity
    worst_margin_m: float        # margin with this dim at its worst bound
    delta_m: float               # worst - nominal (negative = margin lost)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SensitivityReport:
    seeds: Tuple[int, ...]
    nominal_margin_m: float
    per_dim: Dict[str, DimSensitivity]

    def tornado(self) -> List[DimSensitivity]:
        """Dimensions ranked by |delta|, largest impact first."""
        return sorted(self.per_dim.values(),
                      key=lambda d: abs(d.delta_m), reverse=True)

    def insensitive_dims(self, tol_m: float = 1.0) -> List[str]:
        """Dimensions the sim currently does not respond to at all.

        These are fidelity LIMITATIONS, not robustness: a zero delta means
        the sim cannot tell us what this parameter costs, so its gate must
        pass before any margin conclusion crosses the reality gap.
        """
        return [d.gap_dim for d in self.per_dim.values()
                if abs(d.delta_m) < tol_m]

    def to_dict(self) -> dict:
        return {
            "seeds": list(self.seeds),
            "nominal_margin_m": self.nominal_margin_m,
            "per_dim": {k: v.to_dict() for k, v in self.per_dim.items()},
            "tornado_order": [d.gap_dim for d in self.tornado()],
            "insensitive_dims": self.insensitive_dims(),
        }


def oat_sweep(seeds: Tuple[int, ...] = SWEEP_SEEDS) -> SensitivityReport:
    """Run nominal once, then each dimension at its worst bound."""
    nominal_margin = margin_over_seeds(RealityGap.nominal(), seeds=seeds)

    per_dim: Dict[str, DimSensitivity] = {}
    for dim in GAP_DIMS:
        worst = GAP_ENVELOPE[dim][1]
        worst_margin = margin_over_seeds(RealityGap.single(dim, worst),
                                         seeds=seeds)
        per_dim[dim] = DimSensitivity(
            gap_dim=dim,
            worst_value=worst,
            nominal_margin_m=nominal_margin,
            worst_margin_m=worst_margin,
            delta_m=round(worst_margin - nominal_margin, 2),
        )

    return SensitivityReport(seeds=seeds,
                             nominal_margin_m=nominal_margin,
                             per_dim=per_dim)
