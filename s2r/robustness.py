"""Domain-randomized conclusion-stability study.

Samples K RealityGaps uniformly from the envelope and re-runs the probe
under each. The headline conclusion ("the defense intercepts the raid with
at least MARGIN_THRESHOLD_M of standoff") HOLDS under a gap if
margin_m >= threshold.

The deliverable is the stability fraction: in what share of the plausible
reality envelope does the conclusion survive? A conclusion that only holds
at nominal fidelity is a simulation artefact, not a result.
"""
from __future__ import annotations

from typing import List

import numpy as np

from s2r.episodes import MARGIN_THRESHOLD_M, run_probe_episode
from s2r.gap import RealityGap


def robustness_study(n_samples: int = 8, seed: int = 0,
                     threshold: float = MARGIN_THRESHOLD_M) -> dict:
    """Run the probe under n_samples random gaps; report stability."""
    rng = np.random.default_rng(seed)
    gaps = [RealityGap.sample(rng) for _ in range(n_samples)]

    results: List[dict] = []
    for i, gap in enumerate(gaps):
        r = run_probe_episode(gap, seed=seed + i)
        r["holds"] = bool(r["margin_m"] >= threshold)
        results.append(r)

    holds = [r for r in results if r["holds"]]
    fails = [r for r in results if not r["holds"]]
    worst = min(results, key=lambda r: r["margin_m"])

    return {
        "seed": seed,
        "n_samples": n_samples,
        "threshold_m": threshold,
        "stability_fraction": round(len(holds) / max(n_samples, 1), 4),
        "worst_case": worst,
        "failing_gaps": [r["gap"] for r in fails],
        "results": results,
    }
