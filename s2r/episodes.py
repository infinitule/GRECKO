"""Probe engagement under a perturbed world.

A single fixed attack (the "probe") is run against the full Blue stack with
the sensor mesh and comms config perturbed by a RealityGap. Holding the
attack constant isolates the effect of the gap: any change in outcome is
attributable to the perturbation, not to red-team variation.

Headline metric — engagement margin: the closest distance to the asset at
which any intercept occurs. Interceptors pursue in truth space once
assigned, so raw intercept COUNT is insensitive to sensing fidelity; what
sensing degradation actually costs is reaction time, which shows up as
intercepts happening closer to the asset. Margin is the standoff the
defense preserved — the quantity a real deployment cares about.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np

from league.policy import SwarmPolicy
from s2r.gap import RealityGap
from sim.bridge.scenario import BridgeScenario

MAX_PROBE_TIME = 25.0     # s — main group arrives at ~27 s, so the window
                          # covers the whole engageable approach
MARGIN_THRESHOLD_M = 100.0  # headline holds if every intercept is at least
                            # this far from the asset (standoff/debris margin)

# Fixed probe attack: 5 UAS (4 main + 1 feint) from the north at 600 m.
# Sized to the bridge blue team (3 consumable interceptors): contested but
# winnable at nominal fidelity, with margin to lose under degradation.
_PROBE_THETA = np.array([
    5.0,            # n_total
    0.20,           # feint_frac  -> 1 feint, 4 main
    0.0,            # screen_frac
    np.pi / 2,      # main_angle (north)
    np.pi / 2,      # feint_offset
    22.0,           # main_speed
    14.0,           # feint_speed
    600.0,          # main_range
    480.0,          # feint_range
    10.0,           # t_feint_turn
    60.0,           # main_spread
    80.0,           # feint_spread
    0.05,           # weave_amp
    0.0,            # timing_offset
])


def probe_policy(gap: RealityGap) -> SwarmPolicy:
    """The fixed probe attack, with only target_speed_scale applied."""
    theta = _PROBE_THETA.copy()
    theta[5] *= gap.target_speed_scale   # main_speed
    theta[6] *= gap.target_speed_scale   # feint_speed
    return SwarmPolicy(theta=theta)


def run_probe_episode(gap: RealityGap, seed: int = 0,
                      loadout: Optional[list] = None) -> dict:
    """Run the probe under `gap`; return outcome metrics.

    margin_m is the headline: min intercept distance from the asset
    (0.0 if no intercept happened at all).

    `loadout` (optional) sets the Blue effector per interceptor, so the adapted
    Blue from mutual co-evolution can be re-validated across the reality gap.
    """
    sc = BridgeScenario(
        seed=seed,
        auto_authorize=True,
        policy=probe_policy(gap),
        sensor_mesh=gap.apply_to_mesh(),
        comms_cfg=gap.make_comms_cfg(),
        loadout=loadout,
    )
    while not sc.world.is_engagement_over() and sc.world.t < MAX_PROBE_TIME:
        sc.tick()

    summary = sc.world.summary()
    intercept_ranges = [
        float(np.linalg.norm(e["position"]))
        for e in sc.world.log.events if e["type"] == "INTERCEPT"
    ]
    n_threats = len(sc.world.hostiles)
    intercepts = int(summary["intercepts"])
    return {
        "gap": gap.to_dict(),
        "seed": seed,
        "n_threats": n_threats,
        "intercepts": intercepts,
        "leakers": int(summary["leakers"]),
        "asset_hp": float(summary["asset_hp"]),
        "time_s": round(float(sc.world.t), 3),
        "margin_m": round(min(intercept_ranges), 2) if intercept_ranges else 0.0,
        "mean_intercept_range_m": (
            round(float(np.mean(intercept_ranges)), 2) if intercept_ranges else 0.0
        ),
        "log_hash": sc.log_hash(),
    }


def margin_over_seeds(gap: RealityGap, seeds=(0, 1, 2)) -> float:
    """Worst (minimum) engagement margin across seeds — the robust statistic
    used by the sensitivity sweep so single-seed noise doesn't drive ranking."""
    return min(run_probe_episode(gap, seed=s)["margin_m"] for s in seeds)


def result_hash(result: dict) -> str:
    """SHA-256 over the canonical JSON of a result (determinism criterion)."""
    canon = json.dumps(result, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()
