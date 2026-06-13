"""Behavioral diversity metrics for the league population.

Behavioral fingerprint: 10-dimensional vector capturing the attack
pattern characteristics that matter for distinctness:
  [n_total/16, feint_frac, screen_frac, cos(main_angle), sin(main_angle),
   main_speed/35, feint_speed/22, t_feint_turn/45, main_range/3500,
   feint_range/2800]

Two policies are considered behaviorally distinct if their fingerprint
L2-distance > MIN_DISTINCT_DISTANCE (default 0.15 in the unit hypercube).
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from league.policy import SwarmPolicy

MIN_DISTINCT_DISTANCE = 0.15


def behavioral_fingerprint(policy: SwarmPolicy) -> np.ndarray:
    """Return a normalised 10-d fingerprint vector for this policy."""
    return np.array([
        policy.n_total / 16.0,
        policy.feint_frac,
        policy.screen_frac,
        math.cos(policy.main_angle),
        math.sin(policy.main_angle),
        policy.main_speed / 35.0,
        policy.feint_speed / 22.0,
        policy.t_feint_turn / 45.0,
        policy.main_range / 3500.0,
        policy.feint_range / 2800.0,
    ], dtype=float)


def population_diversity(policies: List[SwarmPolicy]) -> float:
    """Mean pairwise L2 distance between behavioral fingerprints."""
    if len(policies) < 2:
        return 0.0
    fps = [behavioral_fingerprint(p) for p in policies]
    n = len(fps)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(np.array(fps[i]) - np.array(fps[j])))
            count += 1
    return total / count if count > 0 else 0.0


def distinct_clusters(
    policies: List[SwarmPolicy],
    min_distance: float = MIN_DISTINCT_DISTANCE,
) -> int:
    """Count qualitatively distinct attack patterns via greedy clustering.

    Greedily builds a set of cluster centroids: a policy is placed in a
    new cluster if it is farther than min_distance from all existing
    centroids (in fingerprint space).
    """
    if not policies:
        return 0
    fps = [behavioral_fingerprint(p) for p in policies]
    centroids: List[np.ndarray] = [fps[0]]
    for fp in fps[1:]:
        fp_arr = np.array(fp)
        if all(float(np.linalg.norm(fp_arr - c)) > min_distance
               for c in centroids):
            centroids.append(fp_arr)
    return len(centroids)
