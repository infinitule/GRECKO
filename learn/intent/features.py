"""Clustering and per-cluster feature extraction for the intent model.

The swarm is treated as a graph: agents are nodes, proximity/velocity
similarity define edges (single-linkage). Each connected cluster yields one
feature vector summarising collective behaviour — the lightweight stand-in
for the graph-attention encoder (ADR-005b records the upgrade path).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

# Single-linkage thresholds: two agents are connected if both hold
LINK_DIST = 500.0       # m
LINK_VEL = 15.0         # m/s velocity-difference threshold

N_FEATURES = 10


def cluster_agents(positions: np.ndarray, velocities: np.ndarray) -> List[List[int]]:
    """Single-linkage clustering on proximity AND velocity similarity.
    Returns list of clusters, each a list of agent indices."""
    n = positions.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) <= LINK_DIST and \
               np.linalg.norm(velocities[i] - velocities[j]) <= LINK_VEL:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: -len(g))


def cluster_features(
    positions: np.ndarray,
    velocities: np.ndarray,
    member_idx: List[int],
    asset_pos: np.ndarray,
    total_observed: int,
) -> np.ndarray:
    """Feature vector for one cluster. All features are scale-normalised."""
    p = positions[member_idx]
    v = velocities[member_idx]
    n = len(member_idx)

    centroid = p.mean(axis=0)
    rel = asset_pos - centroid
    rng_to_asset = float(np.linalg.norm(rel))
    asset_dir = rel / max(rng_to_asset, 1.0)

    speeds = np.linalg.norm(v, axis=1)
    mean_speed = float(speeds.mean())
    speed_std = float(speeds.std())

    # heading error to asset per member
    errs = []
    for k in range(n):
        s = speeds[k]
        if s < 0.5:
            errs.append(math.pi)
            continue
        cos_a = float(np.clip(np.dot(v[k] / s, asset_dir), -1, 1))
        errs.append(math.acos(cos_a))
    mean_hdg_err = float(np.mean(errs))
    hdg_err_std = float(np.std(errs))

    # formation tightness: mean distance to centroid
    tightness = float(np.linalg.norm(p - centroid, axis=1).mean()) if n > 1 else 0.0

    # closing rate of the centroid
    centroid_vel = v.mean(axis=0)
    closing = float(np.dot(centroid_vel, asset_dir))

    return np.array([
        n / max(total_observed, 1),       # relative cluster size
        min(n / 12.0, 1.0),               # absolute size, saturating
        mean_speed / 30.0,
        speed_std / 10.0,
        mean_hdg_err / math.pi,
        hdg_err_std / math.pi,
        min(tightness / 300.0, 1.0),
        rng_to_asset / 4000.0,
        closing / 30.0,
        float(np.linalg.norm(centroid_vel)) / 30.0,
    ])


def observe(
    trajectories: np.ndarray,
    step: int,
    dropout: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partial observation at one time step: randomly occlude a fraction of
    agents (the observation-dropout mechanism). Returns
    (positions, velocities, visible_indices)."""
    n = trajectories.shape[0]
    visible = np.array([i for i in range(n) if rng.random() >= dropout])
    if len(visible) == 0:
        visible = np.array([int(rng.integers(0, n))])
    pos = trajectories[visible, step, :2]
    vel = trajectories[visible, step, 2:]
    return pos, vel, visible
