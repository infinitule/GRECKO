"""IntentPredictor — the deployment wrapper Pillar A and the C2 console use.

Consumes track states (positions + velocities — fusion estimates, never
truth), clusters them, runs the model, and emits SwarmIntent messages per
/proto/intent.schema.json, including the trajectory forecast and the
value_multiplier that wires into the allocator's v_j.

Also provides KinematicsHeuristic: the heading-only baseline the acceptance
test measures lead time against.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional

import numpy as np

from learn.intent.doctrines import INTENT_CLASSES
from learn.intent.features import cluster_agents, cluster_features
from learn.intent.model import IntentMLP

FORECAST_HORIZON_S = 10.0
FORECAST_SIGMA_GROWTH = 4.0   # m of 1-sigma uncertainty added per second


@dataclasses.dataclass
class SwarmIntent:
    """Mirrors /proto/intent.schema.json."""
    t: float
    cluster_id: str
    member_track_ids: List[str]
    intent_distribution: Dict[str, float]
    forecast_centroids: np.ndarray     # (H, 2)
    forecast_sigma: np.ndarray         # (H,)
    value_multiplier: float

    def dominant_intent(self) -> str:
        return max(self.intent_distribution, key=self.intent_distribution.get)

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "cluster_id": self.cluster_id,
            "member_track_ids": self.member_track_ids,
            "intent_distribution": self.intent_distribution,
            "forecast": {
                "horizon_s": FORECAST_HORIZON_S,
                "centroid_positions": self.forecast_centroids.tolist(),
                "sigma": self.forecast_sigma.tolist(),
            },
            "value_multiplier": self.value_multiplier,
        }


def _value_multiplier(dist: Dict[str, float]) -> float:
    """v_j adjustment: amplify main-axis tracks, attenuate confirmed feints.
    m = 1 + 0.8*P(main) - 0.9*P(feint) - 0.5*P(isr) - 0.5*P(reserve),
    clipped to [0.1, 2.0]."""
    m = (1.0
         + 0.8 * dist["main_axis"]
         - 0.9 * dist["feint"]
         - 0.5 * dist["isr"]
         - 0.5 * dist["reserve"])
    return float(np.clip(m, 0.1, 2.0))


class IntentPredictor:
    def __init__(self, model: IntentMLP, asset_pos: np.ndarray):
        self.model = model
        self.asset_pos = asset_pos

    def predict(
        self,
        t: float,
        track_ids: List[str],
        positions: np.ndarray,      # (n, 2) fusion estimates
        velocities: np.ndarray,     # (n, 2)
    ) -> List[SwarmIntent]:
        if len(track_ids) == 0:
            return []
        clusters = cluster_agents(positions, velocities)
        out: List[SwarmIntent] = []
        for ci, members in enumerate(clusters):
            feats = cluster_features(positions, velocities, members,
                                     self.asset_pos, len(track_ids))
            probs = self.model.predict_proba(feats)[0]
            dist = {c: float(p) for c, p in zip(INTENT_CLASSES, probs)}

            centroid = positions[members].mean(axis=0)
            cvel = velocities[members].mean(axis=0)
            steps = int(FORECAST_HORIZON_S)
            centroids = np.array([centroid + cvel * (s + 1) for s in range(steps)])
            sigma = np.array([(s + 1) * FORECAST_SIGMA_GROWTH for s in range(steps)])

            out.append(SwarmIntent(
                t=t,
                cluster_id=f"C{ci}",
                member_track_ids=[track_ids[m] for m in members],
                intent_distribution=dist,
                forecast_centroids=centroids,
                forecast_sigma=sigma,
                value_multiplier=_value_multiplier(dist),
            ))
        return out


class KinematicsHeuristic:
    """Heading-only baseline: the cluster with the smallest mean heading
    error to the asset is called main_axis; everything else unknown.
    This is what 'current best practice' can do without collective inference."""

    def __init__(self, asset_pos: np.ndarray):
        self.asset_pos = asset_pos

    def main_axis_cluster(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> Optional[List[int]]:
        clusters = cluster_agents(positions, velocities)
        best, best_err = None, math.pi
        for members in clusters:
            if len(members) < 2:
                continue
            errs = []
            for m in members:
                v = velocities[m]
                s = float(np.linalg.norm(v))
                if s < 0.5:
                    errs.append(math.pi)
                    continue
                rel = self.asset_pos - positions[m]
                d = float(np.linalg.norm(rel))
                if d < 1.0:
                    errs.append(0.0)
                    continue
                cos_a = float(np.clip(np.dot(v / s, rel / d), -1, 1))
                errs.append(math.acos(cos_a))
            mean_err = float(np.mean(errs))
            if mean_err < best_err:
                best_err = mean_err
                best = members
        return best
