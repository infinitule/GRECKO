"""Threat classifier and priority scorer.

`Classifier` is the swappable interface. `RuleClassifier` is the v1
transparent rule/score-based implementation. A learned model slots in by
implementing the same interface — the allocator and C2 console depend only
on `ThreatAssessment`, never on internals.

Explainability is an architectural property here, not a feature: every
assessment must carry a human-readable `why` string. This doubles as the
audit trail the HOTL console displays.
"""
from __future__ import annotations

import abc
import dataclasses
import math
from typing import Dict, List

import numpy as np

from sim.classify.features import FeatureVector


# ---------------------------------------------------------------------------
# Output message (mirrors /proto/threat_assessment.schema.json)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ThreatAssessment:
    t: float
    track_id: str
    label: str          # "hostile" | "unknown" | "friendly"
    confidence: float   # [0, 1]
    priority_score: float  # higher = more urgent; dominant term = 1/tta
    features: FeatureVector
    why: str            # required non-empty rationale

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "priority_score": round(self.priority_score, 4),
            "features": dataclasses.asdict(self.features),
            "why": self.why,
        }


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class Classifier(abc.ABC):
    @abc.abstractmethod
    def classify(
        self,
        features: FeatureVector,
        is_known_friendly: bool,
        asset_pos: np.ndarray,
        track_state: np.ndarray,
    ) -> ThreatAssessment:
        """Map a FeatureVector to a ThreatAssessment.

        `is_known_friendly`: set by the C2 operator (MARK FRIENDLY action);
        once set, label is capped at "unknown" unless corroborating hostile
        behaviour is observed (heading + speed + approach rate all hostile).
        """


# ---------------------------------------------------------------------------
# v1 Rule-based implementation
# ---------------------------------------------------------------------------

# Thresholds — exposed as class-level constants so they appear in the ADR
# and can be tuned per-scenario without touching logic.
_APPROACH_RATE_HOSTILE = 10.0   # m/s; closing at less than this is not threatening
_HEADING_HOSTILE = 0.52         # radians (~30°); heading within this of asset dir
_SPEED_BIRD_MAX = 12.0          # m/s; below this, could be a bird
_WEAVE_BIRD_MIN = 0.015         # rad^2/s^2; erratic enough to be a bird
_AGE_MIN_FOR_HOSTILE = 1.5      # s; too-new track can't be confirmed hostile


class RuleClassifier(Classifier):
    """Transparent scored rule classifier.

    Scoring: each rule contributes a weighted boolean score in [0, 1].
    Final confidence = clipped weighted mean. Rationale string lists which
    rules fired.
    """

    def classify(
        self,
        features: FeatureVector,
        is_known_friendly: bool,
        asset_pos: np.ndarray,
        track_state: np.ndarray,
    ) -> ThreatAssessment:

        rules_fired: List[str] = []
        rules_failed: List[str] = []
        score = 0.0
        weight_total = 0.0

        def _rule(name: str, fired: bool, weight: float) -> None:
            nonlocal score, weight_total
            weight_total += weight
            if fired:
                score += weight
                rules_fired.append(name)
            else:
                rules_failed.append(name)

        # --- feature rules ---
        approaching = features.approach_rate >= _APPROACH_RATE_HOSTILE
        _rule("approach_rate", approaching, weight=3.0)

        heading_aligned = features.heading_to_asset <= _HEADING_HOSTILE
        _rule("heading_to_asset", heading_aligned, weight=2.5)

        not_bird_speed = features.speed > _SPEED_BIRD_MAX
        _rule("speed_above_bird", not_bird_speed, weight=1.5)

        not_bird_weave = features.weave_energy < _WEAVE_BIRD_MIN
        _rule("low_weave_energy", not_bird_weave, weight=1.0)

        mature_track = features.track_age >= _AGE_MIN_FOR_HOSTILE
        _rule("track_maturity", mature_track, weight=0.5)

        confidence = score / weight_total if weight_total > 0 else 0.0

        # --- label decision ---
        if confidence >= 0.65:
            label = "hostile"
        elif confidence <= 0.25:
            label = "unknown"
        else:
            label = "unknown"

        # Friendly protection: a known-friendly track is capped at "unknown"
        # UNLESS all three primary hostile indicators fire simultaneously
        # (approach_rate AND heading_to_asset AND speed — corroborating hostile
        # behaviour that overrides the friendly flag).
        if is_known_friendly:
            corroborating = approaching and heading_aligned and not_bird_speed
            if not corroborating:
                label = "unknown"
                rules_fired = [r for r in rules_fired] + ["[friendly_flag]"]
                confidence = min(confidence, 0.40)

        # --- priority score ---
        pos = track_state[:2]
        vel = track_state[2:4]
        asset_dist = float(np.linalg.norm(asset_pos - pos))
        speed = features.speed
        if speed > 1e-6 and features.approach_rate > 0:
            tta = asset_dist / features.approach_rate
        elif asset_dist < 1.0:
            tta = 0.001
        else:
            tta = 9999.0

        # priority = (1/tta) * confidence * hostile_multiplier
        hostile_mult = 1.0 if label == "hostile" else (0.5 if label == "unknown" else 0.1)
        priority = (1.0 / max(tta, 0.5)) * confidence * hostile_mult

        # --- rationale ---
        why_parts = []
        if rules_fired:
            why_parts.append(f"FIRED: {', '.join(rules_fired)}")
        if rules_failed:
            why_parts.append(f"NOT_FIRED: {', '.join(rules_failed)}")
        why_parts.append(
            f"conf={confidence:.2f} tta={tta:.0f}s "
            f"approach={features.approach_rate:.1f}m/s "
            f"hdg_err={math.degrees(features.heading_to_asset):.0f}°"
        )
        why = " | ".join(why_parts)

        return ThreatAssessment(
            t=features.t,
            track_id=features.track_id,
            label=label,
            confidence=float(confidence),
            priority_score=float(priority),
            features=features,
            why=why,
        )
