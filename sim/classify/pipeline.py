"""Classification pipeline: tracks in -> assessments out.

Maintains per-track history windows for feature extraction, manages
operator-set friendly flags, and runs the pluggable classifier.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set

import numpy as np

from sim.classify.classifier import Classifier, RuleClassifier, ThreatAssessment
from sim.classify.features import extract
from sim.fusion.tracker import TrackMessage


_HISTORY_LEN = 20   # TrackMessages kept per track for weave-energy computation


class ClassificationPipeline:
    def __init__(
        self,
        asset_pos: np.ndarray,
        classifier: Classifier | None = None,
        sensor_type_hints: Dict[str, str] | None = None,
    ):
        self.asset_pos = asset_pos
        self.classifier: Classifier = classifier or RuleClassifier()
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=_HISTORY_LEN))
        self._friendly_flags: Set[str] = set()
        self._sensor_type_hints: Dict[str, str] = sensor_type_hints or {}

    def mark_friendly(self, track_id: str) -> None:
        self._friendly_flags.add(track_id)

    def unmark_friendly(self, track_id: str) -> None:
        self._friendly_flags.discard(track_id)

    def update(self, tracks: List[TrackMessage]) -> List[ThreatAssessment]:
        assessments: List[ThreatAssessment] = []
        for trk in tracks:
            if trk.status == "tentative":
                continue  # only classify confirmed/coasted tracks
            self._history[trk.track_id].append(trk)
            history = list(self._history[trk.track_id])
            hint = self._sensor_type_hints.get(trk.track_id, "")
            features = extract(trk, history, self.asset_pos, hint)
            is_friendly = trk.track_id in self._friendly_flags
            assessment = self.classifier.classify(
                features, is_friendly, self.asset_pos, trk.state
            )
            assessments.append(assessment)

        # Sort by priority descending — the allocator consumes this ordering
        assessments.sort(key=lambda a: a.priority_score, reverse=True)
        return assessments
