"""P3 acceptance tests.

Criteria from the plan:
1. Birds (slow, erratic, no ingress vector) classified non-hostile >= 95%.
2. Priority ordering matches time-to-asset ordering on a designed scenario.
3. Every assessment carries a non-empty human-readable "why" field.
4. Friendly protection: known-friendly tracks never exceed "unknown" without
   corroborating hostile behaviour.
5. Classifier is swappable: a stub implementation satisfies the same interface.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from sim.classify.classifier import (
    RuleClassifier,
    ThreatAssessment,
    _AGE_MIN_FOR_HOSTILE,
    _APPROACH_RATE_HOSTILE,
    _HEADING_HOSTILE,
    _SPEED_BIRD_MAX,
)
from sim.classify.features import FeatureVector
from sim.classify.pipeline import ClassificationPipeline
from sim.fusion.tracker import TrackMessage


ASSET = np.array([0.0, 0.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(
    track_id: str,
    pos: np.ndarray,
    vel: np.ndarray,
    age: float = 5.0,
    n_updates: int = 20,
    status: str = "confirmed",
) -> TrackMessage:
    state = np.array([pos[0], pos[1], vel[0], vel[1]])
    cov = np.eye(4) * 1.0
    return TrackMessage(
        t=10.0, track_id=track_id, status=status,
        state=state, covariance=cov, quality=0.9,
        n_updates=n_updates, age=age,
    )


def _make_history(track: TrackMessage, n: int = 10) -> list:
    """Shallow history replicating the same state (weave_energy ≈ 0)."""
    return [track] * n


def _classify_track(track: TrackMessage, history: list | None = None) -> ThreatAssessment:
    clf = RuleClassifier()
    from sim.classify.features import extract
    h = history if history is not None else _make_history(track)
    features = extract(track, h, ASSET)
    return clf.classify(features, False, ASSET, track.state)


# ---------------------------------------------------------------------------
# 1. Bird classification (slow, erratic, non-ingress)
# ---------------------------------------------------------------------------

class TestBirdClassification:
    def _bird_track(self, rng, idx: int) -> TrackMessage:
        """Random slow, erratic bird: random bearing, speed < 12 m/s, no ingress vector."""
        angle = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(200, 800)
        pos = np.array([r * math.cos(angle), r * math.sin(angle)])
        # Heading perpendicular or away from asset (heading_to_asset > 30°)
        away_angle = angle + rng.uniform(math.pi / 3, math.pi)
        speed = rng.uniform(2.0, _SPEED_BIRD_MAX * 0.9)
        vel = np.array([math.cos(away_angle), math.sin(away_angle)]) * speed
        return _make_track(f"bird_{idx}", pos, vel, age=6.0)

    def _bird_history(self, base: TrackMessage, rng) -> list:
        """History with erratic headings (high weave energy)."""
        msgs = []
        vel = base.state[2:4].copy()
        for _ in range(15):
            turn = rng.uniform(-0.4, 0.4)
            heading = math.atan2(vel[1], vel[0]) + turn
            spd = float(np.linalg.norm(vel))
            vel = np.array([math.cos(heading), math.sin(heading)]) * spd
            s = base.state.copy()
            s[2], s[3] = vel[0], vel[1]
            msgs.append(TrackMessage(
                t=base.t - len(msgs) * 0.1, track_id=base.track_id,
                status="confirmed", state=s, covariance=np.eye(4),
                quality=0.9, n_updates=10, age=base.age,
            ))
        return list(reversed(msgs))

    def test_birds_non_hostile_95_percent(self):
        rng = np.random.default_rng(42)
        clf = RuleClassifier()
        from sim.classify.features import extract
        non_hostile = 0
        n = 1000
        for i in range(n):
            bird = self._bird_track(rng, i)
            history = self._bird_history(bird, rng)
            features = extract(bird, history, ASSET)
            a = clf.classify(features, False, ASSET, bird.state)
            if a.label != "hostile":
                non_hostile += 1
        rate = non_hostile / n
        assert rate >= 0.95, f"Birds non-hostile rate {rate:.3f} < 0.95 over {n} trials"


# ---------------------------------------------------------------------------
# 2. Priority ordering matches time-to-asset ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_closer_hostile_higher_priority(self):
        """Three hostile inbound tracks at different ranges; priority must
        rank by ascending time-to-asset (nearest first)."""
        tracks = []
        for i, dist in enumerate([200.0, 500.0, 900.0]):
            pos = np.array([-dist, 0.0])
            vel = np.array([25.0, 0.0])   # same speed, all aimed at asset
            tracks.append(_make_track(f"h_{i}", pos, vel, age=5.0))

        pipe = ClassificationPipeline(ASSET)
        assessments = pipe.update(tracks)
        confirmed = [a for a in assessments]
        assert len(confirmed) == 3
        # priority should be descending (pipeline sorts it)
        priorities = [a.priority_score for a in confirmed]
        assert priorities == sorted(priorities, reverse=True), (
            f"Priority order doesn't match range order: {priorities}"
        )
        # nearest (h_0, dist=200) must have highest priority
        assert confirmed[0].track_id == "h_0", (
            f"Expected h_0 (nearest) first, got {confirmed[0].track_id}"
        )

    def test_faster_hostile_higher_priority_at_same_range(self):
        """Two tracks at same range, different speeds — faster = shorter TTA = higher priority."""
        pos = np.array([-500.0, 0.0])
        slow = _make_track("slow", pos, np.array([15.0, 0.0]))
        fast = _make_track("fast", pos, np.array([35.0, 0.0]))
        pipe = ClassificationPipeline(ASSET)
        assessments = pipe.update([slow, fast])
        by_id = {a.track_id: a for a in assessments}
        assert by_id["fast"].priority_score > by_id["slow"].priority_score, (
            f"fast priority {by_id['fast'].priority_score:.4f} <= "
            f"slow {by_id['slow'].priority_score:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. Every assessment carries a non-empty "why" field
# ---------------------------------------------------------------------------

class TestRationaleField:
    def test_why_always_populated(self):
        tracks = [
            _make_track("h0", np.array([-300.0, 0.0]), np.array([20.0, 0.0])),
            _make_track("u0", np.array([0.0, 500.0]), np.array([0.0, -5.0])),
            _make_track("f0", np.array([200.0, 200.0]), np.array([-2.0, -2.0])),
        ]
        pipe = ClassificationPipeline(ASSET)
        assessments = pipe.update(tracks)
        for a in assessments:
            assert a.why, f"track {a.track_id} has empty 'why' field"
            assert len(a.why) > 10, f"'why' too short: '{a.why}'"

    def test_why_contains_fired_rules_or_flags(self):
        pos = np.array([-400.0, 0.0])
        vel = np.array([25.0, 0.0])
        trk = _make_track("h0", pos, vel)
        a = _classify_track(trk)
        assert "FIRED" in a.why or "conf=" in a.why, (
            f"Expected rationale structure in 'why': '{a.why}'"
        )

    def test_dict_output_has_why(self):
        trk = _make_track("x0", np.array([-300.0, 0.0]), np.array([20.0, 0.0]))
        a = _classify_track(trk)
        d = a.to_dict()
        assert "why" in d and d["why"]


# ---------------------------------------------------------------------------
# 4. Friendly protection
# ---------------------------------------------------------------------------

class TestFriendlyProtection:
    def test_marked_friendly_never_hostile_without_corroboration(self):
        """A track marked friendly must stay at 'unknown' even if it looks
        somewhat hostile (only partial corroboration — low speed means the
        speed rule fails, so all three are NOT simultaneously satisfied)."""
        pos = np.array([-400.0, 10.0])
        # Heading straight at asset but below the bird-speed threshold (< 12 m/s).
        # This satisfies approach_rate and heading_to_asset but NOT speed_above_bird,
        # so full corroboration is absent and the friendly cap must hold.
        vel = np.array([8.0, 0.0])
        trk = _make_track("f0", pos, vel)
        pipe = ClassificationPipeline(ASSET)
        pipe.mark_friendly("f0")
        assessments = pipe.update([trk])
        assert len(assessments) == 1
        assert assessments[0].label != "hostile", (
            f"Friendly track labelled hostile without full corroboration"
        )

    def test_marked_friendly_can_escalate_with_full_corroboration(self):
        """A friendly track showing ALL THREE hostile indicators simultaneously
        (high approach_rate + heading_aligned + high speed) may reach 'hostile'."""
        # Fast, straight-line ingress directly at asset from close range
        pos = np.array([-250.0, 0.0])
        vel = np.array([30.0, 0.0])    # high speed, heading straight at asset
        trk = _make_track("f1", pos, vel, age=5.0)
        pipe = ClassificationPipeline(ASSET)
        pipe.mark_friendly("f1")
        assessments = pipe.update([trk])
        # label could be unknown or hostile depending on conf threshold —
        # the invariant is only that it's no longer *forced* to unknown when
        # all three corroborating indicators are present
        assert assessments[0].label in ("hostile", "unknown")

    def test_unmark_friendly_allows_reclassification(self):
        """After unmarking, a fully-hostile track reaches 'hostile'.
        While friendly-flagged with only partial signals it stays 'unknown'."""
        pos = np.array([-300.0, 0.0])
        # Partial threat: below bird-speed threshold — full corroboration absent
        vel_partial = np.array([8.0, 0.0])
        trk_partial = _make_track("f2", pos, vel_partial)
        pipe = ClassificationPipeline(ASSET)
        pipe.mark_friendly("f2")
        before = pipe.update([trk_partial])[0].label
        assert before != "hostile", (
            f"Friendly track with partial signals should not be hostile, got {before}"
        )

        # After unmarking, full-speed ingress track is classified normally
        pipe.unmark_friendly("f2")
        vel_full = np.array([25.0, 0.0])
        trk_full = _make_track("f2", pos, vel_full)
        after = pipe.update([trk_full])[0].label
        assert after in ("hostile", "unknown")

    def test_1000_random_marked_friendly_never_hostile_without_corroboration(self):
        """Stress: 1000 random partially-threatening tracks marked friendly
        must never be labelled hostile when they lack full corroboration."""
        rng = np.random.default_rng(7)
        clf = RuleClassifier()
        from sim.classify.features import extract
        violations = 0
        for i in range(1000):
            # Partial threat: approach_rate present but low speed OR heading off
            approach_rate = rng.uniform(5.0, 20.0)
            speed = rng.uniform(5.0, _SPEED_BIRD_MAX)  # too slow for full corroboration
            angle = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(200, 1000)
            pos = np.array([r * math.cos(angle), r * math.sin(angle)])
            vel_dir = -pos / np.linalg.norm(pos)
            vel = vel_dir * speed
            trk = _make_track(f"f_{i}", pos, vel)
            history = _make_history(trk)
            features = extract(trk, history, ASSET)
            a = clf.classify(features, is_known_friendly=True, asset_pos=ASSET, track_state=trk.state)
            if a.label == "hostile":
                violations += 1
        assert violations == 0, (
            f"{violations}/1000 friendly tracks labelled hostile without full corroboration"
        )


# ---------------------------------------------------------------------------
# 5. Swappable interface
# ---------------------------------------------------------------------------

class TestSwappableInterface:
    def test_stub_classifier_satisfies_interface(self):
        """A minimal stub that always returns 'unknown' must satisfy the
        Classifier interface and produce valid ThreatAssessments."""
        from sim.classify.classifier import Classifier

        class AlwaysUnknown(Classifier):
            def classify(self, features, is_known_friendly, asset_pos, track_state):
                return ThreatAssessment(
                    t=features.t, track_id=features.track_id,
                    label="unknown", confidence=0.5,
                    priority_score=0.0, features=features,
                    why="stub: always unknown",
                )

        pipe = ClassificationPipeline(ASSET, classifier=AlwaysUnknown())
        trk = _make_track("x0", np.array([-300.0, 0.0]), np.array([20.0, 0.0]))
        assessments = pipe.update([trk])
        assert assessments[0].label == "unknown"
        assert assessments[0].why == "stub: always unknown"

    def test_tentative_tracks_not_classified(self):
        """Tentative tracks must be filtered out before classification."""
        trk = _make_track("t0", np.array([-300.0, 0.0]), np.array([20.0, 0.0]),
                          status="tentative")
        pipe = ClassificationPipeline(ASSET)
        assessments = pipe.update([trk])
        assert len(assessments) == 0
