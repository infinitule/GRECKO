"""PB acceptance tests.

Criteria from the plan:
1. On held-out feint+main-axis scenarios with 40% of tracks occluded, the
   model identifies the true main axis EARLIER than a kinematics-only heading
   heuristic (lead-time delta in seconds, reported).
2. Calibration: predicted intent probabilities are well-calibrated (ECE bound).
3. Held-out classification accuracy across all five doctrines.
4. STAGED WIRING: feeding the intent value_multiplier into PA's v_j makes
   EconomicMDP spend less on the feint than with the uniform (stub) prior —
   the delta is recorded.
5. Forecast uncertainty grows with horizon.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pytest

from learn.intent.doctrines import (
    INTENT_CLASSES, N_STEPS, SAMPLE_DT, feint_main_axis, generate_dataset,
)
from learn.intent.features import cluster_agents, observe
from learn.intent.model import IntentMLP, build_training_set
from learn.intent.predictor import IntentPredictor, KinematicsHeuristic

ASSET = np.zeros(2)
EVAL_DROPOUT = 0.4


# ---------------------------------------------------------------------------
# Shared trained model (train once per session — fast, deterministic)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> IntentMLP:
    scenarios = generate_dataset(20, seed=100)          # train seeds
    X, y = build_training_set(scenarios, dropout=0.4, samples_per_scenario=5, seed=101)
    m = IntentMLP(seed=0)
    m.fit(X, y, epochs=300)
    return m


def _holdout_feints(n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    return [feint_main_axis(rng) for _ in range(n)]


def _main_axis_call_time(
    sc, predictor_fn, rng: np.random.Generator,
) -> Optional[float]:
    """Earliest time t such that predictor_fn names the true main-axis cluster
    at t and at every later sample. Returns None if never stably correct."""
    true_main = {i for i, lab in enumerate(sc.labels) if lab == "main_axis"}
    calls: List[Optional[bool]] = []
    times: List[float] = []
    for step in range(4, N_STEPS, 2):
        pos, vel, visible = observe(sc.trajectories, step, EVAL_DROPOUT, rng)
        chosen = predictor_fn(pos, vel)        # set of visible-array indices or None
        if chosen is None:
            calls.append(False)
        else:
            chosen_truth = {int(visible[m]) for m in chosen}
            # correct if majority of chosen agents are truly main-axis
            overlap = len(chosen_truth & true_main) / max(len(chosen_truth), 1)
            calls.append(overlap > 0.5)
        times.append(step * SAMPLE_DT)

    # earliest index from which all subsequent calls are correct
    for k in range(len(calls)):
        if all(calls[k:]):
            return times[k]
    return None


# ---------------------------------------------------------------------------
# 1. Lead-time vs kinematics heuristic — THE acceptance criterion
# ---------------------------------------------------------------------------

class TestLeadTime:
    def test_model_calls_main_axis_earlier_than_heuristic(self, model):
        scenarios = _holdout_feints(12, seed=999)       # held-out seeds
        pred = IntentPredictor(model, ASSET)
        heur = KinematicsHeuristic(ASSET)
        rng = np.random.default_rng(7)

        def model_fn(pos, vel):
            ids = [str(i) for i in range(len(pos))]
            intents = pred.predict(0.0, ids, pos, vel)
            best, best_p = None, 0.0
            for si in intents:
                p = si.intent_distribution["main_axis"]
                if p > best_p:
                    best_p = p
                    best = [int(tid) for tid in si.member_track_ids]
            return best

        def heur_fn(pos, vel):
            return heur.main_axis_cluster(pos, vel)

        deltas = []
        n_model_called, n_heur_called = 0, 0
        for sc in scenarios:
            t_model = _main_axis_call_time(sc, model_fn, rng)
            t_heur = _main_axis_call_time(sc, heur_fn, rng)
            if t_model is not None:
                n_model_called += 1
            if t_heur is not None:
                n_heur_called += 1
            if t_model is not None and t_heur is not None:
                deltas.append(t_heur - t_model)

        assert n_model_called >= 10, (
            f"model stably called main axis in only {n_model_called}/12 held-out scenarios"
        )
        assert deltas, "no scenario where both predictors made a stable call"
        mean_delta = float(np.mean(deltas))
        print(f"\nLEAD-TIME DELTA (heuristic - model): {mean_delta:+.1f} s "
              f"over {len(deltas)} scenarios "
              f"(model called {n_model_called}/12, heuristic {n_heur_called}/12)")
        assert mean_delta > 0.0, (
            f"model is NOT earlier than the kinematics heuristic: delta={mean_delta:+.1f}s"
        )


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_expected_calibration_error_bounded(self, model):
        scenarios = generate_dataset(8, seed=500)       # held-out seeds
        X, y = build_training_set(scenarios, dropout=EVAL_DROPOUT,
                                  samples_per_scenario=5, seed=501)
        probs = model.predict_proba(X)
        conf = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == y).astype(float)

        bins = np.linspace(0.2, 1.0, 9)
        ece, total = 0.0, len(conf)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
            if mask.sum() == 0:
                continue
            gap = abs(conf[mask].mean() - correct[mask].mean())
            ece += (mask.sum() / total) * gap
        print(f"\nECE on held-out set: {ece:.3f} (n={total})")
        assert ece < 0.15, f"Expected calibration error {ece:.3f} >= 0.15"

    def test_holdout_accuracy(self, model):
        scenarios = generate_dataset(8, seed=600)
        X, y = build_training_set(scenarios, dropout=EVAL_DROPOUT,
                                  samples_per_scenario=5, seed=601)
        probs = model.predict_proba(X)
        acc = float((probs.argmax(axis=1) == y).mean())
        print(f"\nHeld-out accuracy at 40% dropout: {acc:.3f} (n={len(y)})")
        assert acc > 0.80, f"held-out accuracy {acc:.3f} <= 0.80"


# ---------------------------------------------------------------------------
# 3. PA wiring — the staged wiring protocol step
# ---------------------------------------------------------------------------

class TestPAWiring:
    def test_intent_multiplier_reduces_feint_engagement(self, model):
        """Re-run a feint+main allocation with (a) stub uniform intent and
        (b) PB intent multipliers wired into v_j. The wired allocator must
        spend fewer rounds on the feint. Delta recorded."""
        from sim.alloc import AllocInput, EconomicMDP, InterceptorState, MagazineState
        from sim.classify.classifier import ThreatAssessment
        from sim.classify.features import FeatureVector
        from sim.effectors.catalogue import CATALOGUE

        rng = np.random.default_rng(123)
        sc = feint_main_axis(rng)
        step = 20   # t=10s — feint still heading in; heading-only can't tell
        pos = sc.trajectories[:, step, :2]
        vel = sc.trajectories[:, step, 2:]
        ids = [f"T{i}" for i in range(sc.n_agents)]

        pred = IntentPredictor(model, ASSET)
        intents = pred.predict(10.0, ids, pos, vel)
        mult = {}
        for si in intents:
            for tid in si.member_track_ids:
                mult[tid] = si.value_multiplier

        def make_assessments(use_intent: bool):
            out = []
            for i, tid in enumerate(ids):
                p = pos[i]; v = vel[i]
                speed = float(np.linalg.norm(v))
                dist = float(np.linalg.norm(ASSET - p))
                closing = float(np.dot(v, (ASSET - p) / max(dist, 1.0)))
                tta = dist / max(closing, 0.1) if closing > 0 else 9999.0
                # v_j scale chosen so that, unweighted, BOTH the feint and the
                # main axis clear the λ engagement threshold — the stub
                # allocator will spend rounds on the feint. The PB multiplier
                # must push the feint back below threshold.
                base = 10.0 / max(tta, 1.0)
                m = mult.get(tid, 1.0) if use_intent else 1.0
                fv = FeatureVector(track_id=tid, t=10.0, pos=p.copy(), vel=v.copy(),
                                   speed=speed, heading_to_asset=0.1,
                                   approach_rate=max(closing, 0.0), weave_energy=0.0,
                                   altitude_band=0, rf_emitter=False,
                                   track_age=10.0, n_updates=20)
                out.append(ThreatAssessment(
                    t=10.0, track_id=tid, label="hostile", confidence=0.9,
                    priority_score=base * m, features=fv, why="test",
                ))
            return sorted(out, key=lambda a: a.priority_score, reverse=True)

        n_iv = 8
        ivs = [InterceptorState(f"i{k}", np.array([float(k * 60 - 200), 0.0]),
                                "kinetic_interceptor", 120.0, 60.0) for k in range(n_iv)]
        adj = {iv.interceptor_id: [o.interceptor_id for o in ivs if o is not iv] for iv in ivs}

        def run(use_intent: bool) -> int:
            mag = MagazineState({"kinetic_interceptor": n_iv, "net_capture_drone": 0,
                                 "ew_soft_kill": 0, "collision_drone": 0})
            inp = AllocInput(t=10.0, interceptors=ivs,
                             assessments=make_assessments(use_intent),
                             magazine=mag, effector_catalogue=CATALOGUE,
                             adjacency=adj, asset_pos=ASSET, lambda_cost=0.3)
            result = EconomicMDP().allocate(inp)
            feint_idx = {ids[i] for i, lab in enumerate(sc.labels) if lab == "feint"}
            return sum(1 for a in result
                       if a.action == "ASSIGN" and a.track_id in feint_idx)

        feint_rounds_stub = run(use_intent=False)
        feint_rounds_wired = run(use_intent=True)
        print(f"\nPA WIRING DELTA: feint rounds {feint_rounds_stub} (stub) -> "
              f"{feint_rounds_wired} (PB wired)")
        assert feint_rounds_stub > 0, (
            "test scenario broken: stub allocator never engaged the feint, "
            "so the wiring delta is vacuous"
        )
        assert feint_rounds_wired < feint_rounds_stub, (
            f"wiring PB intent did not REDUCE feint engagement: "
            f"{feint_rounds_stub} -> {feint_rounds_wired}"
        )

    def test_value_multiplier_direction(self, model):
        """Main-axis clusters get multiplier > 1; feint clusters < 1, on a
        scenario where the model is confident."""
        rng = np.random.default_rng(321)
        sc = feint_main_axis(rng)
        step = 30
        pos = sc.trajectories[:, step, :2]
        vel = sc.trajectories[:, step, 2:]
        ids = [f"T{i}" for i in range(sc.n_agents)]
        pred = IntentPredictor(model, ASSET)
        intents = pred.predict(15.0, ids, pos, vel)

        true_main = {f"T{i}" for i, lab in enumerate(sc.labels) if lab == "main_axis"}
        for si in intents:
            members = set(si.member_track_ids)
            if len(members & true_main) / len(members) > 0.8:
                assert si.value_multiplier > 1.0, (
                    f"main-axis cluster multiplier {si.value_multiplier:.2f} <= 1"
                )
            elif len(members & true_main) == 0 and si.intent_distribution["feint"] > 0.6:
                assert si.value_multiplier < 1.0, (
                    f"confident-feint cluster multiplier {si.value_multiplier:.2f} >= 1"
                )


# ---------------------------------------------------------------------------
# 4. Forecast
# ---------------------------------------------------------------------------

class TestForecast:
    def test_sigma_grows_with_horizon(self, model):
        rng = np.random.default_rng(11)
        sc = feint_main_axis(rng)
        pos = sc.trajectories[:, 10, :2]
        vel = sc.trajectories[:, 10, 2:]
        ids = [f"T{i}" for i in range(sc.n_agents)]
        intents = IntentPredictor(model, ASSET).predict(5.0, ids, pos, vel)
        assert intents
        for si in intents:
            sig = si.forecast_sigma
            assert all(sig[i] < sig[i + 1] for i in range(len(sig) - 1)), (
                "forecast sigma must grow with horizon"
            )

    def test_forecast_tracks_straight_ingress(self, model):
        """On a straight-ingress cluster, the 5-second forecast centroid must
        be close to where the true centroid actually goes."""
        rng = np.random.default_rng(12)
        sc = feint_main_axis(rng)
        step = 20
        pos = sc.trajectories[:, step, :2]
        vel = sc.trajectories[:, step, 2:]
        ids = [f"T{i}" for i in range(sc.n_agents)]
        intents = IntentPredictor(model, ASSET).predict(10.0, ids, pos, vel)

        true_main = {i for i, lab in enumerate(sc.labels) if lab == "main_axis"}
        for si in intents:
            members = [int(t.lstrip("T")) for t in si.member_track_ids]
            if len(set(members) & true_main) / len(members) <= 0.8:
                continue
            # true centroid 5 s later (10 samples at 0.5 s)
            future = sc.trajectories[members, step + 10, :2].mean(axis=0)
            pred5 = si.forecast_centroids[4]   # index 4 = 5th second
            err = float(np.linalg.norm(future - pred5))
            assert err < 100.0, f"5-s forecast error {err:.0f} m >= 100 m"


# ---------------------------------------------------------------------------
# 5. Message contract
# ---------------------------------------------------------------------------

class TestContract:
    def test_intent_distribution_sums_to_one(self, model):
        rng = np.random.default_rng(13)
        sc = feint_main_axis(rng)
        pos = sc.trajectories[:, 20, :2]
        vel = sc.trajectories[:, 20, 2:]
        ids = [f"T{i}" for i in range(sc.n_agents)]
        for si in IntentPredictor(model, ASSET).predict(10.0, ids, pos, vel):
            total = sum(si.intent_distribution.values())
            assert abs(total - 1.0) < 1e-6
            assert set(si.intent_distribution) == set(INTENT_CLASSES)

    def test_to_dict_schema_fields(self, model):
        rng = np.random.default_rng(14)
        sc = feint_main_axis(rng)
        pos = sc.trajectories[:, 20, :2]
        vel = sc.trajectories[:, 20, 2:]
        ids = [f"T{i}" for i in range(sc.n_agents)]
        intents = IntentPredictor(model, ASSET).predict(10.0, ids, pos, vel)
        d = intents[0].to_dict()
        for key in ["t", "cluster_id", "member_track_ids", "intent_distribution",
                    "forecast", "value_multiplier"]:
            assert key in d
        assert "sigma" in d["forecast"]
