"""PS acceptance tests — P-S2R sim-to-real validation strategy.

Criteria:
1. Gate coverage: every RealityGap dimension has exactly one documented
   real-world validation gate (machine-checked).
2. Determinism: the probe episode is seed-reproducible (same gap + seed
   -> identical event-log hash).
3. The sensitivity sweep produces a meaningful tornado ranking: at least
   3 dimensions cost material engagement margin at their worst bound, and
   no dimension *improves* the margin (degradation sanity).
4. Dimensions the sim is insensitive to are surfaced as explicit fidelity
   limitations — and still carry a validation gate.
5. The domain-randomized robustness study reports a conclusion-stability
   fraction with identified failing gaps.
6. Scope: gates describe measurement only — no effector/RF-design language.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from s2r.episodes import (
    MARGIN_THRESHOLD_M,
    probe_policy,
    result_hash,
    run_probe_episode,
)
from s2r.gap import GAP_DIMS, GAP_ENVELOPE, N_GAP_DIMS, RealityGap
from s2r.gates import VALID_TIERS, VALIDATION_GATES, coverage_check, gates_by_dim
from s2r.robustness import robustness_study
from s2r.sensitivity import oat_sweep
from sim.bridge.scenario import default_sensor_mesh


# ---------------------------------------------------------------------------
# Shared expensive fixtures (module scope — computed once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nominal_result():
    return run_probe_episode(RealityGap.nominal(), seed=0)


@pytest.fixture(scope="module")
def oat_report():
    return oat_sweep(seeds=(0, 1, 2))


@pytest.fixture(scope="module")
def rob_report():
    return robustness_study(n_samples=8, seed=0)


# ---------------------------------------------------------------------------
# 1. RealityGap envelope
# ---------------------------------------------------------------------------

class TestRealityGap:
    def test_nominal_is_identity_on_mesh(self):
        nominal_mesh = default_sensor_mesh()
        gap_mesh = RealityGap.nominal().apply_to_mesh()
        for orig, pert in zip(nominal_mesh.sensors, gap_mesh.sensors):
            assert pert.pd_max == pytest.approx(orig.pd_max)
            assert np.allclose(pert.noise_std, orig.noise_std)
            assert pert.false_alarm_rate == pytest.approx(orig.false_alarm_rate)
            assert pert.latency == pytest.approx(orig.latency)
            assert pert.update_rate == pytest.approx(orig.update_rate)

    def test_apply_does_not_mutate_source_mesh(self):
        mesh = default_sensor_mesh()
        before = [s.pd_max for s in mesh.sensors]
        RealityGap.worst_case().apply_to_mesh(mesh)
        assert [s.pd_max for s in mesh.sensors] == before

    def test_worst_case_within_envelope(self):
        assert RealityGap.worst_case().is_within_envelope()

    def test_sample_within_envelope(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            assert RealityGap.sample(rng).is_within_envelope()

    def test_single_unknown_dim_raises(self):
        with pytest.raises(KeyError):
            RealityGap.single("rf_waveform", 1.0)  # not a thing here, ever

    def test_single_out_of_envelope_raises(self):
        with pytest.raises(ValueError):
            RealityGap.single("pd_scale", 0.1)  # envelope floor is 0.6

    def test_worst_case_perturbs_every_sensor_field(self):
        mesh = RealityGap.worst_case().apply_to_mesh()
        nominal = default_sensor_mesh()
        for orig, pert in zip(nominal.sensors, mesh.sensors):
            assert pert.pd_max < orig.pd_max
            assert np.all(pert.noise_std > orig.noise_std)
            assert pert.false_alarm_rate > orig.false_alarm_rate
            assert pert.latency > orig.latency
            assert pert.update_rate < orig.update_rate

    def test_comms_cfg_perturbed(self):
        cfg = RealityGap.worst_case().make_comms_cfg()
        assert cfg.comm_radius == pytest.approx(1000.0)
        assert cfg.base_drop_rate == pytest.approx(0.30)

    def test_to_dict_covers_all_dims(self):
        d = RealityGap.nominal().to_dict()
        assert set(d.keys()) == set(GAP_DIMS)
        assert len(d) == N_GAP_DIMS


# ---------------------------------------------------------------------------
# 2. Validation gates (the strategy)
# ---------------------------------------------------------------------------

class TestValidationGates:
    def test_full_coverage(self):
        """Acceptance 1: every gap dimension has a validation gate."""
        assert coverage_check() == []

    def test_one_gate_per_dim(self):
        dims = [g.gap_dim for g in VALIDATION_GATES]
        assert len(dims) == len(set(dims)) == N_GAP_DIMS

    def test_gates_fully_specified(self):
        for g in VALIDATION_GATES:
            assert g.tier in VALID_TIERS
            assert len(g.measurement) > 30
            assert len(g.accept_criterion) > 30
            assert len(g.at_risk) > 10

    def test_gates_by_dim_lookup(self):
        by_dim = gates_by_dim()
        for dim in GAP_DIMS:
            assert by_dim[dim].gap_dim == dim

    def test_scope_boundary_in_gate_text(self):
        """Gates are measurement-only: no effector / RF-design vocabulary."""
        forbidden = ("waveform", "jammer", "fire-control", "warhead",
                     "transmit power", "weapon release")
        for g in VALIDATION_GATES:
            text = (g.measurement + " " + g.accept_criterion).lower()
            for term in forbidden:
                assert term not in text, f"{g.gap_dim}: forbidden term {term!r}"


# ---------------------------------------------------------------------------
# 3. Probe episode
# ---------------------------------------------------------------------------

class TestProbeEpisode:
    def test_required_keys(self, nominal_result):
        for key in ("gap", "seed", "n_threats", "intercepts", "leakers",
                    "asset_hp", "time_s", "margin_m",
                    "mean_intercept_range_m", "log_hash"):
            assert key in nominal_result

    def test_probe_size_fixed(self, nominal_result):
        assert nominal_result["n_threats"] == 5

    def test_nominal_margin_clears_threshold(self, nominal_result):
        """At face-value fidelity the headline conclusion holds."""
        assert nominal_result["margin_m"] >= MARGIN_THRESHOLD_M

    def test_determinism(self, nominal_result):
        """Acceptance 2: same gap + seed -> identical event log."""
        again = run_probe_episode(RealityGap.nominal(), seed=0)
        assert again["log_hash"] == nominal_result["log_hash"]
        assert result_hash(again) == result_hash(nominal_result)

    def test_different_seed_differs(self, nominal_result):
        other = run_probe_episode(RealityGap.nominal(), seed=99)
        assert other["log_hash"] != nominal_result["log_hash"]

    def test_speed_scale_reaches_probe_policy(self):
        gap = RealityGap.single("target_speed_scale", 1.3)
        assert probe_policy(gap).main_speed == pytest.approx(22.0 * 1.3)
        assert probe_policy(RealityGap.nominal()).main_speed == pytest.approx(22.0)


# ---------------------------------------------------------------------------
# 4. Sensitivity sweep + tornado
# ---------------------------------------------------------------------------

class TestSensitivity:
    def test_all_dims_swept(self, oat_report):
        assert set(oat_report.per_dim.keys()) == set(GAP_DIMS)

    def test_tornado_sorted_by_impact(self, oat_report):
        deltas = [abs(d.delta_m) for d in oat_report.tornado()]
        assert deltas == sorted(deltas, reverse=True)

    def test_degradation_never_helps(self, oat_report):
        """No worst-bound perturbation should IMPROVE the margin (tolerance
        for discrete intercept-geometry jitter)."""
        for d in oat_report.per_dim.values():
            assert d.delta_m <= 5.0, (
                f"{d.gap_dim} at worst bound improved margin by {d.delta_m} m")

    def test_material_signal_exists(self, oat_report):
        """Acceptance 3: >=3 dims cost material margin -> the tornado
        actually prioritises the validation campaign."""
        material = [d for d in oat_report.per_dim.values() if d.delta_m <= -10.0]
        assert len(material) >= 3

    def test_insensitive_dims_have_gates(self, oat_report):
        """Acceptance 4: parameters the sim cannot price still carry a
        validation gate — insensitivity is a limitation, not a free pass."""
        by_dim = gates_by_dim()
        for dim in oat_report.insensitive_dims():
            assert dim in by_dim

    def test_report_serialisable(self, oat_report):
        d = oat_report.to_dict()
        assert "tornado_order" in d and len(d["tornado_order"]) == N_GAP_DIMS
        assert "insensitive_dims" in d


# ---------------------------------------------------------------------------
# 5. Domain-randomized robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_stability_fraction_bounded(self, rob_report):
        assert 0.0 <= rob_report["stability_fraction"] <= 1.0

    def test_sample_count(self, rob_report):
        assert len(rob_report["results"]) == rob_report["n_samples"] == 8

    def test_worst_case_is_min_margin(self, rob_report):
        margins = [r["margin_m"] for r in rob_report["results"]]
        assert rob_report["worst_case"]["margin_m"] == min(margins)

    def test_holds_flags_consistent(self, rob_report):
        thr = rob_report["threshold_m"]
        for r in rob_report["results"]:
            assert r["holds"] == (r["margin_m"] >= thr)

    def test_failing_gaps_are_reported(self, rob_report):
        n_fails = sum(1 for r in rob_report["results"] if not r["holds"])
        assert len(rob_report["failing_gaps"]) == n_fails

    def test_reproducible(self):
        a = robustness_study(n_samples=2, seed=5)
        b = robustness_study(n_samples=2, seed=5)
        assert a["stability_fraction"] == b["stability_fraction"]
        assert [r["log_hash"] for r in a["results"]] == \
               [r["log_hash"] for r in b["results"]]


# ---------------------------------------------------------------------------
# 6. Acceptance roll-up
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    def test_a_gate_coverage_complete(self):
        assert coverage_check() == []

    def test_b_headline_holds_at_nominal_fails_under_stress(self, oat_report):
        """The margin conclusion is fidelity-sensitive — which is exactly why
        the gates exist. Nominal clears the threshold; the worst single-dim
        stress does not leave it untouched."""
        assert oat_report.nominal_margin_m >= MARGIN_THRESHOLD_M
        top = oat_report.tornado()[0]
        assert top.worst_margin_m < oat_report.nominal_margin_m

    def test_c_stability_fraction_reported(self, rob_report):
        """The deliverable: a quantified share of the reality envelope in
        which the headline conclusion survives."""
        assert isinstance(rob_report["stability_fraction"], float)
        assert "worst_case" in rob_report and "failing_gaps" in rob_report

    def test_d_gap_dataclass_frozen(self):
        """Gaps are immutable — a study cannot silently drift mid-run."""
        gap = RealityGap.nominal()
        with pytest.raises(dataclasses.FrozenInstanceError):
            gap.pd_scale = 0.5  # type: ignore[misc]
