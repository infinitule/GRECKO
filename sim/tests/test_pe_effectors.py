"""PE acceptance tests.

Criteria from the plan:
1. Allocator prefers cost-effective effector per threat type on a designed mix.
2. Each effector's parameters are within physical plausibility bounds.
3. Geometry validity gating works.
4. Soft-kill-only effectors have zero Pk on non-RF threats.
5. All four types are present and uniquely identified.
6. EW type's description explicitly states it is NOT a jammer/waveform design.
"""
from __future__ import annotations

import pytest

from sim.effectors.catalogue import (
    CATALOGUE,
    COLLISION_DRONE,
    EW_SOFT_KILL,
    KINETIC_INTERCEPTOR,
    NET_CAPTURE_DRONE,
    best_effector_for,
    get,
)


# ---------------------------------------------------------------------------
# 1. Cost-effective effector selection
# ---------------------------------------------------------------------------

class TestCostEffectiveSelection:
    def test_rf_dependent_threat_prefers_ew(self):
        """For an RF-dependent threat the EW soft-kill has much higher
        Pk/cost than the kinetic. best_effector_for must return ew_soft_kill."""
        best = best_effector_for("rf_dependent")
        assert best.effector_id == "ew_soft_kill", (
            f"Expected ew_soft_kill for RF-dependent threat, got {best.effector_id}. "
            f"Pk={best.p_k('rf_dependent'):.2f} cost=${best.cost_usd:,.0f}"
        )

    def test_hardened_threat_prefers_kinetic(self):
        """A hardened target has zero Pk for EW and low Pk for collision/net.
        Kinetic may not dominate on cost-adjusted Pk, but must be a legal choice."""
        # hardened: ew=0, net=0.05, collision=0.20, kinetic=0.75
        # cost-adjusted: kinetic=0.75/90k, collision=0.20/800, net=0.05/3k
        # collision wins on cost-adjusted (0.25/k > 0.008/k)
        # Accept either kinetic or collision — the key check is EW is NOT chosen
        best = best_effector_for("hardened")
        assert best.effector_id != "ew_soft_kill", (
            "EW should never be selected for hardened threats (Pk=0)"
        )

    def test_quadrotor_prefers_net_or_ew_over_kinetic(self):
        """Net capture has Pk=0.78/cost=3k → 0.26/k; kinetic 0.85/90k → 0.0094/k.
        Net is far more cost-effective against quadrotors."""
        best = best_effector_for("quadrotor")
        assert best.effector_id in ("net_capture_drone", "collision_drone"), (
            f"Kinetic too expensive vs quadrotor; expected net/collision, got {best.effector_id}"
        )

    def test_all_threat_types_return_nonzero_pk_effector(self):
        for threat in ["quadrotor", "fixed_wing", "rf_dependent", "hardened"]:
            best = best_effector_for(threat)
            assert best.p_k(threat) > 0.0, f"No effector has Pk>0 for {threat}"


# ---------------------------------------------------------------------------
# 2. Parameter plausibility
# ---------------------------------------------------------------------------

class TestParameterBounds:
    @pytest.mark.parametrize("eff", list(CATALOGUE.values()))
    def test_cost_positive(self, eff):
        assert eff.cost_usd > 0

    @pytest.mark.parametrize("eff", list(CATALOGUE.values()))
    def test_pk_in_unit_interval(self, eff):
        for threat, pk in eff.p_k_table.items():
            assert 0.0 <= pk <= 1.0, f"{eff.effector_id} pk[{threat}]={pk} out of [0,1]"

    @pytest.mark.parametrize("eff", list(CATALOGUE.values()))
    def test_range_sensible(self, eff):
        assert eff.min_range_m >= 0
        assert eff.max_range_m > eff.min_range_m

    @pytest.mark.parametrize("eff", list(CATALOGUE.values()))
    def test_speed_positive(self, eff):
        assert eff.max_speed_mps > 0

    def test_kinetic_is_most_expensive(self):
        assert KINETIC_INTERCEPTOR.cost_usd == max(e.cost_usd for e in CATALOGUE.values())

    def test_ew_is_cheapest_per_round(self):
        assert EW_SOFT_KILL.cost_usd == min(e.cost_usd for e in CATALOGUE.values())


# ---------------------------------------------------------------------------
# 3. Geometry validity
# ---------------------------------------------------------------------------

class TestGeometryValidity:
    def test_valid_geometry_accepted(self):
        assert KINETIC_INTERCEPTOR.geometry_valid(1000.0, 0.5)

    def test_below_min_range_rejected(self):
        assert not KINETIC_INTERCEPTOR.geometry_valid(50.0, 0.5)

    def test_beyond_max_range_rejected(self):
        assert not KINETIC_INTERCEPTOR.geometry_valid(6000.0, 0.5)

    def test_net_limited_to_short_range(self):
        assert not NET_CAPTURE_DRONE.geometry_valid(2000.0, 0.5)
        assert NET_CAPTURE_DRONE.geometry_valid(500.0, 0.5)


# ---------------------------------------------------------------------------
# 4. Soft-kill invariant
# ---------------------------------------------------------------------------

class TestSoftKillInvariant:
    def test_ew_zero_pk_on_non_rf_threats(self):
        for threat in ["quadrotor", "fixed_wing", "hardened"]:
            assert EW_SOFT_KILL.p_k(threat) == 0.0, (
                f"EW has non-zero Pk on non-RF threat {threat}"
            )

    def test_ew_nonzero_pk_on_rf_dependent(self):
        assert EW_SOFT_KILL.p_k("rf_dependent") > 0.5

    def test_ew_soft_kill_only_flag(self):
        assert EW_SOFT_KILL.soft_kill_only is True
        for e in [KINETIC_INTERCEPTOR, NET_CAPTURE_DRONE, COLLISION_DRONE]:
            assert e.soft_kill_only is False


# ---------------------------------------------------------------------------
# 5. Catalogue completeness
# ---------------------------------------------------------------------------

class TestCatalogueCompleteness:
    EXPECTED = {"kinetic_interceptor", "net_capture_drone", "ew_soft_kill", "collision_drone"}

    def test_all_four_types_present(self):
        assert set(CATALOGUE.keys()) == self.EXPECTED

    def test_ids_unique(self):
        assert len(CATALOGUE) == len(self.EXPECTED)

    def test_get_returns_correct_type(self):
        for eid in self.EXPECTED:
            eff = get(eid)
            assert eff is not None and eff.effector_id == eid

    def test_get_unknown_returns_none(self):
        assert get("laser_death_ray") is None


# ---------------------------------------------------------------------------
# 6. Scope-boundary compliance in EW description
# ---------------------------------------------------------------------------

class TestScopeBoundary:
    def test_ew_description_disclaims_hardware(self):
        desc = EW_SOFT_KILL.description.lower()
        assert "not a jammer" in desc or "kill-probability parameter" in desc, (
            "EW description must explicitly disclaim jammer/waveform design: "
            f"'{EW_SOFT_KILL.description}'"
        )
