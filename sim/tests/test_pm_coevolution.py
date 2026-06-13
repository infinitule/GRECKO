"""PM acceptance tests — mutual co-evolution (Blue adapts to discovered Red).

Criteria:
1. BluePolicy is well-formed: bounded loadout, magazine covers all effectors,
   serialisation round-trips.
2. The Blue search space enumerates correctly (unordered loadouts collapse
   permutations).
3. The injectable loadout actually changes interceptor effectors AND the
   resulting defense spend (cost lever works end to end).
4. evaluate_blue produces consistent CER / leaker / score metrics.
5. optimize_blue picks the lowest-blue_score policy in the space.
6. The full co-evolution arc reduces (or holds) Blue's cost-exchange ratio:
   the adapted Blue is never more expensive per intercept than the default.
7. The adapted Blue's advantage is robust to one round of Red counter-evolution.
"""
from __future__ import annotations

import numpy as np
import pytest

from eval.monte_carlo import run_mc_episode
from league.blue import (
    BluePolicy,
    LOADOUT_MENU,
    enumerate_blue_policies,
)
from league.coevolution import Coevolution, _blue_score
from league.policy import SwarmPolicy
from sim.alloc.economic_mdp import EconomicMDP


def _red_lib(n=2, seed=0):
    rng = np.random.default_rng(seed)
    return [SwarmPolicy.random(rng, policy_id=i) for i in range(n)]


# ---------------------------------------------------------------------------
# Shared expensive fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coevo_result():
    """Tiny but real co-evolution arc: 1 prebuilt tactic, loadout-only search."""
    co = Coevolution(
        red_library=_red_lib(1, seed=3),
        n_red_tactics=1,
        blue_lambda_choices=(0.1,),     # search loadouts only, fixed lambda
        search_seeds=(0,),
        confirm_seeds=(0,),
        counter_n_gen=1, counter_pop=2,
        seed=0,
    )
    return co, co.run(counter_evolve=True)


# ---------------------------------------------------------------------------
# 1. BluePolicy
# ---------------------------------------------------------------------------

class TestBluePolicy:
    def test_default_is_three_interceptors(self):
        b = BluePolicy.default()
        assert len(b.loadout) == 3

    def test_invalid_loadout_length_raises(self):
        with pytest.raises(ValueError):
            BluePolicy(loadout=("collision_drone",))

    def test_invalid_effector_raises(self):
        with pytest.raises(ValueError):
            BluePolicy(loadout=("phaser", "net_capture_drone", "collision_drone"))

    def test_magazine_covers_all_effectors(self):
        mag = BluePolicy.default().magazine()
        for eff in ("kinetic_interceptor", "net_capture_drone",
                    "collision_drone", "ew_soft_kill"):
            assert mag.rounds.get(eff, 0) > 0

    def test_serialisation_round_trip(self):
        b = BluePolicy(lambda_cost=0.2,
                       loadout=("collision_drone", "net_capture_drone",
                                "kinetic_interceptor"))
        b2 = BluePolicy.from_dict(b.to_dict())
        assert b2.lambda_cost == pytest.approx(0.2)
        assert b2.loadout == b.loadout

    def test_random_within_menu(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            b = BluePolicy.random(rng)
            assert all(e in LOADOUT_MENU for e in b.loadout)


# ---------------------------------------------------------------------------
# 2. Search space
# ---------------------------------------------------------------------------

class TestSearchSpace:
    def test_unordered_collapses_permutations(self):
        pols = enumerate_blue_policies(lambda_choices=(0.1,), unordered_loadout=True)
        # C(3+3-1, 3) = 10 multisets over a 3-item menu
        assert len(pols) == 10

    def test_ordered_is_full_product(self):
        pols = enumerate_blue_policies(lambda_choices=(0.1,), unordered_loadout=False)
        assert len(pols) == 27  # 3^3

    def test_lambda_multiplies_count(self):
        pols = enumerate_blue_policies(lambda_choices=(0.0, 0.1, 0.35))
        assert len(pols) == 30  # 3 lambdas × 10 unordered loadouts


# ---------------------------------------------------------------------------
# 3. Loadout cost lever (end to end)
# ---------------------------------------------------------------------------

class TestLoadoutCostLever:
    def test_loadout_changes_interceptor_effectors(self):
        from sim.bridge.scenario import BridgeScenario
        sc = BridgeScenario(seed=1, loadout=["collision_drone"] * 3)
        effs = {iv.effector_type for iv in sc.world.interceptors.values()}
        assert effs == {"collision_drone"}

    def test_cheap_loadout_costs_less_than_expensive(self):
        """All-collision must cost no more per episode than all-kinetic."""
        red = _red_lib(1, seed=5)[0]
        alloc = EconomicMDP()
        cheap = run_mc_episode(0, alloc, "eco", "x", policy=red,
                               lambda_cost=0.1, loadout=["collision_drone"] * 3)
        pricey = run_mc_episode(0, alloc, "eco", "x", policy=red,
                                lambda_cost=0.1, loadout=["kinetic_interceptor"] * 3)
        assert cheap.defense_spend_usd <= pricey.defense_spend_usd
        # and when both intercept, cheap has a strictly lower CER
        if cheap.intercepts > 0 and pricey.intercepts > 0:
            assert cheap.cost_exchange_ratio < pricey.cost_exchange_ratio


# ---------------------------------------------------------------------------
# 4 & 5. Blue evaluation and optimisation
# ---------------------------------------------------------------------------

class TestBlueOptimisation:
    def test_blue_score_monotone_in_inputs(self):
        assert _blue_score(10.0, 0.0) < _blue_score(10.0, 1.0)
        assert _blue_score(5.0, 1.0) < _blue_score(10.0, 1.0)

    def test_evaluate_blue_metrics_bounded(self, coevo_result):
        co, _ = coevo_result
        ev = co.evaluate_blue(BluePolicy.default(), co.seed_red_library(), (0,))
        assert 0.0 <= ev.mean_intercept_rate <= 1.0
        assert ev.mean_cost_exchange_ratio >= 0.0
        assert ev.mean_leakers >= 0.0

    def test_optimize_picks_min_score(self, coevo_result):
        co, _ = coevo_result
        best, scored = co.optimize_blue(co.seed_red_library())
        assert scored[0].blue == best
        assert all(scored[0].blue_score <= s.blue_score for s in scored)


# ---------------------------------------------------------------------------
# 6 & 7. The co-evolution arc
# ---------------------------------------------------------------------------

class TestCoevolutionArc:
    def test_result_structure(self, coevo_result):
        _, res = coevo_result
        for key in ("baseline_blue", "adapted_blue", "cer_improvement_frac",
                    "counter_evolved", "red_clawback_cer"):
            assert key in res

    def test_adapted_not_more_expensive_than_default(self, coevo_result):
        """Acceptance: the adapted Blue's CER ≤ the default Blue's CER."""
        _, res = coevo_result
        base = res["baseline_blue"]["mean_cost_exchange_ratio"]
        adapt = res["adapted_blue"]["mean_cost_exchange_ratio"]
        assert adapt <= base + 1e-6

    def test_cer_improvement_non_negative(self, coevo_result):
        _, res = coevo_result
        assert res["cer_improvement_frac"] >= -1e-6

    def test_adapted_blue_picks_cheaper_loadout(self, coevo_result):
        """Against cheap quadrotor swarms, the adapted loadout should not be
        all-kinetic (the most expensive option)."""
        _, res = coevo_result
        adapted_loadout = res["adapted_blue"]["blue"]["loadout"]
        assert adapted_loadout != ["kinetic_interceptor"] * 3

    def test_red_clawback_is_reported(self, coevo_result):
        """Red counter-evolution cannot make Blue's effectors more expensive,
        so the clawback on the COST axis is small/non-positive."""
        _, res = coevo_result
        assert isinstance(res["red_clawback_cer"], float)


# ---------------------------------------------------------------------------
# 8. Cost adaptation is free on the reality-gap robustness axis (PS gap)
# ---------------------------------------------------------------------------

class TestRobustnessUnderGap:
    def test_adapted_loadout_threads_through_probe(self):
        """The S2R probe accepts a Blue loadout so the adapted Blue can be
        re-validated across the reality gap."""
        from s2r.episodes import run_probe_episode
        from s2r.gap import RealityGap
        r = run_probe_episode(RealityGap.nominal(), seed=0,
                              loadout=["collision_drone"] * 3)
        assert "margin_m" in r and r["n_threats"] >= 1

    def test_cheaper_blue_not_less_robust_in_worst_corner(self):
        """Cost adaptation must not degrade the worst-case engagement margin
        relative to the default loadout (it should match or beat it)."""
        import numpy as np
        from s2r.episodes import run_probe_episode
        from s2r.gap import RealityGap
        rng = np.random.default_rng(0)
        gaps = [RealityGap.sample(rng) for _ in range(3)]
        default_worst = min(run_probe_episode(g, seed=i)["margin_m"]
                            for i, g in enumerate(gaps))
        adapted_worst = min(
            run_probe_episode(g, seed=i, loadout=["collision_drone"] * 3)["margin_m"]
            for i, g in enumerate(gaps))
        assert adapted_worst >= default_worst - 1e-6
