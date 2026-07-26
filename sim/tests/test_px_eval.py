"""PX acceptance tests — Monte Carlo cost-exchange evaluation.

Criteria:
1. Episode metrics are structurally complete with physically-grounded cost model
   (defense spend from event log, not alloc-cycle magazine bookkeeping).
2. Aggregate statistics are deterministic: same config → same result.
3. EconomicMDP achieves a lower or equal cost-exchange ratio (CER) than
   GreedyMyopic across the scenario mix, demonstrating the value of
   magazine-conscious rationing.
4. The headline figure is reported as a structured dict with both allocators,
   their CERs, and the improvement fraction.
5. Per-scenario breakdown covers all scenarios in the study.
6. Intercept rate (IR) and CER are both reported; they trade off against each
   other (Greedy IR ≥ Eco IR for equal or higher spend).
"""
from __future__ import annotations

import pytest
import numpy as np

import eval.runner as erunner
from eval.metrics import aggregate, headline_figure, split_by
from eval.monte_carlo import (
    EpisodeMetrics,
    MC_MAX_TIME,
    THREAT_COST_USD,
    run_mc_episode,
)
from league.policy import SwarmPolicy, PARAM_BOUNDS
from s2r.gap import RealityGap
from sim.alloc.economic_mdp import EconomicMDP
from sim.alloc.greedy import GreedyMyopic

# ---------------------------------------------------------------------------
# Shared expensive fixture — runs once for the whole module
# ---------------------------------------------------------------------------

_ORIG_N_GEN = erunner._PX_LEAGUE_N_GEN
_ORIG_POP   = erunner._PX_LEAGUE_POP
_ORIG_SEED  = erunner._PX_LEAGUE_SEED


@pytest.fixture(scope="module")
def px_result():
    """Fast PX study: 2 gen × 4 pop league, 3 seeds, 1 PL tactic."""
    erunner._PX_LEAGUE_N_GEN = 2
    erunner._PX_LEAGUE_POP = 4
    erunner._PX_LEAGUE_SEED = 77
    result = erunner.run_px_study(n_seeds=3, n_tactics=1)
    # restore module state for other tests in the broader suite
    erunner._PX_LEAGUE_N_GEN = _ORIG_N_GEN
    erunner._PX_LEAGUE_POP   = _ORIG_POP
    erunner._PX_LEAGUE_SEED  = _ORIG_SEED
    return result


# ---------------------------------------------------------------------------
# 1. Episode metrics
# ---------------------------------------------------------------------------

class TestEpisodeMetrics:
    def _run_default_episode(self, alloc, name, seed=0):
        return run_mc_episode(seed, alloc, name, "default",
                              policy=None, gap=RealityGap.nominal())

    def test_required_keys(self):
        ep = self._run_default_episode(EconomicMDP(), "eco")
        d = ep.to_dict()
        for key in ("seed", "allocator_name", "scenario_label", "n_threats",
                    "intercepts", "leakers", "asset_hp", "defense_spend_usd",
                    "threat_value_at_risk", "intercepted_value",
                    "asset_damage_usd", "time_s",
                    "intercept_rate", "cost_exchange_ratio"):
            assert key in d, f"missing key: {key}"

    def test_intercept_rate_bounded(self):
        ep = self._run_default_episode(EconomicMDP(), "eco")
        assert 0.0 <= ep.intercept_rate <= 1.0

    def test_defense_spend_from_event_log(self):
        """Defense spend must come from physical intercept events, not magazine."""
        from sim.effectors.catalogue import CATALOGUE
        ep = self._run_default_episode(EconomicMDP(), "eco")
        # If 0 intercepts, spend should be 0
        if ep.intercepts == 0:
            assert ep.defense_spend_usd == 0.0
        else:
            # Each intercept costs at least min effector cost
            min_cost = min(e.cost_usd for e in CATALOGUE.values())
            assert ep.defense_spend_usd >= ep.intercepts * min_cost

    def test_threat_value_at_risk_proportional(self):
        ep = self._run_default_episode(EconomicMDP(), "eco")
        assert ep.threat_value_at_risk == pytest.approx(ep.n_threats * THREAT_COST_USD)

    def test_asset_damage_non_negative(self):
        ep = self._run_default_episode(EconomicMDP(), "eco")
        assert ep.asset_damage_usd >= 0.0

    def test_episode_time_within_cap(self):
        ep = self._run_default_episode(EconomicMDP(), "eco")
        assert ep.time_s <= MC_MAX_TIME + 0.1  # slight floating point tolerance

    def test_determinism(self):
        """Same seed + allocator + scenario → identical metrics."""
        a = self._run_default_episode(EconomicMDP(), "eco", seed=7)
        b = self._run_default_episode(EconomicMDP(), "eco", seed=7)
        assert a.intercepts == b.intercepts
        assert a.defense_spend_usd == b.defense_spend_usd
        assert a.time_s == b.time_s

    def test_pl_policy_episode_runs(self):
        """Episodes with a SwarmPolicy policy spawn correctly."""
        rng = np.random.default_rng(0)
        policy = SwarmPolicy.random(rng, policy_id=0)
        ep = run_mc_episode(0, EconomicMDP(), "eco", "pl_test",
                            policy=policy, gap=RealityGap.nominal())
        assert ep.n_threats >= 1


# ---------------------------------------------------------------------------
# 2. Aggregate metrics
# ---------------------------------------------------------------------------

class TestCostExchangeMetrics:
    def _make_episodes(self, n=6):
        episodes = []
        alloc = EconomicMDP()
        for i in range(n):
            ep = run_mc_episode(i, alloc, "EconomicMDP", "default",
                                policy=None, gap=RealityGap.nominal())
            episodes.append(ep)
        return episodes

    def test_aggregate_keys(self):
        eps = self._make_episodes(3)
        stats = aggregate(eps, "test")
        d = stats.to_dict()
        for key in ("label", "n", "mean_intercept_rate", "mean_cost_exchange_ratio",
                    "p10_cost_exchange", "p90_cost_exchange",
                    "mean_leakers", "mean_asset_damage_usd", "mean_defense_spend_usd"):
            assert key in d

    def test_aggregate_n_correct(self):
        eps = self._make_episodes(4)
        assert aggregate(eps).n == 4

    def test_percentile_ordering(self):
        eps = self._make_episodes(6)
        s = aggregate(eps)
        assert s.p10_cost_exchange <= s.mean_cost_exchange_ratio <= s.p90_cost_exchange + 1e-9

    def test_headline_figure_keys(self):
        eco_eps = self._make_episodes(3)
        grd_eps = [run_mc_episode(i, GreedyMyopic(), "grd", "default")
                   for i in range(3)]
        hf = headline_figure(eco_eps, grd_eps)
        assert "economic_mdp" in hf and "greedy_myopic" in hf
        assert "cer_improvement_frac" in hf
        assert "intercept_rate_delta" in hf

    def test_empty_aggregate_raises(self):
        with pytest.raises(ValueError):
            aggregate([])

    def test_split_by_scenario(self):
        eco_eps = self._make_episodes(3)
        groups = split_by(eco_eps, "scenario_label")
        assert "default" in groups

    def test_split_by_allocator(self):
        eps = (self._make_episodes(2) +
               [run_mc_episode(i, GreedyMyopic(), "GreedyMyopic", "default") for i in range(2)])
        groups = split_by(eps, "allocator_name")
        assert "EconomicMDP" in groups and "GreedyMyopic" in groups


# ---------------------------------------------------------------------------
# 3. PX runner (uses module fixture)
# ---------------------------------------------------------------------------

class TestPXRunner:
    def test_result_has_required_keys(self, px_result):
        for key in ("config", "headline", "by_scenario", "all_episodes"):
            assert key in px_result

    def test_config_fields(self, px_result):
        cfg = px_result["config"]
        assert cfg["n_seeds"] == 3
        assert cfg["n_tactics"] == 1

    def test_both_allocators_in_episodes(self, px_result):
        assert "EconomicMDP" in px_result["all_episodes"]
        assert "GreedyMyopic" in px_result["all_episodes"]

    def test_episode_count(self, px_result):
        n_scenarios = 2  # default + 1 PL tactic
        n_seeds = 3
        assert len(px_result["all_episodes"]["EconomicMDP"]) == n_scenarios * n_seeds
        assert len(px_result["all_episodes"]["GreedyMyopic"]) == n_scenarios * n_seeds

    def test_per_scenario_breakdown_covers_all_scenarios(self, px_result):
        eco_scenarios = set(px_result["by_scenario"]["EconomicMDP"].keys())
        grd_scenarios = set(px_result["by_scenario"]["GreedyMyopic"].keys())
        assert eco_scenarios == grd_scenarios
        assert "default" in eco_scenarios

    def test_headline_has_both_allocators(self, px_result):
        h = px_result["headline"]
        assert "economic_mdp" in h and "greedy_myopic" in h

    def test_reproducible(self, px_result):
        erunner._PX_LEAGUE_N_GEN = 2
        erunner._PX_LEAGUE_POP = 4
        erunner._PX_LEAGUE_SEED = 77
        result2 = erunner.run_px_study(n_seeds=3, n_tactics=1)
        erunner._PX_LEAGUE_N_GEN = _ORIG_N_GEN
        erunner._PX_LEAGUE_POP = _ORIG_POP
        erunner._PX_LEAGUE_SEED = _ORIG_SEED
        h1 = px_result["headline"]
        h2 = result2["headline"]
        assert h1["economic_mdp"]["mean_cost_exchange_ratio"] == pytest.approx(
            h2["economic_mdp"]["mean_cost_exchange_ratio"], rel=1e-5)


# ---------------------------------------------------------------------------
# 4. Acceptance criteria
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    def test_a_headline_figure_is_complete(self, px_result):
        """Acceptance A: the deliverable dict contains both allocators + deltas."""
        h = px_result["headline"]
        assert isinstance(h["economic_mdp"]["mean_cost_exchange_ratio"], float)
        assert isinstance(h["greedy_myopic"]["mean_cost_exchange_ratio"], float)
        assert isinstance(h["cer_improvement_frac"], float)
        assert isinstance(h["intercept_rate_delta"], float)

    def test_b_economic_mdp_cer_not_worse_than_greedy(self, px_result):
        """Acceptance B: EconomicMDP CER ≤ Greedy CER.

        The economic allocator is never more expensive per intercept than the
        greedy baseline. Some deterioration in intercept RATE is acceptable
        (it holds fire on low-value targets), but cost efficiency must be
        maintained or improved.
        """
        h = px_result["headline"]
        eco_cer = h["economic_mdp"]["mean_cost_exchange_ratio"]
        grd_cer = h["greedy_myopic"]["mean_cost_exchange_ratio"]
        assert eco_cer <= grd_cer, (
            f"EconomicMDP CER {eco_cer:.2f} exceeds Greedy CER {grd_cer:.2f} "
            f"— magazine rationing is not saving cost."
        )

    def test_c_cer_improvement_non_negative(self, px_result):
        """Acceptance C: reported improvement fraction is consistent with B."""
        h = px_result["headline"]
        assert h["cer_improvement_frac"] >= -0.01  # allow tiny noise

    def test_d_per_scenario_stats_structured(self, px_result):
        """Acceptance D: every scenario has the required aggregate fields."""
        required = {"mean_intercept_rate", "mean_cost_exchange_ratio",
                    "p10_cost_exchange", "p90_cost_exchange", "n"}
        for alloc_name in ("EconomicMDP", "GreedyMyopic"):
            for sc_label, stats in px_result["by_scenario"][alloc_name].items():
                missing = required - set(stats.keys())
                assert not missing, f"{alloc_name}/{sc_label} missing {missing}"

    def test_e_cer_in_sane_range(self, px_result):
        """Acceptance E: CERs are in a physically interpretable range.

        A CER of 0 means the defense is free; > 1000 means the defense is
        spending 1000× the threat value per intercept — neither makes sense
        for the modelled effector and threat costs.
        """
        for alloc_key in ("economic_mdp", "greedy_myopic"):
            cer = px_result["headline"][alloc_key]["mean_cost_exchange_ratio"]
            assert 0.0 < cer < 1000.0, f"{alloc_key} CER={cer} outside sane range"
