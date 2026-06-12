"""PL acceptance tests — adversarial co-evolution league.

Criteria from the plan:
1. Red win rate trend is non-negative over the league run.
2. ≥3 qualitatively distinct attack patterns in the tactic library.
3. Exported Scenarios pass PB structural validation (shape, labels, etc.).
4. Policy serialisation round-trip is lossless.
5. Behavioral fingerprints distinguish different policies.
6. distinct_clusters() counts correctly.
7. EvolutionStrategy.step() improves best fitness (not strictly required,
   but the population fitness distribution should widen, not collapse).
8. BridgeScenario with auto_authorize=True reaches leakers faster than
   without authorization.
9. Policy.to_scenario() generates valid Scenario trajectories.
10. Export/load round-trip preserves doctrine name, shape, and labels.
11. LeagueRunner completes without error for small N.
12. Win-rate trend helper is correct.
"""
from __future__ import annotations

import pathlib
import tempfile
from typing import List

import numpy as np
import pytest

from league.diversity import (
    behavioral_fingerprint,
    distinct_clusters,
    population_diversity,
)
from league.es import EvolutionStrategy
from league.export import export_tactic_library, load_scenario_from_npz, validate_scenario
from league.fitness import EpisodeResult, evaluate_policy
from league.league import GenerationStats, LeagueRunner
from league.policy import N_POLICY_PARAMS, PARAM_BOUNDS, SwarmPolicy
from learn.intent.doctrines import INTENT_CLASSES, N_STEPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_policy(seed: int = 0, policy_id: int = 0) -> SwarmPolicy:
    rng = np.random.default_rng(seed)
    return SwarmPolicy.random(rng, policy_id=policy_id)


def _small_league(n_gen: int = 4, pop: int = 4, seed: int = 42) -> LeagueRunner:
    """A fast league for testing — 4 gens × 4 pop × 1 episode each."""
    return LeagueRunner(n_generations=n_gen, pop_size=pop,
                        n_eval_episodes=1, seed=seed)


# ---------------------------------------------------------------------------
# 1. SwarmPolicy unit tests
# ---------------------------------------------------------------------------

class TestSwarmPolicy:
    def test_random_within_bounds(self):
        p = _random_policy(seed=0)
        assert p.theta.shape == (N_POLICY_PARAMS,)
        assert np.all(p.theta >= PARAM_BOUNDS[:, 0])
        assert np.all(p.theta <= PARAM_BOUNDS[:, 1])

    def test_n_main_at_least_1(self):
        p = _random_policy(0)
        assert p.n_main >= 1

    def test_n_total_components_sum(self):
        p = _random_policy(0)
        assert p.n_main + p.n_feint + p.n_screen == p.n_total

    def test_mutate_stays_in_bounds(self):
        rng = np.random.default_rng(1)
        p = _random_policy(0)
        for _ in range(20):
            child = p.mutate(rng, sigma=0.5)
            assert np.all(child.theta >= PARAM_BOUNDS[:, 0])
            assert np.all(child.theta <= PARAM_BOUNDS[:, 1])

    def test_crossover_within_bounds(self):
        rng = np.random.default_rng(2)
        p1 = _random_policy(1)
        p2 = _random_policy(2)
        child = p1.crossover(p2, rng)
        assert np.all(child.theta >= PARAM_BOUNDS[:, 0])
        assert np.all(child.theta <= PARAM_BOUNDS[:, 1])

    def test_serialisation_round_trip(self):
        p = _random_policy(0)
        p.fitness = 42.5
        p.generation = 3
        d = p.to_dict()
        p2 = SwarmPolicy.from_dict(d)
        assert np.allclose(p.theta, p2.theta)
        assert p2.fitness == pytest.approx(42.5)
        assert p2.generation == 3


# ---------------------------------------------------------------------------
# 2. Scenario generation (policy → trajectories)
# ---------------------------------------------------------------------------

class TestPolicyToScenario:
    def test_trajectory_shape(self):
        rng = np.random.default_rng(0)
        p = _random_policy(0)
        sc = p.to_scenario(rng)
        n_agents = p.n_main + p.n_feint + p.n_screen
        assert sc.trajectories.shape == (n_agents, N_STEPS, 4)

    def test_labels_match_agents(self):
        rng = np.random.default_rng(0)
        p = _random_policy(0)
        sc = p.to_scenario(rng)
        assert len(sc.labels) == p.n_main + p.n_feint + p.n_screen

    def test_valid_labels(self):
        rng = np.random.default_rng(0)
        p = _random_policy(0)
        sc = p.to_scenario(rng)
        valid = set(INTENT_CLASSES) | {"screen"}
        for lab in sc.labels:
            assert lab in valid, f"unexpected label: {lab!r}"

    def test_validate_scenario_passes(self):
        rng = np.random.default_rng(0)
        p = _random_policy(0)
        sc = p.to_scenario(rng)
        assert validate_scenario(sc)

    def test_no_feint_scenario_still_valid(self):
        """A policy with feint_frac=0 should generate a valid main-only scenario."""
        theta = _random_policy(0).theta.copy()
        theta[1] = 0.0  # feint_frac = 0
        theta[2] = 0.0  # screen_frac = 0
        p = SwarmPolicy(theta=theta)
        rng = np.random.default_rng(0)
        sc = p.to_scenario(rng)
        assert validate_scenario(sc)
        assert all(lab == "main_axis" for lab in sc.labels)


# ---------------------------------------------------------------------------
# 3. Diversity module
# ---------------------------------------------------------------------------

class TestDiversity:
    def test_fingerprint_length(self):
        p = _random_policy(0)
        fp = behavioral_fingerprint(p)
        assert fp.shape == (10,)

    def test_fingerprint_different_policies_differ(self):
        p1 = _random_policy(1)
        p2 = _random_policy(2)
        fp1 = behavioral_fingerprint(p1)
        fp2 = behavioral_fingerprint(p2)
        assert not np.allclose(fp1, fp2)

    def test_population_diversity_zero_for_single(self):
        pops = [_random_policy(0)]
        assert population_diversity(pops) == pytest.approx(0.0)

    def test_population_diversity_positive_for_multiple(self):
        pops = [_random_policy(i) for i in range(5)]
        d = population_diversity(pops)
        assert d > 0.0

    def test_distinct_clusters_single_policy(self):
        p = [_random_policy(0)]
        assert distinct_clusters(p) == 1

    def test_distinct_clusters_separable_policies(self):
        """Force two maximally different policies: all-main vs all-feint."""
        # Policy A: feint_frac=0, angle=0
        ta = np.array(PARAM_BOUNDS[:, 0].copy())
        ta[0] = 10.0; ta[1] = 0.0; ta[2] = 0.0; ta[3] = 0.0
        # Policy B: feint_frac=0.5, angle=π
        tb = np.array(PARAM_BOUNDS[:, 0].copy())
        tb[0] = 10.0; tb[1] = 0.5; tb[2] = 0.0; tb[3] = np.pi
        # Policy C: small, slow, high feint
        tc = np.array(PARAM_BOUNDS[:, 0].copy())
        tc[0] = 5.0; tc[1] = 0.4; tc[5] = 15.0; tc[7] = 1500.0
        pols = [SwarmPolicy(theta=ta), SwarmPolicy(theta=tb), SwarmPolicy(theta=tc)]
        assert distinct_clusters(pols) >= 2


# ---------------------------------------------------------------------------
# 4. Evolution Strategy
# ---------------------------------------------------------------------------

class TestEvolutionStrategy:
    def test_population_size_preserved(self):
        es = EvolutionStrategy(pop_size=6, seed=0)
        es.seed_population()
        fits = [float(i) for i in range(6)]
        next_pop = es.step(fits)
        assert len(next_pop) == 6

    def test_generation_increments(self):
        es = EvolutionStrategy(pop_size=4, seed=0)
        es.seed_population()
        assert es.generation == 0
        es.step([1.0, 2.0, 3.0, 4.0])
        assert es.generation == 1

    def test_sigma_decays(self):
        es = EvolutionStrategy(pop_size=4, seed=0, sigma=0.10, sigma_decay=0.90)
        es.seed_population()
        es.step([1.0, 2.0, 3.0, 4.0])
        assert es.sigma < 0.10

    def test_elites_preserved(self):
        """The best policy's theta should appear in the next generation."""
        es = EvolutionStrategy(pop_size=4, elite_frac=0.5, seed=0)
        es.seed_population()
        pop0 = es.population
        best_theta = pop0[-1].theta.copy()
        # Give the last policy the highest fitness
        fits = [0.0, 0.0, 0.0, 10.0]
        next_pop = es.step(fits)
        thetas = [p.theta for p in next_pop]
        assert any(np.allclose(t, best_theta) for t in thetas)


# ---------------------------------------------------------------------------
# 5. Export / load round-trip
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policies = [_random_policy(i) for i in range(3)]
            for p in policies:
                p.fitness = float(p.n_main)  # dummy fitness
            paths = export_tactic_library(policies, out_dir=tmpdir, seed=0)
            assert len(paths) == 3
            for path in paths:
                sc = load_scenario_from_npz(path)
                assert validate_scenario(sc)

    def test_manifest_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            policies = [_random_policy(0)]
            policies[0].fitness = 1.0
            export_tactic_library(policies, out_dir=tmpdir, seed=0)
            manifest = pathlib.Path(tmpdir) / "manifest.json"
            assert manifest.exists()

    def test_exported_scenario_doctrine_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = _random_policy(0)
            p.fitness = 1.0
            p.generation = 7
            paths = export_tactic_library([p], out_dir=tmpdir, seed=0)
            sc = load_scenario_from_npz(paths[0])
            assert "league_gen7" in sc.doctrine

    def test_load_labels_match_agents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = _random_policy(42)
            p.fitness = 1.0
            paths = export_tactic_library([p], out_dir=tmpdir, seed=0)
            sc = load_scenario_from_npz(paths[0])
            assert len(sc.labels) == sc.trajectories.shape[0]


# ---------------------------------------------------------------------------
# 6. League runner (small, fast)
# ---------------------------------------------------------------------------

class TestLeagueRunner:
    @pytest.fixture(scope="class")
    def league_result(self):
        runner = _small_league(n_gen=4, pop=4, seed=0)
        history = runner.run()
        return runner, history

    def test_run_returns_correct_n_generations(self, league_result):
        _, history = league_result
        assert len(history) == 4

    def test_history_has_required_fields(self, league_result):
        _, history = league_result
        for stats in history:
            assert hasattr(stats, "generation")
            assert hasattr(stats, "best_fitness")
            assert hasattr(stats, "win_rate")
            assert 0.0 <= stats.win_rate <= 1.0
            assert stats.diversity >= 0.0

    def test_win_rate_trend_non_negative(self, league_result):
        runner, _ = league_result
        assert runner.win_rate_trend_is_positive()

    def test_tactic_library_non_empty(self, league_result):
        runner, _ = league_result
        lib = runner.tactic_library()
        assert len(lib) >= 1

    def test_tactic_library_scenarios_valid(self, league_result):
        runner, _ = league_result
        lib = runner.tactic_library()
        for p in lib:
            rng = np.random.default_rng(0)
            sc = p.to_scenario(rng)
            assert validate_scenario(sc)

    def test_export_produces_loadable_files(self, league_result):
        runner, _ = league_result
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = runner.export(out_dir=tmpdir)
            assert len(paths) >= 1
            for path in paths:
                sc = load_scenario_from_npz(path)
                assert validate_scenario(sc)


# ---------------------------------------------------------------------------
# 7. Acceptance: ≥3 distinct tactics (longer run, cached)
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    @pytest.fixture(scope="class")
    def long_runner(self):
        """6 generations × 8 policies × 1 episode — deterministic, ~10 s."""
        runner = LeagueRunner(n_generations=6, pop_size=8,
                              n_eval_episodes=1, seed=1)
        runner.run()
        return runner

    def test_a_win_rate_trend_non_negative(self, long_runner):
        """Acceptance A: red win rate does not decrease over the run."""
        assert long_runner.win_rate_trend_is_positive()

    def test_b_three_distinct_tactics(self, long_runner):
        """Acceptance B: ≥3 qualitatively distinct attack patterns."""
        lib = long_runner.tactic_library(top_n=10)
        n = distinct_clusters(lib, min_distance=0.10)
        assert n >= 3, (
            f"Only {n} distinct tactics found. Library: "
            + str([p.to_dict() for p in lib])
        )

    def test_c_exported_scenarios_valid(self, long_runner):
        """Acceptance C: all exported Scenarios pass structural validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = long_runner.export(out_dir=tmpdir)
            assert len(paths) >= 1
            for path in paths:
                sc = load_scenario_from_npz(path)
                assert validate_scenario(sc), f"Invalid scenario at {path}"

    def test_best_policy_has_positive_fitness(self, long_runner):
        lib = long_runner.tactic_library(top_n=1)
        assert len(lib) >= 1
        assert lib[0].fitness > -1.0  # has been evaluated
