"""Mutual co-evolution (PM) — Blue adapts back to discovered Red tactics.

PILLAR C (PL) evolved Red against a fixed Blue and showed the league
discovers attacks the scripted doctrines never anticipated. The open question
PX left: does letting Blue *adapt* to those discovered attacks improve the
cost-exchange figure? This module answers it.

The arc (Coevolution.run):
  1. Seed a Red tactic library by running a short PL league vs the default Blue.
  2. Measure the DEFAULT Blue's cost-exchange against that library (baseline).
  3. Search the Blue policy space (effector loadout + rationing knob) for the
     Blue that minimises cost-per-intercept against the same library (adapted).
  4. Let Red COUNTER-EVOLVE one short ES run against the adapted Blue.
  5. Re-measure the adapted Blue against the counter-evolved Red.

The headline is the cost-exchange improvement of the adapted Blue over the
default Blue, and how much of it Red claws back by counter-evolving.

SCOPE: simulation only. A Blue "loadout" is a resource-allocation choice over
effector parameter sets, not a hardware action.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence, Tuple

import numpy as np

from eval.monte_carlo import run_mc_episode
from league.blue import BluePolicy, enumerate_blue_policies
from league.es import EvolutionStrategy
from league.league import LeagueRunner
from league.policy import SwarmPolicy
from sim.alloc.economic_mdp import EconomicMDP

# Blue's objective weights: minimise cost-per-intercept, penalise leakers.
LEAK_WEIGHT = 4.0


@dataclasses.dataclass
class BlueEvaluation:
    blue: BluePolicy
    mean_cost_exchange_ratio: float
    mean_intercept_rate: float
    mean_leakers: float
    mean_defense_spend_usd: float
    blue_score: float            # lower is better

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["blue"] = self.blue.to_dict()
        return d


def _blue_score(cer: float, leakers: float) -> float:
    return cer + LEAK_WEIGHT * leakers


class Coevolution:
    def __init__(
        self,
        red_library: Optional[List[SwarmPolicy]] = None,
        n_red_tactics: int = 3,
        league_n_gen: int = 4,
        league_pop: int = 6,
        league_seed: int = 7,
        blue_lambda_choices: Optional[Tuple[float, ...]] = None,
        search_seeds: Sequence[int] = (0,),
        confirm_seeds: Sequence[int] = (0, 1, 2),
        counter_n_gen: int = 2,
        counter_pop: int = 4,
        seed: int = 0,
    ) -> None:
        self._red_library = red_library
        self.n_red_tactics = n_red_tactics
        self.league_n_gen = league_n_gen
        self.league_pop = league_pop
        self.league_seed = league_seed
        self.blue_lambda_choices = blue_lambda_choices
        self.search_seeds = tuple(search_seeds)
        self.confirm_seeds = tuple(confirm_seeds)
        self.counter_n_gen = counter_n_gen
        self.counter_pop = counter_pop
        self.seed = seed
        self.allocator = EconomicMDP()

    # ------------------------------------------------------------------ #
    # Red library seeding                                                 #
    # ------------------------------------------------------------------ #

    def seed_red_library(self) -> List[SwarmPolicy]:
        if self._red_library is not None:
            return self._red_library
        runner = LeagueRunner(
            n_generations=self.league_n_gen,
            pop_size=self.league_pop,
            n_eval_episodes=1,
            seed=self.league_seed,
        )
        runner.run()
        lib = runner.tactic_library(top_n=self.n_red_tactics)
        self._red_library = lib
        return lib

    # ------------------------------------------------------------------ #
    # Blue evaluation                                                     #
    # ------------------------------------------------------------------ #

    def evaluate_blue(
        self,
        blue: BluePolicy,
        red_library: List[SwarmPolicy],
        seeds: Sequence[int],
    ) -> BlueEvaluation:
        cers, irs, leaks, spends = [], [], [], []
        for ri, red in enumerate(red_library):
            for s in seeds:
                ep = run_mc_episode(
                    seed=s + 100 * ri,
                    allocator=self.allocator,
                    allocator_name="EconomicMDP",
                    scenario_label=f"red_{ri}",
                    policy=red,
                    lambda_cost=blue.lambda_cost,
                    loadout=blue.loadout_list(),
                )
                cers.append(ep.cost_exchange_ratio)
                irs.append(ep.intercept_rate)
                leaks.append(ep.leakers)
                spends.append(ep.defense_spend_usd)
        cer = float(np.mean(cers))
        leak = float(np.mean(leaks))
        return BlueEvaluation(
            blue=blue,
            mean_cost_exchange_ratio=round(cer, 4),
            mean_intercept_rate=round(float(np.mean(irs)), 4),
            mean_leakers=round(leak, 4),
            mean_defense_spend_usd=round(float(np.mean(spends)), 2),
            blue_score=round(_blue_score(cer, leak), 4),
        )

    def optimize_blue(
        self,
        red_library: List[SwarmPolicy],
    ) -> Tuple[BluePolicy, List[BlueEvaluation]]:
        """Exhaustive search over the Blue policy space against the Red library.

        Search uses `search_seeds` (cheap); the winner is what run() then
        confirms on `confirm_seeds`.
        """
        kwargs = {}
        if self.blue_lambda_choices is not None:
            kwargs["lambda_choices"] = self.blue_lambda_choices
        candidates = enumerate_blue_policies(**kwargs)

        scored = [self.evaluate_blue(b, red_library, self.search_seeds)
                  for b in candidates]
        scored.sort(key=lambda e: e.blue_score)
        return scored[0].blue, scored

    # ------------------------------------------------------------------ #
    # Red counter-evolution against a fixed Blue                          #
    # ------------------------------------------------------------------ #

    def _red_fitness(self, red: SwarmPolicy, blue: BluePolicy,
                     seeds: Sequence[int]) -> float:
        """Red wants to leak through / damage the asset against this Blue."""
        scores = []
        for s in seeds:
            ep = run_mc_episode(
                seed=s,
                allocator=self.allocator,
                allocator_name="EconomicMDP",
                scenario_label="counter",
                policy=red,
                lambda_cost=blue.lambda_cost,
                loadout=blue.loadout_list(),
            )
            damage_norm = ep.asset_damage_usd / 1_000_000.0
            scores.append(ep.leakers * 10.0 + damage_norm * 5.0 - ep.intercepts)
        return float(np.mean(scores))

    def counter_evolve_red(self, blue: BluePolicy) -> List[SwarmPolicy]:
        """Short ES of Red against a fixed (adapted) Blue."""
        es = EvolutionStrategy(pop_size=self.counter_pop, seed=self.seed + 1)
        es.seed_population()
        all_evaluated: List[SwarmPolicy] = []
        for _ in range(self.counter_n_gen):
            pop = es.population
            fits = []
            for red in pop:
                f = self._red_fitness(red, blue, seeds=(self.seed,))
                red.fitness = f
                fits.append(f)
            all_evaluated.extend(list(pop))
            es.step(fits)
        # Top tactics by fitness against the adapted Blue
        ranked = sorted(all_evaluated, key=lambda p: p.fitness, reverse=True)
        return ranked[: self.n_red_tactics]

    # ------------------------------------------------------------------ #
    # Full arc                                                            #
    # ------------------------------------------------------------------ #

    def run(self, counter_evolve: bool = True) -> dict:
        red_lib = self.seed_red_library()

        baseline = self.evaluate_blue(
            BluePolicy.default(), red_lib, self.confirm_seeds)

        adapted_blue, search = self.optimize_blue(red_lib)
        adapted = self.evaluate_blue(adapted_blue, red_lib, self.confirm_seeds)

        cer_improvement = 0.0
        if baseline.mean_cost_exchange_ratio > 0:
            cer_improvement = round(
                (baseline.mean_cost_exchange_ratio
                 - adapted.mean_cost_exchange_ratio)
                / baseline.mean_cost_exchange_ratio, 4)

        result = {
            "baseline_blue": baseline.to_dict(),
            "adapted_blue": adapted.to_dict(),
            "cer_improvement_frac": cer_improvement,
            "blue_search_size": len(search),
            "red_library_size": len(red_lib),
        }

        if counter_evolve:
            counter_red = self.counter_evolve_red(adapted_blue)
            counter = self.evaluate_blue(
                adapted_blue, counter_red, self.confirm_seeds)
            clawback = round(
                counter.mean_cost_exchange_ratio
                - adapted.mean_cost_exchange_ratio, 4)
            result["counter_evolved"] = counter.to_dict()
            result["red_clawback_cer"] = clawback

        return result
