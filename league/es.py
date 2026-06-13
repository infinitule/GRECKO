"""Lightweight Evolution Strategy for SwarmPolicy optimisation.

(μ+λ)-ES with truncation selection and sigma annealing.

μ = elite count (top-k survive unchanged)
λ = offspring count (mutants + crossovers fill the rest of the population)
"""
from __future__ import annotations

from typing import List

import numpy as np

from league.policy import SwarmPolicy


class EvolutionStrategy:
    """Population-based ES: select elites, generate offspring via mutation."""

    def __init__(
        self,
        pop_size: int = 10,
        elite_frac: float = 0.30,
        sigma: float = 0.08,
        sigma_decay: float = 0.97,
        crossover_prob: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.crossover_prob = crossover_prob
        self.rng = np.random.default_rng(seed)
        self._population: List[SwarmPolicy] = []
        self._generation = 0

    def seed_population(
        self, policies: List[SwarmPolicy] | None = None
    ) -> None:
        if policies is not None:
            self._population = list(policies)
        else:
            self._population = [
                SwarmPolicy.random(self.rng, policy_id=i)
                for i in range(self.pop_size)
            ]

    @property
    def population(self) -> List[SwarmPolicy]:
        return list(self._population)

    @property
    def generation(self) -> int:
        return self._generation

    def step(self, fitness_scores: List[float]) -> List[SwarmPolicy]:
        """Rank → select elites → generate offspring → advance generation."""
        assert len(fitness_scores) == len(self._population)

        ranked = sorted(
            zip(fitness_scores, self._population),
            key=lambda x: x[0],
            reverse=True,
        )
        for fit, pol in ranked:
            pol.fitness = fit

        n_elite = max(1, int(self.pop_size * self.elite_frac))
        elites = [p for _, p in ranked[:n_elite]]

        new_pop: List[SwarmPolicy] = []
        # Elites survive unchanged
        for p in elites:
            p.generation = self._generation + 1
            new_pop.append(p)

        # Fill with mutants (+ optional crossover)
        pid = len(new_pop)
        while len(new_pop) < self.pop_size:
            parent_a = elites[self.rng.integers(0, len(elites))]
            if (len(elites) >= 2
                    and self.rng.random() < self.crossover_prob):
                parent_b = elites[self.rng.integers(0, len(elites))]
                child = parent_a.crossover(parent_b, self.rng)
            else:
                child = parent_a.mutate(self.rng, self.sigma)
            child.generation = self._generation + 1
            child.policy_id = pid
            pid += 1
            new_pop.append(child)

        self._population = new_pop
        self._generation += 1
        self.sigma = max(0.01, self.sigma * self.sigma_decay)
        return new_pop

    def best(self) -> SwarmPolicy:
        return max(self._population, key=lambda p: p.fitness)
