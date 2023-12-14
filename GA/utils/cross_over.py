""" 
Module to perform crossover (genetic operator) to the current generation
"""

import numpy as np
from numpy import ndarray

__all__ = ["_CrossOver"]


class _CrossOver:
    def __init__(self):
        # This class should not be initialized standalone.
        # Purpose of this class is to supply relevant functionalities to the GA class.
        pass

    @staticmethod
    def split_and_glue_population(current_population: ndarray):
        """
        Performs split-and-glue crossover to current population (assuming 1&2 is paired, 3&4, etc.)

        Inputs: Current population
        Outputs: Population of children (pairwise cross-over)
        """
        # shuffle population order
        _idx = np.random.rand(current_population.shape[0]).argsort(axis=0)  # type: ignore
        new_population = current_population[_idx]

        _ = current_population.shape[0] // 2
        new_population = _CrossOver._split_and_glue(
            new_population[:_, :], new_population[_:, :]
        )

        # NOTE: vectorized solution is avialble
        # count = 0
        # `self.pop_size` is always even
        # for pair in np.arange(current_population.shape[0] // 2):
        #     (
        #         new_population[count],
        #         new_population[count + 1],
        #     ) = _CrossOver._split_and_glue(
        #         current_population[count], current_population[count + 1]
        #     )
        #     count += 2
        return new_population

    @staticmethod
    def _split_and_glue(parent1, parent2):
        """
        Crossover two parents to create two children.
        The method used here is a simple split and glue approach.
        The split position is randomly created.

        Inputs: Two parent organisms
        Outputs: Two child organisms (crossed-over)
        """
        _idx = np.zeros_like(parent1).astype(bool)
        for _ in _idx:
            _cross_over_pt = np.clip(np.random.normal(0.5, 0.1), 0, 1)
            _[: int(_cross_over_pt * parent1.shape[-1])] = 1

        child1 = parent1 * _idx + parent2 * (1 - _idx)
        child2 = parent2 * _idx + parent1 * (1 - _idx)
        return np.concatenate([child1, child2], axis=0)
        # NOTE: vectorized solution is avialble
        # cut_idx = np.random.randint(0, len(parent1))
        # child1 = np.concatenate((parent1[0:cut_idx], parent2[cut_idx:]))
        # child2 = np.concatenate((parent2[0:cut_idx], parent1[cut_idx:]))
        # return child1, child2

    # TODO: vectorize as split_and_glue_population
    @staticmethod
    def random_allel_selection_population(current_population):
        """
        Performs random allel selection crossover to current population (assuming 1&2 is paired, 3&4, etc.)

        Inputs: Current population
        Outputs: Population of children (pairwise cross-over)
        """
        count = 0
        new_population = np.zeros(current_population.shape)
        for pair in np.arange(int(current_population.shape[0] / 2)):
            (
                new_population[count],
                new_population[count + 1],
            ) = _CrossOver._random_allel_selection(
                current_population[count], current_population[count + 1]
            )
            count += 2
        return new_population

    @staticmethod
    def _random_allel_selection(parent1, parent2):
        """
        Crossover two parents to create two children.
        The method randomly selects an allel from one of the parents per loci.

        Inputs: Two parent organisms
        Outputs: Two child organisms (crossed-over)
        """
        rng = np.random.default_rng()

        allel_selector = rng.binomial(1, 0.5, size=parent1.shape[0])
        allel_selector_reversed = 1 - allel_selector

        child1 = allel_selector * parent1 + allel_selector_reversed * parent2
        child2 = allel_selector * parent2 + allel_selector_reversed * parent1
        return child1, child2
