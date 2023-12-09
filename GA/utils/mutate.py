""" 
Mutation module with genetic operator "mutation"
"""

import numpy as np
from numpy import ndarray

__all__ = ["_Mutation"]


class _Mutation:
    def __init__(self):
        # This class should not be initialized standalone.
        # Purpose of this class is to supply relevant functionalities to the GA class.
        pass

    def random_mutate(self, current_population: ndarray):
        """
        Randomly switches genes (bit switch) in generation with probability mutate_prob

        Inputs:
            current_population: Generation of organisms (population_size, C)
        Outputs: Generation of mutated organisms of same size
        """
        # initialize random generator
        rng = np.random.default_rng()

        population_new = current_population.copy()
        mutation_locations = rng.binomial(
            1, self.mutate_prob, size=current_population.shape
        )
        mask = mutation_locations == 1
        population_new[mask] = 1 - population_new[mask]  # flip bits using the mask
        return population_new
