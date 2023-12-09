import numpy as np
from numpy import ndarray

__all__ = ["_ParentSelection"]


class _ParentSelection:
    def __init__(self):
        # This class should not be initialized standalone.
        # Purpose of this class is to supply relevant functionalities to the GA class.
        pass

    def _calculate_phi(self):  # NOTE: removed unused `current_population`
        """
        Calculate selection probability as 2r_i/P(P+1)
        """
        P: int = self.pop_size  # Best ranked is P, then P-1, P-2,...,1
        rs = np.arange(P, 0, -1)
        phi = 2 * rs / (P * (P + 1))

        return phi

    def select_from_fitness_rank(self, current_population: ndarray) -> ndarray:
        """
        Choose parents based on fitness ranks
        returns:
            chosen_individuals: sampled parent chromosomes (self.pop_size, self.C)
        """
        selection_prob: ndarray = self._calculate_phi()
        row_idx = np.arange(len(current_population))

        # sample population idx via `selection_prob`
        chosen_rows = np.random.choice(
            row_idx, size=self.pop_size, p=selection_prob, replace=True
        )
        chosen_individuals: ndarray = current_population[chosen_rows]

        return chosen_individuals
