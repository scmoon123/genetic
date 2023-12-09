""" 
Module to calculate the fitness of the current generation
"""
from typing import List, Tuple

import numpy as np
from numpy import ndarray

__all__ = ["_CalculateFit"]


class _CalculateFit:
    def __init__(self):
        # This class should not be initialized standalone.
        # Purpose of this class is to supply relevant functionalities to the GA class.
        pass

    def calc_fit_sort_population(
        self, current_population: ndarray
    ) -> Tuple[ndarray, ndarray]:
        """
        Calculated fitness of organisms and sorts population based on fitness score (AIC). From low AIC (best) to high.

        Inputs:
            current_population: Current population (population_size, C)
        Outputs:
            sorted_population: Sorted current_population (population_size, C)
            sorted_fitness_scores: Sorted fitness scores (population_size,)
        """
        fitness_scores: ndarray = self._calculate_fit_of_population(current_population)
        return self._sort_population(current_population, fitness_scores)

    def _sort_population(
        self, current_population, fitness_scores
    ) -> Tuple[ndarray, ndarray]:
        """
        Sorts population based on fitness score (AIC). From low AIC (best) to high.

        Inputs:
            current_population: Current population
            fitness_scores: Fitness scores per organism
        Outputs:
            sorted_population: Sorted current_population (population_size, C)
            sorted_fitness_scores: Sorted fitness scores (population_size,)
        """

        sort_index = np.argsort(fitness_scores)

        return current_population[sort_index], fitness_scores[sort_index]

    def _calculate_fit_of_population(self, current_population: ndarray) -> ndarray:
        """
        Calculates fitness of all organism in generation.

        Inputs:
            current_population: Current population
        Outputs:
            fitness_scores : Fitness score per organism
        """

        fitness_scores: List[float] = []
        for organism in current_population:
            X_trimmed: ndarray = self._select_features(organism)
            fitness_scores.append(self._calculate_fit_per_organism(X_trimmed))

        return np.array(fitness_scores)

    def _calculate_fit_per_organism(self, X_trimmed: ndarray) -> float:
        """
        Calculates fitness of one organism based on trimmed data according to its allels.

        Inputs:
            X_trimmed: Trimmed data
        Outputs:
            score: Fitness score of organism
        """
        mod_fitted = self.mod(self.y, X_trimmed).fit()
        return mod_fitted.aic

    def _select_features(self, organism: ndarray) -> ndarray:
        """
        Drops non-relevant features from data based on allels of an organism.
        (Filter `self.X` by `organism` bool idx)

        Inputs:
            organism: Single organism (1, C)
        Outputs:
            X_trimmed: Data to be used for fitness calculation of this organism (num_samples, sampled_C)
                       (sampled_C < sample)
        """
        # NOTE: self.X is numpy not pd.DataFrame => using numpy api instead
        # X_trimmed = self.X.drop(columns=self.X.columns[organism == 0], axis=1)
        X_trimmed = self.X[:, organism != 0]
        return X_trimmed
