import random
from typing import Callable, Union

import numpy as np
from numpy import ndarray

from utils import CalculateFit, CrossOver, Mutation, ParentSelection


class GA(
    CalculateFit,
    ParentSelection,
    CrossOver,
    Mutation,
):
    supported_int_types = Union[
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]
    supported_float_types = Union[float, np.float16, np.float32, np.float64]

    def __init__(
        self,
        X: ndarray,
        y: ndarray,
        mod: Callable,
        max_iter: int,
        pop_size: int = None,  # type: ignore
        # fitness_func = "AIC",
        starting_population: int = None,  # type: ignore
        mutate_prob: float = 0.01,
        save_sols: bool = False,
        random_seed: int = None,  # type: ignore
    ):
        """
        parameters:
            X: input feature
            y: label
            mod: regression method
            max_iter: max iteration for ..
            pop_size: ...
            starting_population: GA start popluation
            mutate_prob: GA mutate probability
            save_sols: ...
            random_seed: random_seed
        """
        self.random_seed: int = random_seed
        if random_seed:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        # C: feature_size (TODO: rmv comments)
        self.C: int = X.shape[1]  # CHECK: this is assuming intercept column

        if pop_size is None:
            self.pop_size: int = int(1.5 * self.C)  # C < P < 2C
        else:
            self.pop_size: int = pop_size

        self.X: ndarray = X
        self.y: ndarray = y
        self.mod: Callable = mod
        self.max_iter: int = max_iter
        self.mutate_prob: float = mutate_prob
        # self.fitness_func = fitness_func
        self.starting_population: ndarray = starting_population
        self.current_population = None

        if save_sols is True:
            # Pre-specify matrix for storing solutions
            self.solutions_matrix = np.zeros((self.max_iter, self.C))
        else:
            pass

    def initialize_pop(self) -> ndarray:
        """
        Creates the starting population
        returns:
            starting_population: ndarray (random bool matrix used to sample self.X)
        """
        if not isinstance(self.starting_population, ndarray):
            # If self.starting_population not initialized (e.g. None)
            # Specify a starting pop

            rows: int = self.pop_size
            if rows % 2 == 1:  # If pop_size is odd
                # Only allow even number for population size
                self.pop_size = self.pop_size + 1

            cols: int = self.C

            # Complete random generation
            self.starting_population = np.random.choice([0, 1], size=(rows, cols))
        else:
            pass

        # Replace chromosome of all zeros
        self.starting_population = self.replace_zero_chromosome(
            self.starting_population
        )

        return self.starting_population

    def select(self, operator_list: List[Callable] = [GA.random_mutate]):
        """
        Runs variable selection based on a user-defined genetic operator sequence: operator_list
        """
        import pdb

        pdb.set_trace()
        starting_pop: ndarray = self.initialize_pop()
        current_pop: ndarray = starting_pop.copy()

        for _ in range(self.max_iter):
            """Calculates fitness and pairs parents"""
            # chrom_ranked: ordered bool matrix(current_pop) from the fittest to unfittest
            chrom_ranked, fitness_val = self.calc_fit_sort_population(current_pop)
            parents = self.select_from_fitness_rank(chrom_ranked)
            current_pop = parents  # update current_pop's chromosoe

            """Runs genetic operator sequence"""
            for method in operator_list:
                new_population = method(current_pop)
                current_pop = new_population
            # Check if any chromosome of zeros and replace the row
            current_pop = self.replace_zero_chromosome(current_pop)

        final_pop = current_pop.copy()
        self.final_pop_sorted, self.final_fitness_val = self.calc_fit_sort_population(
            final_pop
        )

        return (self.final_pop_sorted[0], self.final_fitness_val[0])

    def replace_zero_chromosome(self, population: ndarray):  # TODO: vectorize
        """
        Finds if any chromosome is all zeros, and replaces the zero rows with random 0,1s
        """
        while np.any((population == 0).all(axis=1)):
            # Find the indices of rows with all zeros
            zero_rows_indices = np.where((population == 0).all(axis=1))[0]

            # Replace each zero row with a randomly generated 0,1 row
            for row_index in zero_rows_indices:
                population[row_index] = np.random.randint(0, 2, self.C)

        return population
