import argparse
from typing import Tuple

import pandas as pd
import statsmodels.api as sm
from numpy import ndarray

from GA import GA


def load_dataset(file_dir: str) -> Tuple[ndarray, ndarray]:
    data = pd.read_csv(file_dir, delimiter=" ")
    print("Shape of data: ", data.shape)

    X = data.drop("salary", axis=1)
    y = data["salary"]

    return X.values, y.values


def main(config):
    X, y = load_dataset(config.data)

    # fmt: off
    ga = GA(
        pop_size=config.population_size, 
        X=X, 
        y=y, 
        mod=getattr(sm, config.fitness_fn), 
        max_iter=config.max_iter,
        mutate_prob=0.1
    )
    # fmt: on
    best_solution, best_score = ga.select(
        [getattr(GA, config.cross_over_fn), GA.random_mutate]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA example", add_help=True)

    # fmt: off
    parser.add_argument("--data", default="assets/baseball.dat", type=str, help="csv dataset path (default: baseball.dat)")
    parser.add_argument("--population_size", default=1000, type=int, help="population size used for genetic algorithm (default: 100)")
    parser.add_argument("--max_iter", default=100, type=int, help="genetic algorithm iteration steps (default: 100)")
    parser.add_argument("--mutate_prob", default=0.1, type=float, help="genetic algorithm mutation probability (default: 0.1)")
    parser.add_argument("--fitness_fn", default="OLS", type=str, help="objective function type (default: OLS)")
    parser.add_argument("--cross_over_fn", default="split_and_glue_population", type=str, help="`split_and_glue_population` or `random_allel_selection_population` (default: split_and_glue_population)")

    opt = parser.parse_args()
    main(opt)
