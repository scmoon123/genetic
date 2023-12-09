import argparse

import pandas as pd
import statsmodels.api as sm

from GA import GA


def load_dataset(file_dir: str):
    data = pd.read_csv(file_dir, delimiter=" ")
    print("Shape of data: ", data.shape)

    X = data.drop("salary", axis=1)
    y = data["salary"]

    return X, y


def main(config):
    X, y = load_dataset(config.data)

    GA = GA(pop_size=26, X=X, y=y, mod=sm.OLS, max_iter=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA example", add_help=True)

    # fmt: off
    parser.add_argument("--data", default="baseball.dat", type=str, help="csv dataset path (default: baseball.dat)")

    opt = parser.parse_args()
    main(opt)
