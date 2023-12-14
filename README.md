# Genetic-Algorithm
STAT 243 Final: genetic algorithm for variable selection

## Run example.py
```
# this will test GA algoithm on Abalone dataset with GLM, OLS, Poisson
sh examples.sh
```
type `python3 example.py --help` to see available flag options


## Run test_main.py
```
pytest test_main.py -v
```
modify `params` inside `@pytest.fixture(autouse=True, scope="module", params=[_ for _ in range(41, 71)])` 
in order to test cases on different seed values.
