# Portfolio optimization

## Problem statement

Given a list of [assets](https://github.com/Bravos-Power/pyoframe/tree/main/tests/examples/portfolio_optim/input_data/assets.csv), their [covariance](https://github.com/Bravos-Power/pyoframe/tree/main/tests/examples/portfolio_optim/input_data/covariance.csv), and some [portfolio parameters](https://github.com/Bravos-Power/pyoframe/tree/main/tests/examples/portfolio_optim/input_data/portfolio_params.csv), select the portfolio weights that minimize risk (i.e. variance) while achieving a target return.

## Solution

<!-- invisible-code-block: python
import os

os.chdir(os.path.join(os.getcwd(), "tests/examples/portfolio_optim/input_data"))
-->

```python
import pandas as pd
import pyoframe as pf

# Read input data
assets = pd.read_csv("assets.csv").set_index("asset")
covariance = pd.read_csv("covariance.csv").set_index(["asset_i", "asset_j"])
params = pd.read_csv("portfolio_params.csv").set_index("param")["value"]

m = pf.Model()

# Decision variables: portfolio weights
m.weight = pf.Variable(assets.index, lb=0, ub=params.loc["max_weight"])

# Constraint: weights must sum to 1
m.con_weights_sum = m.weight.sum() == 1

# Constraint: minimum expected return
m.con_min_return = (m.weight * assets["expected_return"]).sum() >= params.loc[
    "min_return"
]

# Objective: minimize portfolio variance (quadratic)
# Variance = sum over i,j of weight_i * cov_ij * weight_j
# We use 'rename' to make the dimensions match properly
m.minimize = (
    m.weight.rename({"asset": "asset_i"})
    * covariance["covariance"]
    * m.weight.rename({"asset": "asset_j"})
).sum()

m.optimize()
```

And the result should be:
```pycon
>>> m.weight.solution
┌───────┬──────────┐
│ asset ┆ solution │
│ ---   ┆ ---      │
│ str   ┆ f64      │
╞═══════╪══════════╡
│ A     ┆ 0.36067  │
│ B     ┆ 0.147212 │
│ C     ┆ 0.209338 │
│ D     ┆ 0.145308 │
│ E     ┆ 0.137472 │
└───────┴──────────┘
>>> m.objective.value
0.0195

```
