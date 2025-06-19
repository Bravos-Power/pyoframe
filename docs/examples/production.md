# Production planning

This classical problem seeks to determine which products should be manufactured (and in what quantities) given that a) each product must go through a sequence of machines, b) every machine takes a different amount of time to process every product, and c) machines have a maximal lifetime before needing maintenance. (See Eiselt and Sandblom, p. 20, for details.)

*[Eiselt and Sandblom]: H.A Eiselt and Carl-Louis Sandblom. Operations Research: A Model-Based Approach 3rd Edition, Springer, 2022.

## Input Data

- [machines_availability.csv](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/production_planning/input_data/machines_availability.csv)
- [processing_times.csv](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/production_planning/input_data/processing_times.csv)
- [products_profit.csv](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/production_planning/input_data/products_profit.csv)

## Model

<!-- invisible-code-block: python
import os

os.chdir(os.path.join(os.getcwd(), "tests/examples/production_planning/input_data"))
-->

```python
import polars as pl

import pyoframe as pf


def solve_model():
    processing_times = pl.read_csv("processing_times.csv")
    machines_availability = pl.read_csv("machines_availability.csv")
    products_profit = pl.read_csv("products_profit.csv")

    m = pf.Model()
    m.Production = pf.Variable(products_profit[["products"]], lb=0)

    machine_usage = m.Production * processing_times
    m.machine_capacity = pf.sum_by("machines", machine_usage) <= machines_availability

    m.maximize = pf.sum(products_profit * m.Production)

    m.optimize()

    return m


m = solve_model()
```

```pycon
>>> m.Production.solution
┌──────────┬──────────┐
│ products ┆ solution │
│ ---      ┆ ---      │
│ i64      ┆ f64      │
╞══════════╪══════════╡
│ 1        ┆ 20.0     │
│ 2        ┆ 0.0      │
│ 3        ┆ 120.0    │
└──────────┴──────────┘

```