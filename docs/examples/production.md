# Production planning

## Problem statement



This classical problem (Eiselt and Sandblom, p. 20) seeks to determine which products should be manufactured (and in what quantities) given that:

1. Each product must go through all the machines.

2. Every machine takes a different amount of time to process every product ([`processing_times.csv`](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/production_planning/input_data/processing_times.csv)).

3. Machines have a maximum lifetime before needing maintenance ([`machines_availability.csv`](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/production_planning/input_data/machines_availability.csv)).

4. Each product yields a different amount of profit ([`products_profit.csv`](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/production_planning/input_data/products_profit.csv)).

*[Eiselt and Sandblom]: H.A Eiselt and Carl-Louis Sandblom. Operations Research: A Model-Based Approach 3rd Edition, Springer, 2022.

## Model

<!-- invisible-code-block: python
import os

os.chdir(os.path.join(os.getcwd(), "tests/examples/production_planning/input_data"))
-->

```python
import pandas as pd

import pyoframe as pf


processing_times = pd.read_csv("processing_times.csv")
machines_availability = pd.read_csv("machines_availability.csv")
products_profit = pd.read_csv("products_profit.csv")

m = pf.Model()
m.Production = pf.Variable(products_profit["products"], lb=0)

machine_usage = m.Production * processing_times
m.machine_capacity = machine_usage.sum_by("machines") <= machines_availability

m.maximize = (products_profit * m.Production).sum()

m.optimize()
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