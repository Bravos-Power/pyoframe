# Diet problem

## Problem statement

Given a list of potential foods, their costs, and their availability ([`foods.csv`](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/diet_problem/input_data/foods.csv)), and a list of the nutrients (e.g., protein, fats, etc.) contained in each food ([`foods_to_nutrients.csv`](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/diet_problem/input_data/foods_to_nutrients.csv)), how can you satisfy your dietary requirements ([`nutrients.csv`](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/diet_problem/input_data/nutrients.csv)) while minimizing total costs?

## Model

<!-- invisible-code-block: python
import os

os.chdir(os.path.join(os.getcwd(), "tests/examples/diet_problem/input_data"))
-->

```python
import pandas as pd

import pyoframe as pf

# Import data
food = pd.read_csv("foods.csv")
nutrients = pd.read_csv("nutrients.csv")
food_nutrients = pd.read_csv("foods_to_nutrients.csv")

# Construct model
m = pf.Model()
m.Buy = pf.Variable(food["food"], lb=0, ub=food[["food", "stock"]])

nutrient_intake = (m.Buy * food_nutrients).sum_by("category")
m.min_nutrients = (
    nutrients[["category", "min"]] <= nutrient_intake.drop_extras()  # (1)!
)
m.max_nutrients = nutrient_intake.drop_extras() <= nutrients[["category", "max"]]

total_cost = (m.Buy * food[["food", "cost"]]).sum()
m.minimize = total_cost

# Solve model
m.optimize()
```

1. `.drop_extras()` ensures that if `min_nutrient` is `null` for certain foods, no constraint will be created for those foods. [Learn more](../learn/concepts/addition.md)

So the solution is...

```pycon
>>> total_cost.evaluate()
12.060249999999998
>>> m.Buy.solution
┌───────────┬──────────┐
│ food      ┆ solution │
│ ---       ┆ ---      │
│ str       ┆ f64      │
╞═══════════╪══════════╡
│ hamburger ┆ 0.555263 │
│ chicken   ┆ 0.0      │
│ hot_dog   ┆ 0.0      │
│ fries     ┆ 0.0      │
│ macaroni  ┆ 0.0      │
│ pizza     ┆ 0.0      │
│ salad     ┆ 0.0      │
│ milk      ┆ 6.8      │
│ ice_cream ┆ 2.909211 │
└───────────┴──────────┘

```

Not a very balanced diet :thinking:.