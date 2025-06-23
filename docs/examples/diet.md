# Diet Problem

Given a dataset of food options, each with different nutritional contents, how do you satisfy your dietary requirements while minimizing cost?

## Input Data

- [foods.csv](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/diet_problem/input_data/foods.csv)
- [foods_to_nutrients.csv](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/diet_problem/input_data/foods_to_nutrients.csv)
- [nutrients.csv](https://github.com/Bravos-Power/pyoframe/blob/main/tests/examples/diet_problem/input_data/nutrients.csv)

## Model

<!-- invisible-code-block: python
import os

os.chdir(os.path.join(os.getcwd(), "tests/examples/diet_problem/input_data"))
-->

```python
import polars as pl

from pyoframe import Model, Variable, sum, sum_by


def solve_model():
    # Import data
    food = pl.read_csv("foods.csv")
    nutrients = pl.read_csv("nutrients.csv")
    min_nutrient = nutrients.select(["category", "min"]).to_expr()
    max_nutrient = nutrients.select(["category", "max"]).to_expr()
    food_nutrients = pl.read_csv("foods_to_nutrients.csv").to_expr()

    # Construct model
    m = Model()
    m.Buy = Variable(food[["food"]], lb=0, ub=food[["food", "stock"]])

    nutrient_intake = sum_by("category", m.Buy * food_nutrients)
    m.min_nutrients = min_nutrient <= nutrient_intake.drop_unmatched()  # (1)!
    m.max_nutrients = nutrient_intake.drop_unmatched() <= max_nutrient

    m.minimize = sum(m.Buy * food[["food", "cost"]])

    m.optimize()

    return m


m = solve_model()
```

1. `.drop_unmatched()` ensures that if `min_nutrient` is `null` for certain foods, no constraint will be created for those foods. [Learn more](../learn/getting-started/special-functions.md#drop_unmatched-and-keep_unmatched)

So what should you eat...

```pycon
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