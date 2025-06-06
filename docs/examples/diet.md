# Diet Problem

## Input Data

## Model

```{.python hide}
import os
from pathlib import Path
input_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "input_data"
```

```python
import polars as pl

from pyoframe import Model, Variable, sum


def solve_model():
    # Import data
    food = pl.read_csv(input_dir / "foods.csv")
    nutrients = pl.read_csv(input_dir / "nutrients.csv")
    min_nutrient = nutrients.select(["category", "min"]).to_expr()
    max_nutrient = nutrients.select(["category", "max"]).to_expr()
    food_nutrients = pl.read_csv(input_dir / "foods_to_nutrients.csv").to_expr()

    # Construct model
    m = Model()
    m.Buy = Variable(food[["food"]], lb=0, ub=food[["food", "stock"]])

    nutrient_intake = sum_by("nutrient", m.Buy * food_nutrients)
    m.min_nutrients = min_nutrient <= nutrient_intake.drop_unmatched()
    m.max_nutrients = nutrient_intake.drop_unmatched() <= max_nutrient

    m.minimize = sum(m.Buy * food[["food", "cost"]])

    # Optimize !
    m.optimize()

    return m


if __name__ == "__main__":
    m = solve_model()
```
