"""
Diet module example
"""

# pyright: reportAttributeAccessIssue=false

import os
from pathlib import Path
import polars as pl

from convop import sum, Model, Variable


def main(working_dir: Path | str):
    working_dir = Path(working_dir)
    input_dir = working_dir / "input_data"

    food_cost = pl.read_csv(input_dir / "foods.csv").to_expr()
    nutrients = pl.read_csv(input_dir / "nutrients.csv")
    min_nutrient = nutrients.select(["category", "min"]).to_expr()
    max_nutrient = nutrients.select(["category", "max"]).to_expr()
    food_nutrients = pl.read_csv(input_dir / "foods_to_nutrients.csv").to_expr()

    m = Model("diet")
    m.Buy = Variable(food_cost, lb=0)

    m.con_min_nutrients = min_nutrient <= sum(
        "food", m.Buy * food_nutrients.within(min_nutrient)
    )
    m.con_max_nutrients = (
        sum("food", m.Buy * food_nutrients.within(max_nutrient)) <= max_nutrient
    )

    m.minimize = sum(m.Buy * food_cost)

    m.solve("gurobi", working_dir / "results")


if __name__ == "__main__":
    main(os.path.dirname(os.path.realpath(__file__)))
