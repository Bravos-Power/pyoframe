"""
Diet module example
"""

# pyright: reportAttributeAccessIssue=false

import os
from pathlib import Path
import polars as pl

from pyoframe import sum, Model, Variable


def main(input_dir, output_dir: Path):
    food_cost = pl.read_csv(input_dir / "foods.csv").to_expr()
    nutrients = pl.read_csv(input_dir / "nutrients.csv")
    min_nutrient = nutrients.select(["category", "min"]).to_expr()
    max_nutrient = nutrients.select(["category", "max"]).to_expr()
    food_nutrients = pl.read_csv(input_dir / "foods_to_nutrients.csv").to_expr()

    m = Model()
    m.Buy = Variable(food_cost, lb=0)

    m.min_nutrients = (
        min_nutrient <= sum("food", m.Buy * food_nutrients).drop_unmatched()
    )
    m.max_nutrients = (
        sum("food", m.Buy * food_nutrients).drop_unmatched() <= max_nutrient
    )

    m.minimize = sum(m.Buy * food_cost)

    gurobi_model = m.solve("gurobi", output_dir)
    return gurobi_model.getObjective().getValue()


if __name__ == "__main__":
    working_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    main(working_dir / "input_data", working_dir / "results")
