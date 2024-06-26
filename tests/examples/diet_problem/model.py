"""
Diet module example
"""

# pyright: reportAttributeAccessIssue=false

import os
from pathlib import Path
import polars as pl

from pyoframe import sum, Model, Variable


def main(input_dir, directory, **kwargs):
    food = pl.read_csv(input_dir / "foods.csv")
    nutrients = pl.read_csv(input_dir / "nutrients.csv")
    min_nutrient = nutrients.select(["category", "min"]).to_expr()
    max_nutrient = nutrients.select(["category", "max"]).to_expr()
    food_nutrients = pl.read_csv(input_dir / "foods_to_nutrients.csv").to_expr()

    m = Model("min")
    m.Buy = Variable(food[["food"]], lb=0, ub=food[["food", "stock"]])

    m.min_nutrients = (
        min_nutrient <= sum("food", m.Buy * food_nutrients).drop_unmatched()
    )
    m.max_nutrients = (
        sum("food", m.Buy * food_nutrients).drop_unmatched() <= max_nutrient
    )

    m.objective = sum(m.Buy * food[["food", "cost"]])

    m.solve("gurobi", directory=directory, **kwargs)

    # Write results to CSV files
    m.Buy.solution.write_csv(directory / "buy.csv")  # type: ignore
    m.min_nutrients.dual.write_csv(directory / "min_nutrients.csv")  # type: ignore
    m.max_nutrients.dual.write_csv(directory / "max_nutrients.csv")  # type: ignore

    return m


if __name__ == "__main__":
    working_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    main(
        working_dir / "input_data",
        directory=working_dir / "results",
        use_var_names=True,
    )
