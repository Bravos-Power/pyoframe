"""Example Pyoframe formulation of the classic diet problem."""

import os
from pathlib import Path

import polars as pl

from pyoframe import Model, Variable, sum

_input_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "input_data"


def solve_model(use_var_names=False):
    food = pl.read_csv(_input_dir / "foods.csv")
    nutrients = pl.read_csv(_input_dir / "nutrients.csv")
    min_nutrient = nutrients.select(["category", "min"]).to_expr()
    max_nutrient = nutrients.select(["category", "max"]).to_expr()
    food_nutrients = pl.read_csv(_input_dir / "foods_to_nutrients.csv").to_expr()

    m = Model(use_var_names=use_var_names)
    m.Buy = Variable(food["food"], lb=0, ub=food[["food", "stock"]])

    m.min_nutrients = (
        min_nutrient <= sum("food", m.Buy * food_nutrients).drop_unmatched()
    )
    m.max_nutrients = (
        sum("food", m.Buy * food_nutrients).drop_unmatched() <= max_nutrient
    )

    m.minimize = sum(m.Buy * food[["food", "cost"]])

    m.optimize()

    return m


if __name__ == "__main__":
    solve_model()
