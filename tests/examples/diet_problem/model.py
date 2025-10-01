"""Example Pyoframe formulation of the classic diet problem."""

import os
from pathlib import Path

import polars as pl

import pyoframe as pf

_input_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "input_data"


def solve_model(use_var_names):
    food = pl.read_csv(_input_dir / "foods.csv")
    nutrients = pl.read_csv(_input_dir / "nutrients.csv")
    min_nutrient = nutrients.select(["category", "min"]).to_expr()
    max_nutrient = nutrients.select(["category", "max"]).to_expr()
    food_nutrients = pl.read_csv(_input_dir / "foods_to_nutrients.csv").to_expr()

    m = pf.Model(solver_uses_variable_names=use_var_names)
    m.Buy = pf.Variable(food["food"], lb=0, ub=food[["food", "stock"]])

    m.min_nutrients = min_nutrient <= (m.Buy * food_nutrients).sum("food").drop_extras()
    m.max_nutrients = (m.Buy * food_nutrients).sum("food").drop_extras() <= max_nutrient

    m.minimize = (m.Buy * food[["food", "cost"]]).sum()

    m.optimize()

    return m


if __name__ == "__main__":
    solve_model()
