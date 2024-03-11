"""
Diet module example
"""

import os
from pathlib import Path


import convop as cp
import numpy as np


def main():
    m = cp.Model("diet_model")

    working_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    input_dir = working_dir / "input_data"

    food_cost = cp.Parameter(input_dir / "foods.csv")
    nutrients = cp.Parameters(
        input_dir / "nutrients.csv",
        dim=1,
        defaults={"min_amount": 0, "max_amount": np.inf},
    )
    nutrients_per_food = cp.Parameter(input_dir / "foods_to_nutrients.csv")

    m.BuyFood = cp.Variable(food_cost, lb=0)

    m.MinNutrients = cp.Constraint(
        nutrients["min_amount"] <= cp.sum("food", m.BuyFood * nutrients_per_food)
    )
    m.MaxNutrients = cp.Constraint(
        cp.sum("food", m.BuyFood * nutrients_per_food) <= nutrients["max_amount"]
    )

    m.Cost = cp.Objective(cp.sum("food", food_cost * m.BuyFood), sense="min")

    m.solve("gurobi", working_dir / "results")


if __name__ == "__main__":
    main()
