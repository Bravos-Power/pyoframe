import os
from pathlib import Path

import convop as cp
from convop.constraints import Constraint
from convop.model import Model

from convop.model_builder import load_parameters
from convop.objective import Objective
from convop.variables import Variable


def main():
    m = Model()

    input_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "input_data"

    food_cost = load_parameters(input_dir / "foods.csv", "cost")
    min_nutrients, max_nutrients = load_parameters(
        input_dir / "nutrients.csv", ["min_amount", "max_amount"]
    )
    nutrients_per_food = load_parameters(
        input_dir / "foods_to_nutrients.csv", ["amount"]
    )

    m.BuyFood = Variable(food_cost, lb=0)

    m.MinNutrients = Constraint(
        min_nutrients <= cp.sum("food", m.BuyFood * nutrients_per_food)
    )
    m.MaxNutrients = Constraint(
        cp.sum("food", m.BuyFood * nutrients_per_food) <= max_nutrients
    )

    m.Cost = Objective(cp.sum("food", food_cost * m.BuyFood), sense="min")

    m.to_file("diet.lp")


if __name__ == "__main__":
    main()
