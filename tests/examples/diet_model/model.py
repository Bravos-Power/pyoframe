# pylint: disable=attribute-defined-outside-init,invalid-name,missing-function-docstring

import os
from pathlib import Path

import convop as cp

from convop.model_builder import load_parameters
from convop.parameters import Parameters



class DietModel(cp.ModelBuilder):
    """
    A simple diet model example.
    """
    def load_data(self):
        # Get directory of this file
        input_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "input_data"

        self.food_cost: Parameters = load_parameters(input_dir / "foods.csv", "cost")
        self.min_nutrients, self.max_nutrients = load_parameters(
            input_dir / "nutrients.csv", ["min_amount", "max_amount"]
        )
        self.nutrients_per_food = load_parameters(
            input_dir / "foods_to_nutrients.csv", ["amount"]
        )

    def create_variables(self):
        self.BuyFood = self.m.add_variables(self.food_cost, lb=0)

    def create_constraints(self):
        self.MinNutrients = self.m.add_constraints(
            lhs=self.min_nutrients,
            sense="<=",
            rhs=cp.sum("food", self.BuyFood * self.nutrients_per_food),
        )
        self.MaxNutrients = self.m.add_constraints(
            lhs=cp.sum("food", self.BuyFood * self.nutrients_per_food),
            sense="<=",
            rhs=self.max_nutrients,
        )

    def create_objective(self):
        self.Cost = self.m.set_objective(
            self.food_cost * self.BuyFood, direction="minimize"
        )

    def solve(self):
        self.m.solve()


def run_example():
    model = DietModel()
    model.load_data()
    model.create_variables()
    model.create_constraints()
    model.create_objective()
    model.solve()


if __name__ == "__main__":
    run_example()
