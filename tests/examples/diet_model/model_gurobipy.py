# From https://www.gurobi.com/documentation/current/examples/diet_py.html

import os
from pathlib import Path
from gurobipy import GRB, Model, quicksum
import pandas as pd
import numpy as np


def main(working_dir):
    working_dir = Path(working_dir)
    input_dir = working_dir / "input_data"

    df = pd.read_csv(input_dir / "foods.csv").set_index("food")
    cost = df["cost"].to_dict()
    foods = df.index.tolist()
    nutritionValues = (
        pd.read_csv(input_dir / "foods_to_nutrients.csv")
        .set_index(["food", "category"])["amount"]
        .to_dict()
    )
    nutrients = pd.read_csv(input_dir / "nutrients.csv").set_index("category")
    minNutrition = nutrients["min"].dropna().to_dict()
    maxNutrition = nutrients["max"].dropna().to_dict()
    categories = nutrients.index.tolist()

    # Model
    m = Model("diet")

    # Create decision variables for the foods to buy
    buy = m.addVars(foods, name="Buy")

    # Nutrition constraints
    m.addConstrs(
        (
            quicksum(nutritionValues[f, c] * buy[f] for f in foods) >= minNutrition[c]
            for c in categories
            if c in minNutrition
        ),
        "minNutrition",
    )

    m.addConstrs(
        (
            quicksum(nutritionValues[f, c] * buy[f] for f in foods) <= maxNutrition[c]
            for c in categories
            if c in maxNutrition
        ),
        "maxNutrition",
    )

    m.setObjective(buy.prod(cost), GRB.MINIMIZE)

    # Solve
    m.write(str(working_dir / "results" / "diet-gurobipy.lp"))
    m.optimize()
    m.write(str(working_dir / "results" / "diet-gurobipy.sol"))


if __name__ == "__main__":
    main(os.path.dirname(os.path.realpath(__file__)))
