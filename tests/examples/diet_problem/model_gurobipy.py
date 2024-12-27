"""
Adapted from https://www.gurobi.com/documentation/current/examples/diet_py.html. Copyright belongs there.
"""

import os
from pathlib import Path

import pandas as pd
from gurobipy import GRB, Model, quicksum


def main(input_dir, output_dir):
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
    m = Model()

    # Create decision variables for the foods to buy
    buy = m.addVars(foods, name="Buy", ub=df["stock"])

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
    m.write(str(output_dir / "gurobipy.lp"))
    m.optimize()
    m.write(str(output_dir / "gurobipy.sol"))
    return m.getObjective().getValue()


if __name__ == "__main__":
    working_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    main(working_dir / "input_data", working_dir / "results")
