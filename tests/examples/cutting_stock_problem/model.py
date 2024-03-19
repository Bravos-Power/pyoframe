# pyright: reportAttributeAccessIssue=false
import os
from pathlib import Path
import pandas as pd

from pyoframe import Model, Variable, sum_by, sum


def main(input_dir, output_dir):
    orders = pd.read_csv(input_dir / "orders.csv")
    orders.index.name = "order"

    params = pd.read_csv(input_dir / "parameters.csv").set_index("param")["value"]
    stock_width = params.loc["stock_width"]
    stock_available = params.loc["stock_available"]

    m = Model("cutting_stock")
    m.orders_in_stock = Variable(
        {"stock": range(stock_available)}, orders.index, vtype="integer", lb=0
    )
    m.is_used = Variable({"stock": range(stock_available)}, vtype="binary")

    m.con_within_stock = (
        sum_by("stock", m.orders_in_stock * orders["width"]) <= stock_width * m.is_used
    )
    m.con_meet_orders = sum_by("order", m.orders_in_stock) >= orders["quantity"]

    m.minimize = sum(m.is_used)

    gurobi_model = m.solve("gurobi", output_dir)
    assert gurobi_model.getObjective().getValue() == 73


if __name__ == "__main__":
    working_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    main(working_dir / "input_data", working_dir / "results")
