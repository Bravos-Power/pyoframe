# pyright: reportAttributeAccessIssue=false
import os
from pathlib import Path

import pandas as pd

from pyoframe import Model, Variable, sum, sum_by


def main(input_dir, directory, **kwargs):
    orders = pd.read_csv(input_dir / "orders.csv")
    orders.index.name = "order"

    params = pd.read_csv(input_dir / "parameters.csv").set_index("param")["value"]
    stock_width = params.loc["stock_width"]
    stock_available = params.loc["stock_available"]

    m = Model("min")
    m.orders_in_stock = Variable(
        {"stock": range(stock_available)}, orders.index, vtype="integer", lb=0
    )
    m.is_used = Variable({"stock": range(stock_available)}, vtype="binary")

    m.con_within_stock = (
        sum_by("stock", m.orders_in_stock * orders["width"]) <= stock_width * m.is_used
    )
    m.con_meet_orders = sum_by("order", m.orders_in_stock) >= orders["quantity"]

    m.objective = sum(m.is_used)

    result = m.solve("gurobi", directory=directory, **kwargs)
    assert result.status.is_ok
    assert result.solution.objective == 73  # type: ignore

    # Write results to CSV files
    m.orders_in_stock.solution.write_csv(directory / "orders_in_stock.csv")  # type: ignore
    m.is_used.solution.write_csv(directory / "is_used.csv")  # type: ignore

    return m


if __name__ == "__main__":
    working_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    main(
        working_dir / "input_data",
        directory=working_dir / "results",
        use_var_names=True,
    )
