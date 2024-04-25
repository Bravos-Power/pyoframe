# pyright: reportAttributeAccessIssue=false
import os
from typing import Union
import pandas as pd
from pathlib import Path

from pyoframe import Model, Variable, sum


def main(input_dir, **kwargs):
    plants = pd.read_csv(input_dir / "plants.csv").set_index("plant")
    warehouses = pd.read_csv(input_dir / "wharehouses.csv").set_index("wharehouse")
    transport_costs = (
        pd.read_csv(input_dir / "transport_costs.csv")
        .melt(id_vars="wharehouse", var_name="plant", value_name="cost")
        .astype({"plant": "int64"})
        .set_index(["wharehouse", "plant"])["cost"]
    )

    m = Model()
    m.open = Variable(plants.index, vtype="binary")
    m.transport = Variable(warehouses.index, plants.index, lb=0)

    m.con_max_capacity = sum("wharehouse", m.transport) <= plants.capacity * m.open
    m.con_meet_demand = sum("plant", m.transport) == warehouses.demand

    m.minimize = sum(m.open * plants.fixed_cost) + sum(m.transport * transport_costs)

    return m.solve("gurobi", **kwargs)


if __name__ == "__main__":
    working_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    main(working_dir / "input_data", directory=working_dir / "results")
