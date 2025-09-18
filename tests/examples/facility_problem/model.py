"""Pyoframe formulation of a facility problem."""

import os
from pathlib import Path

import pandas as pd

import pyoframe as pf

_input_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "input_data"


def solve_model(use_var_names):
    plants = pd.read_csv(_input_dir / "plants.csv").set_index("plant")
    warehouses = pd.read_csv(_input_dir / "wharehouses.csv").set_index("wharehouse")
    transport_costs = (
        pd.read_csv(_input_dir / "transport_costs.csv")
        .melt(id_vars="wharehouse", var_name="plant", value_name="cost")
        .astype({"plant": "int64"})
        .set_index(["wharehouse", "plant"])["cost"]
    )

    m = pf.Model(solver_uses_variable_names=use_var_names)
    m.open = pf.Variable(plants.index, vtype="binary")
    m.transport = pf.Variable(warehouses.index, plants.index, lb=0)

    m.con_max_capacity = m.transport.sum("wharehouse") <= plants.capacity * m.open
    m.con_meet_demand = m.transport.sum("plant") == warehouses.demand

    m.minimize = (m.open * plants.fixed_cost).sum() + (
        m.transport * transport_costs
    ).sum()

    if m.solver.name == "gurobi":
        m.params.Method = 2

    m.optimize()

    return m


if __name__ == "__main__":
    solve_model(use_var_names=True)
