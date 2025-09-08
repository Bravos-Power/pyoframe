"""Example Pyoframe formulation of the classic cutting stock problem."""

import os
from pathlib import Path

import pandas as pd

import pyoframe as pf

_input_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "input_data"


def solve_model(use_var_names):
    orders = pd.read_csv(_input_dir / "orders.csv")
    orders.index.name = "order"

    params = pd.read_csv(_input_dir / "parameters.csv").set_index("param")["value"]
    stock_width = params.loc["stock_width"]
    stock_available = params.loc["stock_available"]

    m = pf.Model(solver_uses_variable_names=use_var_names)
    m.orders_in_stock = pf.Variable(
        {"stock": range(stock_available)}, orders.index, vtype="integer", lb=0
    )
    m.is_used = pf.Variable({"stock": range(stock_available)}, vtype="binary")

    m.con_within_stock = (m.orders_in_stock * orders["width"]).sum_by(
        "stock"
    ) <= stock_width * m.is_used
    m.con_meet_orders = m.orders_in_stock.sum_by("order") >= orders["quantity"]

    m.minimize = m.is_used.sum()

    m.optimize()

    return m
