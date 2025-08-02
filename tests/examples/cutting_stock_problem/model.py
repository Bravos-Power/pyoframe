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

    m = pf.Model(use_var_names=use_var_names)
    m.orders_in_stock = pf.Variable(
        {"stock": range(stock_available)}, orders.index, vtype="integer", lb=0
    )
    m.is_used = pf.Variable({"stock": range(stock_available)}, vtype="binary")

    m.con_within_stock = (
        pf.sum_by("stock", m.orders_in_stock * orders["width"])
        <= stock_width * m.is_used
    )
    m.con_meet_orders = pf.sum_by("order", m.orders_in_stock) >= orders["quantity"]

    m.minimize = pf.sum(m.is_used)

    m.optimize()

    return m
