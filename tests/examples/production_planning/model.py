"""Pyoframe formulation of a production planning model.

Based on example at page 20 in book:
      H.A Eiselt and Carl-Louis Sandblom: Operations Research - A Model-Based Approach 3rd Edition, Springer, 2022
"""

from pathlib import Path

import polars as pl

import pyoframe as pf

_input_dir = Path(__file__).parent / "input_data"


def solve_model(use_var_names=True):
    processing_times = pl.read_csv(_input_dir / "processing_times.csv")
    machines_availability = pl.read_csv(_input_dir / "machines_availability.csv")
    products_profit = pl.read_csv(_input_dir / "products_profit.csv")

    m = pf.Model(solver_uses_variable_names=use_var_names)
    m.Production = pf.Variable(products_profit["products"], lb=0)

    machine_usage = m.Production * processing_times
    m.machine_capacity = machine_usage.sum_by("machines") <= machines_availability

    m.maximize = (products_profit * m.Production).sum()

    m.optimize()

    return m


def write_solution(m: pf.Model, output_dir: Path):
    m.Production.solution.write_csv(output_dir / "solution.csv")


if __name__ == "__main__":
    write_solution(solve_model(), Path(__file__).parent / "results")
