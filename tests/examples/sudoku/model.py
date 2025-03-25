# Based on example at page 112 in book:
#       N. Sudermann-Merx: Einführung in Optimierungsmodelle, Springer Nature, 2023
from pathlib import Path

import polars as pl

import pyoframe as pf
from pyoframe import sum_by


def solve_model(use_var_names=True):
    init_values = pf.Set(
        pl.read_csv(Path(__file__).parent / "input_data" / "initial_numbers.csv")
    )

    digits = pl.int_range(1, 10, eager=True)

    cells = pl.DataFrame({"row": digits}).join(
        pl.DataFrame({"column": digits}), how="cross"
    )
    cell_to_box = cells.with_columns(
        box=((pl.col("row") - 1) // 3 + 1) * 10 + ((pl.col("column") - 1) // 3 + 1)
    )
    cell_options = cells.join(pl.DataFrame({"digit": digits}), how="cross")

    m = pf.Model(use_var_names=use_var_names)
    m.Y = pf.Variable(cell_options, vtype=pf.VType.BINARY)

    m.given_values = m.Y.drop_unmatched() == init_values

    m.one_per_row = sum_by(["digit", "row"], m.Y) == 1
    m.one_per_column = sum_by(["digit", "column"], m.Y) == 1
    m.one_per_box = sum_by(["digit", "box"], m.Y.map(cell_to_box)) == 1
    m.one_per_cell = sum_by(["row", "column"], m.Y) == 1

    if m.solver_name == "gurobi":
        m.params.Method = 2

    m.optimize()

    return m


if __name__ == "__main__":
    solve_model()
