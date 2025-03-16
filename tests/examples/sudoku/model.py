# Based on example at page 112 in book:
#       N. Sudermann-Merx: Einführung in Optimierungsmodelle, Springer Nature, 2023
from pathlib import Path

import polars as pl

import pyoframe as pf
from pyoframe import sum


def solve_model(use_var_names=True):
    init_values = pl.read_csv(
        Path(__file__).parent / "input_data" / "initial_numbers.csv"
    )
    init_values = init_values.with_columns(init_values=1)

    digits = pl.int_range(1, 10, eager=True)

    cube9x9x9 = (
        pl.DataFrame({"row": digits})
        .join(pl.DataFrame({"column": digits}), how="cross")
        .join(pl.DataFrame({"digit": digits}), how="cross")
    )
    cube9x9x9 = cube9x9x9.with_columns(
        box=((pl.col("row") - 1) // 3 + 1) * 10 + ((pl.col("column") - 1) // 3 + 1)
    )

    m = pf.Model("sudoku_binary", use_var_names=use_var_names)
    m.Y = pf.Variable(cube9x9x9, vtype=pf.VType.BINARY)

    m.given_values = m.Y.drop_unmatched() == init_values.to_expr().add_dim("box")

    m.just_one_digit_is_set_to_rxc = sum(["digit", "box"], m.Y) == 1
    m.each_row_all_digits = sum(["column", "box"], m.Y) == 1
    m.each_column_all_digits = sum(["row", "box"], m.Y) == 1
    m.each_box_has_all_digits = sum(["row", "column"], m.Y) == 1

    if m.solver_name == "gurobi":
        m.params.Method = 2

    m.optimize()

    return m


if __name__ == "__main__":
    main()
