# Based on example at page 112 in book:
#       N. Sudermann-Merx: Einführung in Optimierungsmodelle, Springer Nature, 2023

import os
from pathlib import Path

import polars as pl

import pyoframe as pf
from pyoframe import sum


def main(input_dir, directory, use_var_names=True, **kwargs):
    init_values = pl.read_csv(input_dir / "initial_numbers.csv")
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

    # model.just_one_digit_is_set_to_rxc = model.Y.sum(['digit', 'box']) == 1
    # model.each_row_all_digits = model.Y.sum(['column', 'box']) == 1
    # model.each_column_all_digits = model.Y.sum(['row', 'box']) == 1
    # model.each_box_has_all_digits = model.Y.sum(['row', 'column']) == 1
    m.just_one_digit_is_set_to_rxc = sum(["digit", "box"], m.Y) == 1
    m.each_row_all_digits = sum(["column", "box"], m.Y) == 1
    m.each_column_all_digits = sum(["row", "box"], m.Y) == 1
    m.each_box_has_all_digits = sum(["row", "column"], m.Y) == 1

    if m.solver_name == "gurobi":
        m.params.Method = 2
    m.write(directory / "pyoframe-problem.lp")
    m.optimize(**kwargs)
    if m.solver_name == "gurobi":
        m.write(directory / "pyoframe-problem.sol")

    # Write results to CSV files
    (
        m.Y.solution.filter(pl.col("solution") == 1)
        .select(["row", "column", "digit"])
        .write_csv(directory / "Y.csv")
    )

    return m


if __name__ == "__main__":
    working_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    main(working_dir / "input_data", directory=working_dir / "results")
