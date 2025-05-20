# Based on example at page 112 in book:
#       N. Sudermann-Merx: Einführung in Optimierungsmodelle, Springer Nature, 2023
from pathlib import Path

import polars as pl

import pyoframe as pf
from pyoframe import sum_by


def solve_model(use_var_names=True):
    init_values = pl.read_csv(
        Path(__file__).parent / "input_data" / "initial_numbers.csv"
    )

    one_to_nine = pl.DataFrame({"digit": range(1, 10)})
    grid = one_to_nine.join(one_to_nine, how="cross").rename(
        {"digit": "row", "digit_right": "column"}
    )

    m = pf.Model(use_var_names=use_var_names)
    m.Y = pf.Variable(grid.join(one_to_nine, how="cross"), vtype=pf.VType.BINARY)

    m.given_values = m.Y.drop_unmatched() == pf.Set(init_values)

    m.one_per_row = sum_by(["digit", "row"], m.Y) == 1
    m.one_per_column = sum_by(["digit", "column"], m.Y) == 1
    m.one_per_cell = sum_by(["row", "column"], m.Y) == 1

    cell_to_box = grid.with_columns(
        box=((pl.col("row") - 1) // 3) * 3 + (pl.col("column") - 1) // 3
    )
    m.one_per_box = m.Y.map(cell_to_box) == 1

    m.optimize()

    return m


def write_solution(m, output_dir: Path):
    sol: pl.DataFrame = m.Y.solution
    sol = sol.filter(pl.col("solution") == 1).drop("solution")
    sol = sol.pivot(on="column", values="digit", sort_columns=True)
    sol = sol.sort(by="row")
    sol.write_csv(output_dir / "solution.csv")


if __name__ == "__main__":
    write_solution(solve_model(), Path(__file__).parent / "results")
