from __future__ import annotations

import io
from typing import TYPE_CHECKING, Literal, Tuple, Union, overload

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.core import Expression


@overload
def csvs_to_dataframe(
    csv_strings: str,
) -> pl.DataFrame: ...


@overload
def csvs_to_dataframe(
    *csv_strings: str,
) -> Tuple[pl.DataFrame, ...]: ...


def csvs_to_dataframe(
    *csv_strings: str,
) -> Union[Tuple[pl.DataFrame, ...], pl.DataFrame]:
    """Convert a sequence of CSV strings to Pyoframe expressions."""
    dfs = []
    for csv_string in csv_strings:
        csv_string = "\n".join(line.strip() for line in csv_string.splitlines())
        dfs.append(pl.read_csv(io.StringIO(csv_string)))
    if len(dfs) == 1:
        return dfs[0]
    return tuple(dfs)


@overload
def csvs_to_expr(
    csv_strings: str,
) -> "Expression": ...


@overload
def csvs_to_expr(
    *csv_strings: str,
) -> Tuple["Expression", ...]: ...


def csvs_to_expr(
    *csv_strings: str,
) -> Union[Tuple["Expression", ...], "Expression"]:
    if len(csv_strings) == 1:
        return csvs_to_dataframe(*csv_strings).to_expr()
    return tuple((df.to_expr() for df in csvs_to_dataframe(*csv_strings)))


_tolerances = {
    "gurobi": {"rtol": 1e-5, "atol": 1e-8},
    "highs": {"rtol": 1e-5, "atol": 1e-8},
    "ipopt": {"rtol": None, "atol": 1e-5},
}


def get_tol_pl(solver) -> dict[Literal["atol"] | Literal["rtol"], float]:
    """Return tolerances for Polars' assert_frame_equal()."""
    if not isinstance(solver, str):
        solver = solver.name
    return _tolerances[solver]


def get_tol(solver):
    """Return tolerances for pytest's approx()."""
    if not isinstance(solver, str):
        solver = solver.name
    tol = _tolerances[solver]
    return dict(
        rel=tol["rtol"],
        abs=tol["atol"],
    )
