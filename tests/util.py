"""Utility functions for testing in Pyoframe."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Literal, overload

import polars as pl

from pyoframe import Param

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe._core import Expression


@overload
def csvs_to_dataframe(
    csv_strings: str,
) -> pl.DataFrame: ...


@overload
def csvs_to_dataframe(
    *csv_strings: str,
) -> tuple[pl.DataFrame, ...]: ...


def csvs_to_dataframe(
    *csv_strings: str,
) -> tuple[pl.DataFrame, ...] | pl.DataFrame:
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
) -> Expression: ...


@overload
def csvs_to_expr(
    *csv_strings: str,
) -> tuple[Expression, ...]: ...


def csvs_to_expr(
    *csv_strings: str,
) -> tuple[Expression, ...] | Expression:
    if len(csv_strings) == 1:
        return Param(csvs_to_dataframe(*csv_strings))
    return tuple(Param(df) for df in csvs_to_dataframe(*csv_strings))


_tolerances = {
    "gurobi": {"rel_tol": 1e-5, "abs_tol": 1e-8},
    "highs": {"rel_tol": 1e-5, "abs_tol": 1e-8},
    "ipopt": {"rel_tol": 1e-5, "abs_tol": 1e-5},
    "copt": {"rel_tol": 1e-5, "abs_tol": 1e-8},
}


def get_tol_pl(solver) -> dict[Literal["abs_tol"] | Literal["rel_tol"], float]:
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
        rel=tol["rel_tol"],
        abs=tol["abs_tol"],
    )
