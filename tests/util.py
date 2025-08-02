"""Utility functions for testing in Pyoframe."""

from __future__ import annotations

import ast
import inspect
import io
import textwrap
from typing import TYPE_CHECKING, Any, Literal, overload

import polars as pl

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
        return csvs_to_dataframe(*csv_strings).to_expr()
    return tuple(df.to_expr() for df in csvs_to_dataframe(*csv_strings))


_tolerances = {
    "gurobi": {"rtol": 1e-5, "atol": 1e-8},
    "highs": {"rtol": 1e-5, "atol": 1e-8},
    "ipopt": {"rtol": 1e-5, "atol": 1e-5},
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


def get_attr_docs(cls: type[Any]) -> dict[str, str]:
    """Copyright (c) David Lord under the MIT License (see: https://davidism.com/attribute-docstrings/).

    Get any docstrings placed after attribute assignments in a class body.
    """
    cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]

    if not isinstance(cls_node, ast.ClassDef):
        raise TypeError("Given object was not a class.")
    out = {}

    # Consider each pair of nodes.
    nodes = cls_node.body
    b = nodes[0]
    for i in range(1, len(nodes)):
        a, b = b, nodes[i]

        # Must be an assignment then a constant string.
        if (
            not isinstance(a, (ast.Assign, ast.AnnAssign))
            or not isinstance(b, ast.Expr)
            or not isinstance(b.value, ast.Constant)
            or not isinstance(b.value.value, str)
        ):
            continue

        doc = inspect.cleandoc(b.value.value)

        if isinstance(a, ast.Assign):
            # An assignment can have multiple targets (a = b = v).
            targets = a.targets
        else:
            # An annotated assignment only has one target.
            targets = [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, ast.Name):
                continue

            out[target.id] = doc

    return out
