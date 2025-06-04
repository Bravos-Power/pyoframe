from __future__ import annotations

import io
import pytest
from typing import TYPE_CHECKING, Tuple, Union, overload

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


def assert_with_solver_tolerance(
    actual: int | float | pl.DataFrame,
    expected: int | float | pl.DataFrame,
    solver,
    abs_tol=1e-6,
    rel_tol=1e-8,
    check_dtypes=False,
    check_row_order=False,
):
    """
    Assert equality with appropriate tolerance based on solver.

    Parameters:
        actual: The actual value or DataFrame
        expected: The expected value or DataFrame
        solver: The solver name (string)
        abs_tol: Absolute tolerance for numerical comparisons when using approximate solvers
        rel_tol: Relative tolerance for numerical comparisons when using approximate solvers
        check_dtypes: Whether to check DataFrame data types (for DataFrame comparisons)
        check_row_order: Whether to check DataFrame row order (for DataFrame comparisons)
    """
    # Determine if solver needs tolerance
    use_tolerance = solver.name == "ipopt"

    # Handle different types of actual/expected values
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        # Simple numeric comparison
        if use_tolerance:
            assert actual == pytest.approx(expected, abs=abs_tol, rel=rel_tol), (
                f"Expected {expected}, got {actual} (with tolerance {abs_tol})"
            )
        else:
            assert actual == expected, f"Expected {expected}, got {actual}"

    elif isinstance(actual, pl.DataFrame) and isinstance(expected, pl.DataFrame):
        # Polars DataFrame comparison
        if actual.shape != expected.shape:
            assert False, (
                f"DataFrames have different shapes: {actual.shape} vs {expected.shape}"
            )

        # Check columns
        assert set(actual.columns) == set(expected.columns), (
            f"DataFrames have different columns: {actual.columns} vs {expected.columns}"
        )

        # If not checking row order, sort both dataframes by all columns for consistent comparison
        if not check_row_order:
            # Find a common column to sort by (prefer ID-like columns)
            sort_cols = actual.columns
            actual = actual.sort(sort_cols)
            expected = expected.sort(sort_cols)

        # Compare all values with appropriate tolerance
        for col in expected.columns:
            actual_col = actual.select(col).to_series()
            expected_col = expected.select(col).to_series()

            # For numeric columns with tolerance
            if actual_col.dtype in [pl.Float32, pl.Float64] and use_tolerance:
                for i, (a, e) in enumerate(zip(actual_col, expected_col)):
                    assert a == pytest.approx(e, abs=abs_tol, rel=rel_tol), (
                        f"Row {i}, column '{col}': Expected {e}, got {a} (with tolerance {abs_tol})"
                    )
            else:
                # For non-numeric columns or when not using tolerance
                assert actual_col.equals(expected_col), (
                    f"Values differ in column '{col}': {actual_col} vs {expected_col}"
                )

        # Check dtypes if required
        if check_dtypes:
            for col in expected.columns:
                assert actual[col].dtype == expected[col].dtype, (
                    f"Different dtypes for column '{col}': {actual[col].dtype} vs {expected[col].dtype}"
                )
    else:
        raise TypeError(
            "Actual and expected must both be numeric types or both be Polars DataFrames."
        )
