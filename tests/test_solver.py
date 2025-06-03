import pandas as pd
import polars as pl
import pyoptinterface as poi
import pytest

import pyoframe as pf


def assert_with_solver_tolerance(
    actual,
    expected,
    solver,
    abs_tol=1e-7,
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
    use_tolerance = "ipopt" in solver.lower()

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

    elif isinstance(actual, pd.DataFrame) and isinstance(expected, pd.DataFrame):
        # Pandas DataFrame comparison (similar to Polars version)
        # Implement similar logic for pandas DataFrames if needed
        raise NotImplementedError("Pandas DataFrame comparison not implemented yet")

    else:
        # Fallback for other types
        if use_tolerance and hasattr(pytest, "approx"):
            assert actual == pytest.approx(expected, abs=abs_tol), (
                f"Expected {expected}, got {actual} (with tolerance {abs_tol})"
            )
        else:
            assert actual == expected, f"Expected {expected}, got {actual}"


def test_retrieving_duals(solver):
    m = pf.Model()

    m.A = pf.Variable(ub=100)
    m.B = pf.Variable(ub=10)
    m.max_AB = 2 * m.A + m.B <= 100
    m.extra_slack_constraint = 2 * m.A + m.B <= 150
    m.maximize = 0.2 * m.A + 2 * m.B

    m.optimize()

    assert_with_solver_tolerance(m.A.solution, 45, solver)
    assert_with_solver_tolerance(m.B.solution, 10, solver)
    assert_with_solver_tolerance(m.maximize.value, 29, solver)
    assert_with_solver_tolerance(m.max_AB.dual, 0.1, solver)
    assert_with_solver_tolerance(m.extra_slack_constraint.dual, 0, solver)

    if solver == "gurobi":
        assert_with_solver_tolerance(m.max_AB.attr.slack, 0, solver)
        assert_with_solver_tolerance(m.extra_slack_constraint.attr.slack, 50, solver)
        assert_with_solver_tolerance(m.A.attr.RC, 0, solver)
        assert_with_solver_tolerance(m.B.attr.RC, 1.9, solver)


def test_retrieving_duals_vectorized(solver):
    m = pf.Model()
    data = pl.DataFrame(
        {"t": [1, 2], "ub": [100, 10], "coef": [2, 1], "obj_coef": [0.2, 2]}
    )
    m.X = pf.Variable(data["t"])
    m.X_ub = m.X <= data[["t", "ub"]]

    constraint_bounds = pl.DataFrame({"c": [1, 2], "bound": [100, 150]})
    m.max_AB = pf.sum(data[["t", "coef"]] * m.X).add_dim("c") <= constraint_bounds
    m.maximize = pf.sum(data[["t", "obj_coef"]] * m.X)

    m.optimize()

    assert_with_solver_tolerance(m.maximize.value, 29, solver)
    assert_with_solver_tolerance(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45, 10]}),
        solver,
        check_row_order=False,
        check_dtypes=False,
    )
    assert_with_solver_tolerance(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        solver,
        check_row_order=False,
        check_dtypes=False,
    )

    if solver == "gurobi":
        assert_with_solver_tolerance(
            m.max_AB.attr.slack,
            pl.DataFrame({"c": [1, 2], "slack": [0, 50]}),
            solver,
            check_row_order=False,
            check_dtypes=False,
        )
        assert_with_solver_tolerance(
            m.X.attr.RC,
            pl.DataFrame({"t": [1, 2], "RC": [0, 0]}),
            # Somehow the reduced cost is 0 since we are no longer using a bound.
            solver,
            check_row_order=False,
            check_dtypes=False,
        )


def test_support_variable_attributes(solver):
    m = pf.Model()
    data = pl.DataFrame(
        {"t": [1, 2], "UpperBound": [100, 10], "coef": [2, 1], "obj_coef": [0.2, 2]}
    )
    m.X = pf.Variable(data["t"])
    m.X.attr.UpperBound = data[["t", "UpperBound"]]

    constraint_bounds = pl.DataFrame({"c": [1, 2], "bound": [100, 150]})
    m.max_AB = pf.sum(data[["t", "coef"]] * m.X).add_dim("c") <= constraint_bounds
    m.maximize = pf.sum(data[["t", "obj_coef"]] * m.X)

    m.optimize()

    assert_with_solver_tolerance(m.maximize.value, 29, solver)
    assert_with_solver_tolerance(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45, 10]}),
        solver,
        check_row_order=False,
        check_dtypes=False,
    )

    if solver == "gurobi":
        assert_with_solver_tolerance(
            m.X.attr.RC,
            pl.DataFrame({"t": [1, 2], "RC": [0.0, 1.9]}),
            solver,
            check_row_order=False,
            check_dtypes=False,
        )
        assert_with_solver_tolerance(
            m.max_AB.attr.slack,
            pl.DataFrame({"c": [1, 2], "slack": [0, 50]}),
            solver,
            check_row_order=False,
            check_dtypes=False,
        )

    assert_with_solver_tolerance(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        solver,
        check_dtypes=False,
        check_row_order=False,
    )


def test_support_variable_raw_attributes(solver):
    if solver != "gurobi":
        pytest.skip("Only valid for gurobi")
    m = pf.Model()
    data = pl.DataFrame(
        {"t": [1, 2], "UB": [100, 10], "coef": [2, 1], "obj_coef": [0.2, 2]}
    )
    m.X = pf.Variable(data["t"])
    m.X.attr.UB = data[["t", "UB"]]

    constraint_bounds = pl.DataFrame({"c": [1, 2], "bound": [100, 150]})
    m.max_AB = pf.sum(data[["t", "coef"]] * m.X).add_dim("c") <= constraint_bounds
    m.maximize = pf.sum(data[["t", "obj_coef"]] * m.X)

    m.optimize()

    assert_with_solver_tolerance(m.maximize.value, 29, solver)
    assert_with_solver_tolerance(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45, 10]}),
        solver,
        check_row_order=False,
        check_dtypes=False,
    )

    if solver == "gurobi":
        assert_with_solver_tolerance(
            m.X.attr.RC,
            pl.DataFrame({"t": [1, 2], "RC": [0.0, 1.9]}),
            solver,
            check_row_order=False,
            check_dtypes=False,
        )

    assert_with_solver_tolerance(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        solver,
        check_dtypes=False,
        check_row_order=False,
    )


def test_setting_constraint_attr(solver):
    if solver != "gurobi":
        pytest.skip("Only valid for gurobi")
    # Build an unbounded model
    m = pf.Model()
    m.A = pf.Variable()
    m.B = pf.Variable(pf.Set(y=[1, 2, 3]))
    m.A_con = m.A >= 10
    m.B_con = m.B >= 5
    m.maximize = m.A + pf.sum(m.B)

    # Solving it should return unbounded
    m.optimize()
    assert m.attr.TerminationStatus != poi.TerminationStatusCode.OPTIMAL

    # Now we make the model bounded by setting the Sense attribute
    m.A_con.attr.Sense = "<"
    m.B_con.attr.Sense = pl.DataFrame({"y": [1, 2, 3], "Sense": ["<", "<", "="]})

    # Now the model should be bounded
    m.optimize()
    assert m.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL


def test_setting_model_attr(solver):
    if solver != "gurobi":
        pytest.skip("Only valid for gurobi")
    # Build an unbounded model
    m = pf.Model()
    m.A = pf.Variable(lb=0)
    m.maximize = m.A

    # Solving it should return unbounded
    m.optimize()
    assert m.attr.TerminationStatus != poi.TerminationStatusCode.OPTIMAL

    # Now we make the model a minimization problem
    m.attr.ModelSense = 1

    # Now the model should be bounded
    m.optimize()
    assert m.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL


def test_const_term_in_objective(use_var_names, solver):
    m = pf.Model(use_var_names=use_var_names)
    m.A = pf.Variable(ub=10)
    m.maximize = 10 + m.A

    m.optimize()
    assert_with_solver_tolerance(m.A.solution, 10, solver)
    assert_with_solver_tolerance(m.maximize.value, 20, solver)
