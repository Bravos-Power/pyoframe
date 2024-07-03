import pyoframe as pf
import polars as pl
from polars.testing import assert_frame_equal
import pytest


def test_retrieving_duals():
    m = pf.Model("max")

    m.A = pf.Variable(ub=100)
    m.B = pf.Variable(ub=10)
    m.max_AB = 2 * m.A + m.B <= 100
    m.extra_slack_constraint = 2 * m.A + m.B <= 150
    m.objective = 0.2 * m.A + 2 * m.B

    m.solve("gurobi")

    assert m.A.solution == 45
    assert m.B.solution == 10
    assert m.objective.value == 29
    assert m.max_AB.dual == 0.1
    assert m.max_AB.slack == 0
    assert m.extra_slack_constraint.dual == 0
    assert m.extra_slack_constraint.slack == 50
    assert m.A.RC == 0
    assert m.B.RC == 1.9


def test_retrieving_duals_vectorized():
    m = pf.Model("max")
    data = pl.DataFrame(
        {"t": [1, 2], "ub": [100, 10], "coef": [2, 1], "obj_coef": [0.2, 2]}
    )
    m.X = pf.Variable(data["t"])
    m.X_ub = m.X <= data[["t", "ub"]]

    constraint_bounds = pl.DataFrame({"c": [1, 2], "bound": [100, 150]})
    m.max_AB = pf.sum(data[["t", "coef"]] * m.X).add_dim("c") <= constraint_bounds
    m.objective = pf.sum(data[["t", "obj_coef"]] * m.X)

    m.solve("gurobi")

    assert m.objective.value == 29
    assert_frame_equal(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45, 10]}),
        check_dtype=False,
        check_row_order=False,
    )
    assert_frame_equal(
        m.X.RC,
        pl.DataFrame(
            {"t": [1, 2], "RC": [0, 0]}
        ),  # Somehow the reduced cost is 0 since we are no longer using a bound.
        check_dtype=False,
        check_row_order=False,
    )
    assert_frame_equal(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        check_dtype=False,
        check_row_order=False,
    )
    assert_frame_equal(
        m.max_AB.slack,
        pl.DataFrame({"c": [1, 2], "slack": [0, 50]}),
        check_dtype=False,
        check_row_order=False,
    )


def test_support_variable_attributes():
    m = pf.Model("max")
    data = pl.DataFrame(
        {"t": [1, 2], "ub": [100, 10], "coef": [2, 1], "obj_coef": [0.2, 2]}
    )
    m.X = pf.Variable(data["t"])
    m.X.attr.UB = data[["t", "ub"]]

    constraint_bounds = pl.DataFrame({"c": [1, 2], "bound": [100, 150]})
    m.max_AB = pf.sum(data[["t", "coef"]] * m.X).add_dim("c") <= constraint_bounds
    m.objective = pf.sum(data[["t", "obj_coef"]] * m.X)

    m.solve("gurobi")

    assert m.objective.value == 29
    assert_frame_equal(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45, 10]}),
        check_dtype=False,
        check_row_order=False,
    )
    assert_frame_equal(
        m.X.RC,
        pl.DataFrame({"t": [1, 2], "RC": [0.0, 1.9]}),
        check_dtype=False,
        check_row_order=False,
    )
    assert_frame_equal(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        check_dtype=False,
        check_row_order=False,
    )
    assert_frame_equal(
        m.max_AB.slack,
        pl.DataFrame({"c": [1, 2], "slack": [0, 50]}),
        check_dtype=False,
        check_row_order=False,
    )


def test_setting_constraint_attr():
    # Build an unbounded model
    m = pf.Model("max")
    m.A = pf.Variable()
    m.B = pf.Variable(pf.Set(y=[1, 2, 3]))
    m.A_con = m.A >= 10
    m.B_con = m.B >= 5
    m.objective = m.A + pf.sum(m.B)

    # Solving it should return unbounded
    result = m.solve()
    assert not result.status.is_ok

    # Now we make the model bounded by setting the Sense attribute
    m.A_con.attr.Sense = "<"
    m.B_con.attr.Sense = pl.DataFrame({"y": [1, 2, 3], "Sense": ["<", "<", "="]})

    # Now the model should be bounded
    result = m.solve()
    assert result.status.is_ok


def test_setting_model_attr():
    # Build an unbounded model
    m = pf.Model("max")
    m.A = pf.Variable(lb=0)
    m.objective = m.A

    # Solving it should return unbounded
    result = m.solve()
    assert not result.status.is_ok

    # Now we make the model a minimization problem
    m.attr.ModelSense = 1

    # Now the model should be bounded
    result = m.solve()
    assert result.status.is_ok


@pytest.mark.parametrize("use_var_names", [True, False])
def test_const_term_in_objective(use_var_names):
    m = pf.Model("max")
    m.A = pf.Variable(ub=10)
    m.objective = 10 + m.A

    result = m.solve(use_var_names=use_var_names)
    assert result.status.is_ok
    assert m.A.solution == 10
    assert m.objective.value == 20
