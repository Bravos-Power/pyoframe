import pyoframe as pf
import polars as pl
from polars.testing import assert_frame_equal


def test_retrieving_duals():
    m = pf.Model()

    m.A = pf.Variable(ub=100)
    m.B = pf.Variable(ub=10)
    m.max_AB = 2 * m.A + m.B <= 100
    m.extra_slack_constraint = 2 * m.A + m.B <= 150
    m.maximize = 0.2 * m.A + 2 * m.B

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
    m = pf.Model()
    data = pl.DataFrame(
        {"t": [1, 2], "ub": [100, 10], "coef": [2, 1], "obj_coef": [0.2, 2]}
    )
    m.X = pf.Variable(data["t"])
    m.X_ub = m.X <= data[["t", "ub"]]

    constraint_bounds = pl.DataFrame({"c": [1, 2], "bound": [100, 150]})
    m.max_AB = pf.sum(data[["t", "coef"]] * m.X).add_dim("c") <= constraint_bounds
    m.maximize = pf.sum(data[["t", "obj_coef"]] * m.X)

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
    m = pf.Model()
    data = pl.DataFrame(
        {"t": [1, 2], "ub": [100, 10], "coef": [2, 1], "obj_coef": [0.2, 2]}
    )
    m.X = pf.Variable(data["t"])
    m.X.attr.UB = data[["t", "ub"]]

    constraint_bounds = pl.DataFrame({"c": [1, 2], "bound": [100, 150]})
    m.max_AB = pf.sum(data[["t", "coef"]] * m.X).add_dim("c") <= constraint_bounds
    m.maximize = pf.sum(data[["t", "obj_coef"]] * m.X)

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
            {"t": [1, 2], "RC": [0, 1.9]}
        ),
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
