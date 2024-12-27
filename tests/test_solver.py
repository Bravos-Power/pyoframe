import polars as pl
import pyoptinterface as poi
import pytest
from polars.testing import assert_frame_equal

import pyoframe as pf
from pyoframe.constants import POLARS_VERSION

check_dtypes_false = (
    {"check_dtypes": False} if POLARS_VERSION.major >= 1 else {"check_dtype": False}
)


def test_retrieving_duals(solver):
    m = pf.Model()

    m.A = pf.Variable(ub=100)
    m.B = pf.Variable(ub=10)
    m.max_AB = 2 * m.A + m.B <= 100
    m.extra_slack_constraint = 2 * m.A + m.B <= 150
    m.maximize = 0.2 * m.A + 2 * m.B

    m.optimize()

    assert m.A.solution == 45
    assert m.B.solution == 10
    assert m.maximize.value == 29
    assert m.max_AB.dual == 0.1
    assert m.extra_slack_constraint.dual == 0
    if solver == "gurobi":
        assert m.max_AB.attr.slack == 0
        assert m.extra_slack_constraint.attr.slack == 50
    if solver == "gurobi":
        assert m.A.attr.RC == 0
        assert m.B.attr.RC == 1.9


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

    assert m.maximize.value == 29
    assert_frame_equal(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45, 10]}),
        check_row_order=False,
        **check_dtypes_false,
    )
    assert_frame_equal(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        check_row_order=False,
        **check_dtypes_false,
    )
    if solver == "gurobi":
        assert_frame_equal(
            m.max_AB.attr.slack,
            pl.DataFrame({"c": [1, 2], "slack": [0, 50]}),
            check_row_order=False,
            **check_dtypes_false,
        )
        assert_frame_equal(
            m.X.attr.RC,
            pl.DataFrame(
                {"t": [1, 2], "RC": [0, 0]}
            ),  # Somehow the reduced cost is 0 since we are no longer using a bound.
            check_row_order=False,
            **check_dtypes_false,
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

    assert m.maximize.value == 29
    assert_frame_equal(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45, 10]}),
        check_row_order=False,
        **check_dtypes_false,
    )
    if solver == "gurobi":
        assert_frame_equal(
            m.X.attr.RC,
            pl.DataFrame({"t": [1, 2], "RC": [0.0, 1.9]}),
            check_row_order=False,
            **check_dtypes_false,
        )
        assert_frame_equal(
            m.max_AB.attr.slack,
            pl.DataFrame({"c": [1, 2], "slack": [0, 50]}),
            check_row_order=False,
            **check_dtypes_false,
        )
    assert_frame_equal(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        **check_dtypes_false,
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

    assert m.maximize.value == 29
    assert_frame_equal(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45, 10]}),
        check_row_order=False,
        **check_dtypes_false,
    )
    if solver == "gurobi":
        assert_frame_equal(
            m.X.attr.RC,
            pl.DataFrame({"t": [1, 2], "RC": [0.0, 1.9]}),
            check_row_order=False,
            **check_dtypes_false,
        )
    assert_frame_equal(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        **check_dtypes_false,
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


@pytest.mark.parametrize("use_var_names", [True, False])
def test_const_term_in_objective(use_var_names):
    m = pf.Model(use_var_names=use_var_names)
    m.A = pf.Variable(ub=10)
    m.maximize = 10 + m.A

    m.optimize()
    assert m.A.solution == 10
    assert m.maximize.value == 20
