"""Tests related to interacting with solvers."""

import polars as pl
import pyoptinterface as poi
import pytest
from polars.testing import assert_frame_equal
from pytest import approx

import pyoframe as pf
from pyoframe._constants import SUPPORTED_SOLVERS, _Solver
from tests.util import get_tol, get_tol_pl


@pytest.mark.parametrize("solver_all", SUPPORTED_SOLVERS, ids=lambda s: s.name)
def test_solver_works(solver_all):
    """Ensures the test suite fails if not all solvers could be tested.

    Note that the function parameter cannot be named "solver" (otherwise it uses the fixture).
    """
    pf.Model(solver_all.name)


def test_Config_default_solver():
    pf.Config.default_solver = "raise"
    with pytest.raises(ValueError, match="No solver specified"):
        pf.Model()

    pf.Config.default_solver = "auto"
    assert pf.Model().solver_name in [s.name for s in SUPPORTED_SOLVERS]

    pf.Config.default_solver = None
    with pytest.raises(ValueError, match="Config.default_solver has an invalid value"):
        pf.Model()


def test_Config_default_solver_specific(solver):
    pf.Config.default_solver = solver.name
    assert pf.Model().solver_name == solver.name

    pf.Config.default_solver = solver
    assert pf.Model().solver_name == solver.name


@pytest.mark.parametrize(
    "sense", [pf.ObjSense.MIN, pf.ObjSense.MAX], ids=["min", "max"]
)
@pytest.mark.parametrize(
    "dimensioned", [True, False], ids=["dimensioned", "un-dimensioned"]
)
def test_retrieving_duals(solver, sense, dimensioned):
    m = pf.Model(solver, sense=sense)

    if dimensioned:
        dim = pf.Set(x=[1, 2])
        m.A = pf.Variable(dim, lb=-100, ub=100)
        m.B = pf.Variable(dim, lb=-10, ub=10)
    else:
        m.A = pf.Variable(lb=-100, ub=100)
        m.B = pf.Variable(lb=-10, ub=10)
    m.max_AB = 2 * m.A + m.B <= 100
    m.min_AB = 2 * m.A + m.B >= -100
    obj = 0.2 * m.A + 2 * m.B
    if dimensioned:
        obj = obj.sum()
    m.objective = obj

    m.optimize()

    flip = 1 if sense == pf.ObjSense.MAX else -1

    def unwrap(value):
        if dimensioned:
            return value.filter(x=1).drop("x").item()
        else:
            return value

    factor = len(dim) if dimensioned else 1

    assert unwrap(m.A.solution) == approx(45 * flip, **get_tol(solver))
    assert unwrap(m.B.solution) == approx(10 * flip, **get_tol(solver))
    assert m.objective.value == approx(29 * flip * factor, **get_tol(solver))
    tight, slack = (
        (m.max_AB, m.min_AB) if sense == pf.ObjSense.MAX else (m.min_AB, m.max_AB)
    )

    assert unwrap(tight.dual) == approx(0.1, **get_tol(solver))
    assert unwrap(slack.dual) == approx(0, **get_tol(solver))

    if solver.name == "gurobi":
        assert unwrap(tight.attr.slack) == approx(0, **get_tol(solver))
        assert unwrap(slack.attr.slack) == approx(-200 * flip, **get_tol(solver))
        assert unwrap(m.A.attr.RC) == approx(0, **get_tol(solver))
        assert unwrap(m.B.attr.RC) == approx(1.9, **get_tol(solver))


def test_retrieving_duals_vectorized(solver):
    m = pf.Model(solver)
    data = pl.DataFrame(
        {"t": [1, 2], "ub": [100, 10], "coef": [2, 1], "obj_coef": [0.2, 2]}
    )
    m.X = pf.Variable(data["t"])
    m.X_ub = m.X <= data[["t", "ub"]]

    constraint_bounds = pl.DataFrame({"c": [1, 2], "bound": [100, 150]})
    m.max_AB = (data[["t", "coef"]] * m.X).sum().over("c") <= constraint_bounds
    m.maximize = (data[["t", "obj_coef"]] * m.X).sum()

    m.optimize()

    assert m.maximize.value == approx(29, **get_tol(solver))
    assert_frame_equal(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45.0, 10.0]}),
        check_row_order=False,
        check_dtypes=False,
        **get_tol_pl(solver),
    )
    assert_frame_equal(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        check_row_order=False,
        check_dtypes=False,
        **get_tol_pl(solver),
    )

    if solver.name == "gurobi":
        assert_frame_equal(
            m.max_AB.attr.slack,
            pl.DataFrame({"c": [1, 2], "slack": [0, 50]}),
            check_row_order=False,
            check_dtypes=False,
            **get_tol_pl(solver),
        )
        assert_frame_equal(
            m.X.attr.RC,
            pl.DataFrame({"t": [1, 2], "RC": [0, 0]}),
            # Somehow the reduced cost is 0 since we are no longer using a bound.
            check_row_order=False,
            check_dtypes=False,
            **get_tol_pl(solver),
        )


def test_support_variable_attributes(solver):
    m = pf.Model(solver)
    data = pl.DataFrame(
        {"t": [1, 2], "UpperBound": [100, 10], "coef": [2, 1], "obj_coef": [0.2, 2]}
    )
    m.X = pf.Variable(data["t"])
    m.X.attr.UpperBound = data[["t", "UpperBound"]]

    constraint_bounds = pl.DataFrame({"c": [1, 2], "bound": [100, 150]})
    m.max_AB = (data[["t", "coef"]] * m.X).sum().over("c") <= constraint_bounds
    m.maximize = (data[["t", "obj_coef"]] * m.X).sum()

    m.optimize()

    assert m.maximize.value == approx(29, **get_tol(solver))
    assert_frame_equal(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45.0, 10.0]}),
        check_row_order=False,
        check_dtypes=False,
        **get_tol_pl(solver),
    )

    if solver.name == "gurobi":
        assert_frame_equal(
            m.X.attr.RC,
            pl.DataFrame({"t": [1, 2], "RC": [0.0, 1.9]}),
            check_row_order=False,
            check_dtypes=False,
            **get_tol_pl(solver),
        )
        assert_frame_equal(
            m.max_AB.attr.slack,
            pl.DataFrame({"c": [1, 2], "slack": [0, 50]}),
            check_row_order=False,
            check_dtypes=False,
            **get_tol_pl(solver),
        )

    assert_frame_equal(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        check_dtypes=False,
        check_row_order=False,
        **get_tol_pl(solver),
    )


def test_support_variable_raw_attributes():
    solver = "gurobi"
    m = pf.Model(solver)
    data = pl.DataFrame(
        {"t": [1, 2], "UB": [100, 10], "coef": [2, 1], "obj_coef": [0.2, 2]}
    )
    m.X = pf.Variable(data["t"])
    m.X.attr.UB = data[["t", "UB"]]

    constraint_bounds = pl.DataFrame({"c": [1, 2], "bound": [100, 150]})
    m.max_AB = (data[["t", "coef"]] * m.X).sum().over("c") <= constraint_bounds
    m.maximize = (data[["t", "obj_coef"]] * m.X).sum()

    m.optimize()

    assert m.maximize.value == approx(29, **get_tol(solver))
    assert_frame_equal(
        m.X.solution,
        pl.DataFrame({"t": [1, 2], "solution": [45, 10]}),
        check_row_order=False,
        check_dtypes=False,
        **get_tol_pl(solver),
    )

    assert_frame_equal(
        m.X.attr.RC,
        pl.DataFrame({"t": [1, 2], "RC": [0.0, 1.9]}),
        check_row_order=False,
        check_dtypes=False,
        **get_tol_pl(solver),
    )

    assert_frame_equal(
        m.max_AB.dual,
        pl.DataFrame({"c": [1, 2], "dual": [0.1, 0]}),
        check_dtypes=False,
        check_row_order=False,
        **get_tol_pl(solver),
    )


def test_setting_gurobi_constraint_attr():
    # Build an unbounded model
    m = pf.Model("gurobi")
    m.A = pf.Variable()
    m.B = pf.Variable(pf.Set(y=[1, 2, 3]))
    m.A_con = m.A >= 10
    m.B_con = m.B >= 5
    m.maximize = m.A + m.B.sum()

    # Solving it should return unbounded
    m.optimize()
    assert m.attr.TerminationStatus != poi.TerminationStatusCode.OPTIMAL

    # Now we make the model bounded by setting the Sense attribute
    m.A_con.attr.Sense = "<"
    m.B_con.attr.Sense = pl.DataFrame({"y": [1, 2, 3], "Sense": ["<", "<", "="]})

    # Now the model should be bounded
    m.optimize()
    assert m.attr.TerminationStatus == poi.TerminationStatusCode.OPTIMAL


def test_setting_gurobi_model_attr():
    # Build an unbounded model
    m = pf.Model("gurobi")
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
    m = pf.Model(solver, solver_uses_variable_names=use_var_names)
    m.A = pf.Variable(ub=10)
    m.maximize = 10 + m.A

    m.optimize()
    assert m.A.solution == approx(10, **get_tol(solver))
    assert m.maximize.value == approx(20, **get_tol(solver))


def test_integers_throw_error(solver: _Solver):
    if solver.supports_integer_variables:
        pytest.skip("This test is only valid for solvers that do not support integers")

    m = pf.Model(solver)
    with pytest.raises(
        ValueError, match="does not support integer or binary variables"
    ):
        m.A = pf.Variable(vtype=pf.VType.INTEGER)
    with pytest.raises(
        ValueError, match="does not support integer or binary variables"
    ):
        m.A = pf.Variable(vtype=pf.VType.BINARY)
