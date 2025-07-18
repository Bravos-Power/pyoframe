import pytest
from pytest import approx

import pyoframe as pf
from tests.util import get_tol


def test_set_objective(solver):
    # There are many ways to set the objective in a model.
    # A)
    m = pf.Model(solver=solver)
    m.minimize = 2

    # B)
    m = pf.Model(solver=solver)
    m.maximize = 2

    # C)
    m = pf.Model(solver=solver, sense="min")
    m.objective = 3

    # We can also add or subtract from the objective
    m.minimize += 3
    m.minimize -= 2
    assert m.minimize.evaluate() == 4

    # Other ways should throw an error
    with pytest.raises(
        ValueError, match="Did you use .objective instead of .minimize or .maximize ?"
    ):
        m = pf.Model(solver=solver)
        m.objective = 3

    with pytest.raises(
        ValueError, match="Can't set .minimize in a maximization problem."
    ):
        m = pf.Model(solver=solver, sense="max")
        m.minimize = 3

    with pytest.raises(
        ValueError, match="Can't set .maximize in a minimization problem."
    ):
        m = pf.Model(solver=solver, sense="min")
        m.maximize = 3

    with pytest.raises(
        ValueError, match="Can't get .minimize in a maximization problem."
    ):
        m = pf.Model(solver=solver)
        m.maximize = 3
        m.minimize

    with pytest.raises(
        ValueError, match="Can't get .maximize in a minimization problem."
    ):
        m = pf.Model(solver=solver)
        m.minimize = 3
        m.maximize


def test_quadratic_objective(solver):
    if not solver.supports_quadratics:
        pytest.skip("Highs solver does not support quadratic objectives.")
    m = pf.Model(solver=solver)
    m.A = pf.Variable(lb=0, ub=5)
    m.B = pf.Variable(lb=0, ub=10)
    m.maximize = m.A * m.B + 2
    m.optimize()
    assert m.A.solution == approx(5, **get_tol(solver))
    assert m.B.solution == approx(10, **get_tol(solver))
    assert m.objective.value == approx(52, **get_tol(solver))
    assert m.objective.evaluate() == approx(52, **get_tol(solver))


def test_solver_detection():
    pf.Model(solver="gurobi")
    pf.Model(solver="Gurobi")
    with pytest.raises(ValueError, match="Unsupported solver: 'g urobi'"):
        pf.Model(solver="g urobi")
