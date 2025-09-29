"""Tests related to defining Pyoframe models."""

import pytest
from pytest import approx

import pyoframe as pf
from tests.util import get_tol


def test_set_objective(solver):
    # There are many ways to set the objective in a model.
    # A)
    m = pf.Model(solver)
    m.minimize = 2

    # B)
    m = pf.Model(solver)
    m.maximize = 2

    # C)
    m = pf.Model(solver, sense="min")
    m.objective = 3

    # We can also add or subtract from the objective
    m.minimize += 3
    m.minimize -= 2
    assert m.minimize.evaluate() == 4

    # Other ways should throw an error
    with pytest.raises(
        ValueError, match="Did you use .objective instead of .minimize or .maximize ?"
    ):
        m = pf.Model(solver)
        m.objective = 3

    with pytest.raises(
        ValueError, match="Can't set .minimize in a maximization problem."
    ):
        m = pf.Model(solver, sense="max")
        m.minimize = 3

    with pytest.raises(
        ValueError, match="Can't set .maximize in a minimization problem."
    ):
        m = pf.Model(solver, sense="min")
        m.maximize = 3

    with pytest.raises(
        ValueError, match="Can't get .minimize in a maximization problem."
    ):
        m = pf.Model(solver)
        m.maximize = 3
        m.minimize

    with pytest.raises(
        ValueError, match="Can't get .maximize in a minimization problem."
    ):
        m = pf.Model(solver)
        m.minimize = 3
        m.maximize


def test_quadratic_objective(solver):
    m = pf.Model(solver)
    m.A = pf.Variable()
    m.B = pf.Variable()
    m.minimize = m.A**2 + m.B**2 - m.A - m.B + 2
    m.optimize()
    assert m.A.solution == approx(0.5, **get_tol(solver))
    assert m.B.solution == approx(0.5, **get_tol(solver))
    assert m.objective.value == approx(1.5, **get_tol(solver))
    assert m.objective.evaluate() == approx(1.5, **get_tol(solver))


def test_solver_detection():
    pf.Model("gurobi")
    pf.Model("Gurobi")
    with pytest.raises(ValueError, match="Unsupported solver: 'g urobi'"):
        pf.Model("g urobi")


def test_params():
    m = pf.Model("gurobi")
    m.params.Method = 2
    assert m.params.Method == 2
    # capitalization shouldn't matter
    m.params.method = 3
    assert m.params.Method == 3

    with pytest.raises(KeyError, match="Unknown parameter: 'lkjdgfsg'"):
        m.params.lkjdgfsg
    with pytest.raises(KeyError, match="Unknown parameter: 'lkjdgfsg'"):
        m.params.lkjdgfsg = 4
