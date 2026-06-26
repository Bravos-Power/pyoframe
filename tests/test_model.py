"""Tests related to defining Pyoframe models."""

import re

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


def test_verbose(default_solver, caplog):
    m = pf.Model(default_solver, verbose=True)

    m.X = pf.Variable(lb=0)
    assert "Added variable 'X'" in caplog.text

    m.constr = m.X >= 5
    assert "Added constraint 'constr'" in caplog.text


def test_protected_variables(default_solver):
    m = pf.Model(default_solver)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Cannot assign name 'constraints' to Variable because 'constraints' is already in use."
        ),
    ):
        m.constraints = pf.Variable()


def test_delete_variable(solver):
    m = pf.Model(solver)

    # Basic case
    m.X = pf.Variable()

    assert m._var_map.mapping_registry.height == 2
    assert len(m.variables) == 1

    m.X.attr.Name

    if not solver.supports_deletion:
        with pytest.raises(Exception, match="does not support deletion"):
            del m.X
        return

    saved_copy = m.X
    del m.X

    assert m._var_map.mapping_registry.height == 1
    assert len(m.variables) == 0

    with pytest.raises(RuntimeError, match="Variable does not exist"):
        saved_copy.attr.Name

    # Dimensional case
    m.X = pf.Variable(pf.Set(y=[1, 2, 3]))
    assert len(m.variables) == 1
    assert m._var_map.mapping_registry.height == 4
    saved_copy = m.X
    saved_copy.attr.Name
    del m.X
    assert len(m.variables) == 0
    assert m._var_map.mapping_registry.height == 1
    with pytest.raises(RuntimeError, match="Variable does not exist"):
        saved_copy.attr.Name

    # When already in constraints it should be as if it were never there
    m.X = pf.Variable(ub=10)
    m.Y = pf.Variable(ub=10)
    m.Z = pf.Variable(ub=10)
    m.maximize = m.X + m.Y + m.Z
    m.optimize()
    assert m.objective.value == 30
    del m.X
    m.optimize()
    assert m.objective.value == 20


def test_delete_constraint(solver):
    m = pf.Model(solver)
    m.X = pf.Variable(lb=0)
    m.Y = pf.Variable(pf.Set(y=[1, 2, 3]), lb=0)
    m.minimize = m.X + m.Y.sum()

    # Basic case
    m.constr = m.X >= 5
    m.optimize()
    assert m.objective.value == approx(5, **get_tol(solver))

    m.constr.attr.Dual

    # TODO once https://github.com/metab0t/PyOptInterface/issues/103 is fixed, we can remove the check for "mosek" and update the error message
    if not solver.supports_deletion or solver.name == "mosek":
        with pytest.raises(Exception, match="does not support constraint deletion"):
            del m.constr
        return

    saved_copy = m.constr
    del m.constr
    assert len(m.constraints) == 0
    # Once this bug in PyOptInterface is fixed, we can change the error message to "Constraint does not exist"
    # See https://github.com/metab0t/PyOptInterface/pull/102
    err_msg = (
        "Constraint does not exist"
        if solver.name != "gurobi"
        else "Variable does not exist"
    )
    with pytest.raises(RuntimeError, match=err_msg):
        saved_copy.attr.Dual

    # Dimensional case
    m.constr = m.Y >= 5
    m.optimize()
    assert m.objective.value == approx(15, **get_tol(solver))
    saved_copy = m.constr
    saved_copy.attr.Dual
    del m.constr
    assert len(m.constraints) == 0
    with pytest.raises(RuntimeError, match=err_msg):
        saved_copy.attr.Dual

    if solver.supports_quadratic_constraints:
        # Quadratic case
        m.constr = m.X**2 >= 25
        m.optimize()
        assert m.objective.value == approx(5, **get_tol(solver))
        del m.constr
        m.optimize()
        assert m.objective.value == approx(0, **get_tol(solver))
