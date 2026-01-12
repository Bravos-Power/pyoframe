"""Tests for the Objective class."""

import re

import pytest

import pyoframe as pf


def test_get_obj_value(solver):
    m = pf.Model(solver=solver)
    m.X = pf.Variable(ub=5)

    m.maximize = m.X

    with pytest.raises(
        ValueError,
        match="Cannot retrieve the objective value before calling model.optimize()",
    ):
        m.objective.value

    m.optimize()

    assert m.objective.value == pytest.approx(5)


@pytest.mark.parametrize("obj", [5, pf.Expression.constant(10)])
def test_creation(default_solver, obj):
    m = pf.Model(solver=default_solver)

    # cannot retrieve undefined objective
    with pytest.raises(ValueError, match=re.escape("Objective is not defined.")):
        m.objective

    # Should throw error because missing sense
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Can't set an objective without specifying the sense. Did you use .objective instead of .minimize or .maximize ?"
        ),
    ):
        m.objective = obj

    # This should work instead
    m = pf.Model(solver=default_solver, sense="min")
    m.objective = obj
    assert m.sense == pf.ObjSense.MIN

    # Cannot redefine objective
    with pytest.raises(
        ValueError,
        match=re.escape("An objective already exists. Use += or -= to modify it."),
    ):
        m.objective = obj + obj

    # But adding should work fine
    m.objective += obj
    m.objective -= obj

    # Using minimize/maximize should also work
    m = pf.Model(solver=default_solver)
    m.minimize = obj
    assert m.sense == pf.ObjSense.MIN

    # Unless the sense keyword conflicts of course
    m = pf.Model(solver=default_solver, sense="max")
    with pytest.raises(
        ValueError,
        match=re.escape("Can't set .minimize in a maximization problem."),
    ):
        m.minimize = obj

    # Modifying an unset objective should still be helpful.
    m = pf.Model(solver=default_solver)
    with pytest.raises(
        ValueError,
        match=re.escape("Objective is not defined."),
    ):
        m.objective += obj
    with pytest.raises(
        ValueError,
        match=re.escape("Objective is not defined."),
    ):
        m.minimize += obj
