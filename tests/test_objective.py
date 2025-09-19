"""Tests for the Objective class."""

import pytest

import pyoframe as pf


def test_get_obj_value(solver):
    if solver.name == "ipopt":
        pytest.skip(
            "See issue: https://github.com/metab0t/PyOptInterface/issues/51 for details."
        )
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
