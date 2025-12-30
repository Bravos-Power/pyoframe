"""Tests relating to the pyoframe.Variable class."""

import re

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyoframe import Expression, Model, Param, Set, Variable, VType


def test_equals_param(solver):
    if not solver.supports_integer_variables:
        pytest.skip(
            f"Solver {solver.name} does not support integer or binary variables, skipping test."
        )
    m = Model(solver)
    index = Set(x=[1, 2, 3])

    m.Choose = Variable(index, vtype=VType.BINARY)
    with pytest.raises(
        AssertionError,
        match=re.escape("Cannot specify both 'equals' and 'indexing_sets'"),
    ):
        m.Choose100 = Variable(index, equals=100 * m.Choose)
    m.Choose100 = Variable(equals=100 * m.Choose)
    m.maximize = m.Choose100.sum()
    m.attr.Silent = True
    m.optimize()
    assert m.maximize.value == 300
    assert m.maximize.evaluate() == 300


def test_equals_param_2(default_solver):
    m = Model(default_solver)

    with pytest.raises(
        ValueError,
        match=re.escape("Cannot specify 'lb' when 'equals' is a constant."),
    ):
        m.x = Variable(equals=4.3, lb=1.0)
    with pytest.raises(
        ValueError,
        match=re.escape("Cannot specify 'ub' when 'equals' is a constant."),
    ):
        m.x = Variable(equals=4.3, ub=10.0)

    x = Set(x=[1, 2, 3])

    m.X1 = Variable(equals=4.3)
    m.X2 = Variable(x, equals=4.3)
    m.X3 = Variable(equals=pl.DataFrame({"x": [1, 2, 3], "value": [1, 2, 3]}))
    m.optimize()

    assert m.X1.solution == 4.3
    assert all(m.X2.solution["solution"] == 4.3)
    assert all(m.X3.solution["solution"] == m.X3.solution["x"])


def test_filter_variable(default_solver):
    m = Model(default_solver)
    m.v = Variable(pl.DataFrame({"dim1": [1, 2, 3]}))
    result = m.v.filter(dim1=2)
    assert isinstance(result, Expression)
    assert_frame_equal(
        result.to_str(return_df=True),
        pl.DataFrame([[2, "v[2]"]], schema=["dim1", "expression"], orient="row"),
    )


def test_auto_broadcast(default_solver):
    m = Model(default_solver)

    val = Param({"dim1": [1, 2, 3], "value": [10, 20, 30]})
    altern_dim = Set(dim2=["a", "b"])

    m.v1 = Variable(altern_dim, val, lb=val)
    m.v2 = Variable(altern_dim, val, ub=val)
