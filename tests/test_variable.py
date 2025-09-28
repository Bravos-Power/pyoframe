"""Tests relating to the pyoframe.Variable class."""

import polars as pl

import pyoframe as pf


def test_equals_param(default_solver):
    m = pf.Model(default_solver)

    x = pf.Set(x=[1, 2, 3])

    m.X1 = pf.Variable(equals=4.3)
    m.X2 = pf.Variable(x, equals=4.3)
    m.X3 = pf.Variable(equals=pl.DataFrame({"x": [1, 2, 3], "value": [1, 2, 3]}))
    m.optimize()

    assert m.X1.solution == 4.3
    assert all(m.X2.solution["solution"] == 4.3)
    assert all(m.X3.solution["solution"] == m.X3.solution["x"])
