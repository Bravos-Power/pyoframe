"""Tests relating to constraints."""

import re

import polars as pl
import pytest

from pyoframe import Model, PyoframeError, Variable


def test_creation(default_solver):
    m = Model(default_solver)
    df = pl.DataFrame({"x": [1, 2, 3], "val": [10, 20, 30]})

    m.X = Variable(df["x"])

    with pytest.raises(
        PyoframeError, match=re.escape("Did you mean to use '<=' or '>=' instead?")
    ):
        _ = m.X > df
    with pytest.raises(
        PyoframeError, match=re.escape("Did you mean to use '<=' or '>=' instead?")
    ):
        _ = df > m.X
    with pytest.raises(
        PyoframeError, match=re.escape("Did you mean to use '<=' or '>=' instead?")
    ):
        _ = m.X < df
    with pytest.raises(
        PyoframeError, match=re.escape("Did you mean to use '<=' or '>=' instead?")
    ):
        _ = df < m.X

    m.c1 = m.X >= df
    m.c2 = m.X <= df
    m.c3 = df >= m.X
    m.c4 = df <= m.X
    m.c5 = m.X == df
