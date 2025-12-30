"""Tests related to the propagation of BaseBlock.name in Pyoframe objects."""

import polars as pl
import pytest

import pyoframe as pf


def test_param_smart_naming():
    df = pl.DataFrame({"dim": [1, 2], "val": [2, 3]})
    assert pf.Param(df).name == "val"


def test_variables(default_solver):
    assert pf.Variable().name == "unnamed"

    m = pf.Model(default_solver)
    m.my_var = pf.Variable()

    assert m.my_var.name == "my_var"
    assert m.my_var.to_expr().name == "my_var"


def test_expressions(default_solver):
    m = pf.Model(default_solver)
    m.X = pf.Variable()
    m.Y = pf.Variable()

    assert (m.X - 1).name == "(X - 1)"
    assert (1 + m.X).name == "(X + 1)"
    assert (m.X * -2).name == "(-2 * X)"
    assert (m.X - m.Y).name == "(X - Y)"
    assert (-m.X).name == "-X"
    assert (-(2 * (1 - m.X))).name == "-(2 * (-X + 1))"
    assert (m.X**2).name == "(X**2)"

    assert (m.X - pf.Expression.constant(1)).name == "(X - 1)"

    # null operation
    assert (m.X + 0).name == "X"
    assert (m.X * 1).name == "X"


def test_transforms(default_solver):
    df = pl.DataFrame({"dim": [1, 2], "val": [2, 3]})

    m = pf.Model(default_solver)
    m.X = pf.Variable(df["dim"])

    assert m.X.next("dim").name == "X.next(…)"

    assert m.X.rename({"dim": "dim2"}).name == "X.rename(…)"

    mapping = pl.DataFrame({"dim": [1, 2], "dim2": [2, 3]})
    assert m.X.map(mapping).name == "X.map(…)"

    assert m.X.pick(dim=2).name == "X.pick(…)"

    # compound
    assert (m.X.next("dim", wrap_around=True) + df).rename(
        {"dim": "dim2"}
    ).name == "(X.next(…) + val).rename(…)"


def test_set():
    a = pf.Set(dim=["a", "b", "c"])
    assert a.name == "unnamed_set"

    b = pf.Set(dim=["b", "c", "d"])

    assert (a + b).name == "(unnamed_set + unnamed_set)"


def test_warning_without_name():
    data = pf.Set(x=[1, 2]).to_expr().data
    with pytest.warns(UserWarning):
        pf.Expression(data)
