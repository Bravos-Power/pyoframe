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


def test_addition_modifiers():
    expr = pf.Param({"dim": [1, 2], "val": [2, 3]})
    assert expr.name == "val"

    assert expr.keep_extras().name == "val.keep_extras()"

    assert expr.keep_extras().keep_extras().name == "val.keep_extras()"
    assert expr.drop_extras().drop_extras().name == "val.drop_extras()"
    assert expr.raise_extras().raise_extras().name == "val"

    assert (expr | expr).name == "(val.keep_extras() + val.keep_extras())"
    assert (
        expr | expr | expr
    ).name == "((val.keep_extras() + val.keep_extras()) + val.keep_extras())"


# --- Dict-style component access tests ---


def test_dict_style_set_get_non_identifier(default_solver):
    """Dict-style set/get with names that aren't valid Python identifiers."""
    m = pf.Model(default_solver)
    m["Variable--2 xy"] = pf.Variable()
    assert m["Variable--2 xy"].name == "Variable--2 xy"

    m["con: budget ≤ 10"] = m["Variable--2 xy"] <= 10
    assert m["con: budget ≤ 10"].name == "con: budget ≤ 10"


def test_dict_style_unified_access(default_solver):
    """Attribute-assigned components are also accessible via dict, and vice versa."""
    m = pf.Model(default_solver)

    # Attribute assignment -> dict access
    m.X = pf.Variable()
    assert m["X"] is m.X

    # Dict assignment with valid identifier -> attribute access
    m["Y"] = pf.Variable()
    assert m.Y is m["Y"]


def test_contains(default_solver):
    """__contains__ works for both attribute and dict-assigned components."""
    m = pf.Model(default_solver)
    m.X = pf.Variable()
    m["fancy name"] = pf.Variable()

    assert "X" in m
    assert "fancy name" in m
    assert "Z" not in m


def test_dict_style_expression_bounds(default_solver):
    """Non-identifier variables with expression bounds create derived _lb/_ub constraints."""
    m = pf.Model(default_solver)
    dims = {"dim": [1, 2]}
    m["my var"] = pf.Variable(dims, lb=0, ub=pf.Param({"dim": [1, 2], "val": [5, 5]}))

    # The derived constraints should be accessible via dict
    assert "my var_ub" in m
    assert isinstance(m["my var_ub"], pf.Constraint)


def test_dict_style_error_reserved_name(default_solver):
    """Setting a reserved name via [] raises an error."""
    m = pf.Model(default_solver)
    with pytest.raises(Exception):
        m["_variables"] = pf.Variable()


def test_dict_style_error_duplicate_name(default_solver):
    """Setting a duplicate name raises an AssertionError."""
    m = pf.Model(default_solver)
    m["X"] = pf.Variable()
    with pytest.raises(AssertionError, match="already created"):
        m["X"] = pf.Variable()


def test_dict_style_error_invalid_type(default_solver):
    """Setting a non-BaseBlock value raises PyoframeError."""
    m = pf.Model(default_solver)
    with pytest.raises(Exception):
        m["X"] = 42


def test_dict_style_error_missing_key(default_solver):
    """Getting a missing key raises KeyError."""
    m = pf.Model(default_solver)
    with pytest.raises(KeyError, match="not found"):
        m["nonexistent"]


def test_dict_style_end_to_end_solve(default_solver):
    """End-to-end: model with dict-style names solves correctly."""
    m = pf.Model(default_solver)
    m["x 1"] = pf.Variable(lb=0)
    m["x 2"] = pf.Variable(lb=0)
    m["budget constraint"] = m["x 1"] + m["x 2"] <= 10
    m.minimize = m["x 1"] + m["x 2"]
    m.optimize()
    assert m["x 1"].solution == pytest.approx(0.0)
    assert m["x 2"].solution == pytest.approx(0.0)
