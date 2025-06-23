import polars as pl
import pytest

import pyoframe as pf


@pytest.fixture
def cropped_expression():
    pf.Config.print_max_lines = 2
    expr = pl.DataFrame({"t": [1, 2, 3], "val": [1, 2, 3]}).to_expr()
    return expr


def test_ellipsis_expression(cropped_expression):
    assert (
        repr(cropped_expression)
        == """<Expression size=3 dimensions={'t': 3} terms=3>
[1]: 1
[2]: 2
 ⋮"""
    )


def test_ellipsis_named(cropped_expression):
    m = pf.Model()
    m.my_expr = cropped_expression
    assert (
        repr(m.my_expr)
        == """<Expression size=3 dimensions={'t': 3} terms=3>
my_expr[1]: 1
my_expr[2]: 2
        ⋮"""
    )


def test_ellipsis_constraint(cropped_expression):
    m = pf.Model()
    m.v = pf.Variable()
    m.c = m.v.add_dim("t") <= cropped_expression
    assert (
        repr(m.c)
        == """<Constraint sense='<=' size=3 dimensions={'t': 3} terms=6>
c[1]: v <= 1
c[2]: v <= 2
  ⋮"""
    )
