"""Tests related to Pyoframe's arithmetic operations."""

import re

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyoframe import Expression, Model, Param, Set, Variable
from pyoframe._arithmetic import PyoframeError
from pyoframe._constants import COEF_KEY, CONST_TERM, VAR_KEY


def test_set_multiplication():
    dim1 = [1, 2, 3]
    dim2 = ["a", "b"]
    assert_frame_equal(Set(x=dim1, y=dim2).data, (Set(x=dim1) * Set(y=dim2)).data)


def test_set_multiplication_same_name():
    dim1 = [1, 2, 3]
    dim2 = ["a", "b"]
    with pytest.raises(AssertionError, match="dimension 'x' is present in both sets"):
        Set(x=dim1) * Set(x=dim2)


def test_set_addition():
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Failed to add sets 'unnamed_set' and 'unnamed_set' because dimensions do not match (['x'] != ['y'])"
        ),
    ):
        Set(x=[1, 2, 3]) + Set(y=[2, 3, 4])

    added_set = Set(x=[1, 2, 3]) + Set(x=[2, 3, 4])
    assert added_set.data.to_dict(as_series=False) == {"x": [1, 2, 3, 4]}


def test_multiplication_no_common_dimensions():
    val_1 = Param({"dim1": [1, 2, 3], "value": [1, 2, 3]})
    val_2 = Param({"dim2": ["a", "b"], "value": [1, 2]})
    result = val_1 * val_2
    assert_frame_equal(
        result.data,
        pl.DataFrame(
            {
                "dim1": [1, 1, 2, 2, 3, 3],
                "dim2": ["a", "b", "a", "b", "a", "b"],
                COEF_KEY: [1, 2, 2, 4, 3, 6],
                VAR_KEY: [CONST_TERM] * 6,
            }
        ),
        check_dtypes=False,
    )


def test_within_set(default_solver):
    m = Model(default_solver)
    small_set = Set(x=[1, 2], y=["a"])
    large_set = Set(x=[1, 2, 3], y=["a", "b", "c"], z=[1])
    m.v = Variable(large_set)
    result = m.v.to_expr().within(small_set)
    assert_frame_equal(
        result.data,
        pl.DataFrame(
            {
                "x": [1, 2],
                "y": ["a", "a"],
                "z": [1, 1],
                COEF_KEY: [1, 1],
                VAR_KEY: [1, 4],
            }
        ),
        check_dtypes=False,
    )


def test_filter_expression():
    expr = Param({"dim1": [1, 2, 3], "value": [1, 2, 3]})
    result = expr.filter(dim1=2)
    assert isinstance(result, Expression)
    assert_frame_equal(
        result.data,
        pl.DataFrame({"dim1": [2], COEF_KEY: [2], VAR_KEY: [CONST_TERM]}),
        check_dtypes=False,
    )


def test_filter_constraint():
    const = Param({"dim1": [1, 2, 3], "value": [1, 2, 3]}) >= 0
    result = const.filter(dim1=2)
    assert_frame_equal(
        result,
        pl.DataFrame({"dim1": [2], COEF_KEY: [2], VAR_KEY: [CONST_TERM]}),
        check_dtypes=False,
    )


def test_filter_set():
    s = Set(x=[1, 2, 3])
    result = s.filter(x=2)
    assert_frame_equal(result.data, pl.DataFrame({"x": [2]}), check_dtypes=False)


def test_drops_na():
    for na in [None, float("nan"), np.nan]:
        df = pd.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, na]}).set_index("dim1")[
            "value"
        ]
        constraint = 5 <= Param(df)

        expected_df = pd.DataFrame({"dim1": [1, 2], "value": [1, 2]}).set_index("dim1")[
            "value"
        ]
        expected_constraint = 5 <= Param(expected_df)
        assert str(constraint) == str(expected_constraint)


def test_adding_expressions_that_cancel(default_solver):
    m = Model(default_solver)
    m.x = Variable(pl.DataFrame({"t": [0, 1]}))
    m.y = Variable(pl.DataFrame({"t": [0, 1]}))

    coef_1 = pl.DataFrame({"t": [0, 1], "value": [1, -1]})
    coef_2 = pl.DataFrame({"t": [0, 1], "value": [1, 1]})

    m.c = coef_1 * m.x + coef_2 * m.x + m.y >= 0


def test_adding_cancelling_expressions_no_dim(default_solver):
    m = Model(default_solver)
    m.X = Variable()
    m.c = m.X - m.X >= 0


def test_adding_empty_expression(default_solver):
    m = Model(default_solver)
    m.x = Variable(pl.DataFrame({"t": [0, 1]}))
    m.y = Variable(pl.DataFrame({"t": [0, 1]}))
    m.z = Variable(pl.DataFrame({"t": [0, 1]}))
    m.c = 0 * m.x + m.y >= 0
    m.c_2 = 0 * m.x + 0 * m.y + m.z >= 0
    m.c_3 = m.z + 0 * m.x + 0 * m.y >= 0


def test_to_and_from_quadratic(default_solver):
    m = Model(default_solver)
    df = pl.DataFrame({"dim": [1, 2, 3], "value": [1, 2, 3]})
    m.x1 = Variable()
    m.x2 = Variable()
    expr1 = df * m.x1
    expr2 = df * m.x2 * 2 + 4
    expr3 = expr1 * expr2
    expr4 = expr3 - df * m.x1 * df * m.x2 * 2
    assert expr3.is_quadratic
    assert not expr4.is_quadratic
    assert expr4.terms == 3


def test_division(default_solver):
    df = Param({"dim": ["A", "B", "C"], "value": [1, 2, 3]})

    # Parameter / Constant
    divided = df / 2
    assert_frame_equal(
        divided.data,
        pl.DataFrame(
            {
                "dim": ["A", "B", "C"],
                COEF_KEY: [1 / 2, 2 / 2, 3 / 2],
                VAR_KEY: [CONST_TERM] * 3,
            }
        ),
        check_dtypes=False,
    )

    # Constant / Parameter
    inverse = 1 / df
    assert_frame_equal(
        inverse.data,
        pl.DataFrame(
            {
                "dim": ["A", "B", "C"],
                COEF_KEY: [1 / 1, 1 / 2, 1 / 3],
                VAR_KEY: [CONST_TERM] * 3,
            }
        ),
        check_dtypes=False,
    )

    # Parameter / Parameter
    result = df / df
    assert_frame_equal(
        result.data,
        pl.DataFrame(
            {
                "dim": ["A", "B", "C"],
                COEF_KEY: [1.0, 1.0, 1.0],
                VAR_KEY: [CONST_TERM] * 3,
            }
        ),
        check_dtypes=False,
    )

    # Variable / Parameter
    m = Model(default_solver)
    m.v = Variable(pl.DataFrame({"dim": ["A", "B", "C"]}))
    m.v_expr = m.v.to_expr()
    result = m.v_expr / df
    assert_frame_equal(
        result.data,
        pl.DataFrame(
            {
                "dim": ["A", "B", "C"],
                COEF_KEY: [1 / 1, 1 / 2, 1 / 3],
                VAR_KEY: [1, 2, 3],
            }
        ),
        check_dtypes=False,
    )

    # Variable / Constant
    result = m.v_expr / 2
    assert_frame_equal(
        result.data,
        pl.DataFrame(
            {
                "dim": ["A", "B", "C"],
                COEF_KEY: [1 / 2, 1 / 2, 1 / 2],
                VAR_KEY: [1, 2, 3],
            }
        ),
        check_dtypes=False,
    )

    # Parameter / Variable
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Cannot divide by 'v' because it is not a number or parameter."
        ),
    ):
        _ = df / m.v

    # Parameter / Expression
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Cannot divide by 'v_expr' because denominators cannot contain variables."
        ),
    ):
        _ = df / m.v_expr


if __name__ == "__main__":
    pytest.main([__file__])
