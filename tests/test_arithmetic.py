"""Tests related to Pyoframe's arithmetic operations."""

import re

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyoframe import Config, Expression, Model, Set, Variable, VType
from pyoframe._arithmetic import PyoframeError
from pyoframe._constants import COEF_KEY, CONST_TERM, VAR_KEY

from .util import csvs_to_expr


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
            "Failed to add sets 'unnamed' and 'unnamed' because dimensions do not match (['x'] != ['y'])"
        ),
    ):
        Set(x=[1, 2, 3]) + Set(y=[2, 3, 4])

    added_set = Set(x=[1, 2, 3]) + Set(x=[2, 3, 4])
    assert added_set.data.to_dict(as_series=False) == {"x": [1, 2, 3, 4]}


def test_multiplication_no_common_dimensions():
    val_1 = pl.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, 3]}).to_expr()
    val_2 = pl.DataFrame({"dim2": ["a", "b"], "value": [1, 2]}).to_expr()
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


def test_within_set(solver):
    m = Model(solver=solver)
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
    expr = pl.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, 3]}).to_expr()
    result = expr.filter(dim1=2)
    assert isinstance(result, Expression)
    assert_frame_equal(
        result.data,
        pl.DataFrame({"dim1": [2], COEF_KEY: [2], VAR_KEY: [CONST_TERM]}),
        check_dtypes=False,
    )


def test_filter_constraint():
    const = pl.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, 3]}).to_expr() >= 0
    result = const.filter(dim1=2)
    assert_frame_equal(
        result,
        pl.DataFrame({"dim1": [2], COEF_KEY: [2], VAR_KEY: [CONST_TERM]}),
        check_dtypes=False,
    )


def test_filter_variable(solver):
    m = Model(solver=solver)
    m.v = Variable(pl.DataFrame({"dim1": [1, 2, 3]}))
    result = m.v.filter(dim1=2)
    assert isinstance(result, Expression)
    assert_frame_equal(
        result.to_str(return_df=True),
        pl.DataFrame([[2, "v[2]"]], schema=["dim1", "expression"], orient="row"),
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
        constraint = 5 <= df.to_expr()

        expected_df = pd.DataFrame({"dim1": [1, 2], "value": [1, 2]}).set_index("dim1")[
            "value"
        ]
        expected_constraint = 5 <= expected_df.to_expr()
        assert str(constraint) == str(expected_constraint)


if __name__ == "__main__":
    pytest.main([__file__])

# Matrix of possibilities
# Has multiple dimensions (dim:yes, no)
# Has multiple variable terms (vars:yes, no)
# Requires adding a dimension (add_dim:no, yes_for_left, yes_for_right, yes_for_both, check_raises)
# Has missing values (no, yes_in_left_drop, yes_in_right_drop, yes_in_both_drop, yes_in_left_fill, yes_in_right_fill, yes_in_both_fill, check_raises)


def test_add_expressions():
    expr = pl.DataFrame({"value": [1]}).to_expr()
    result = expr + expr
    assert_frame_equal(
        result.data,
        pl.DataFrame({VAR_KEY: [CONST_TERM], COEF_KEY: [2]}),
        check_dtypes=False,
        check_column_order=False,
    )


def test_add_expressions_with_vars():
    expr = Expression(pl.DataFrame({VAR_KEY: [1, 2], COEF_KEY: [1, 2]}), name="n/a")
    result = expr + expr
    assert_frame_equal(
        result.data,
        pl.DataFrame({VAR_KEY: [1, 2], COEF_KEY: [2, 4]}),
        check_dtypes=False,
        check_column_order=False,
    )


def test_add_expressions_with_vars_and_dims():
    expr = Expression(
        pl.DataFrame(
            {"dim1": [1, 1, 2, 2], VAR_KEY: [1, 2, 1, 2], COEF_KEY: [1, 2, 3, 4]}
        ),
        name="n/a",
    )
    result = expr + expr
    assert_frame_equal(
        result.data,
        pl.DataFrame(
            {"dim1": [1, 1, 2, 2], VAR_KEY: [1, 2, 1, 2], COEF_KEY: [2, 4, 6, 8]}
        ),
        check_dtypes=False,
        check_column_order=False,
    )


def test_add_expression_with_over():
    expr = pl.DataFrame({"value": [1]}).to_expr()
    expr_with_dim = pl.DataFrame({"dim1": [1], "value": [1]}).to_expr()
    expr_with_two_dim = pl.DataFrame(
        {"dim1": [1], "dim2": ["a"], "value": [1]}
    ).to_expr()

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has missing dimensions ['dim1']. If this is intentional, use .over()"
        ),
    ):
        expr + expr_with_dim
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has missing dimensions ['dim1']. If this is intentional, use .over()"
        ),
    ):
        expr_with_dim + expr
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has missing dimensions ['dim2']. If this is intentional, use .over()"
        ),
    ):
        expr_with_dim + expr_with_two_dim
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has missing dimensions ['dim2']. If this is intentional, use .over()"
        ),
    ):
        expr_with_two_dim + expr_with_dim
    expr.over("dim1") + expr_with_dim
    expr.over("dim1", "dim2") + expr_with_two_dim
    expr_with_dim.over("dim2") + expr_with_two_dim


def test_add_expression_with_vars_and_over(solver):
    m = Model(solver=solver)
    m.v = Variable()
    expr_with_dim = pl.DataFrame({"dim1": [1, 2], "value": [3, 4]}).to_expr()
    lhs = (1 + 2 * m.v).over("dim1")
    result = lhs + expr_with_dim
    expected_result = pl.DataFrame(
        {
            "dim1": [1, 2, 1, 2],
            VAR_KEY: [CONST_TERM, CONST_TERM, 1, 1],
            COEF_KEY: [4, 5, 2, 2],
        }
    )
    assert_frame_equal(
        result.data,
        expected_result,
        check_dtypes=False,
        check_column_order=False,
        check_row_order=False,
    )

    # Now the other way around
    result = expr_with_dim + lhs
    assert_frame_equal(
        result.data,
        expected_result,
        check_dtypes=False,
        check_column_order=False,
        check_row_order=False,
    )


def test_add_expression_with_vars_and_over_many(solver):
    dim1 = Set(x=[1, 2])
    dim2 = Set(y=["a", "b"])
    dim3 = Set(z=[4, 5])
    m = Model(solver=solver)
    m.v1 = Variable(dim1, dim2)
    m.v2 = Variable(dim3, dim2)
    lhs = 1 + 2 * m.v1
    rhs = 3 + 4 * m.v2

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has missing dimensions ['z']. If this is intentional, use .over()"
        ),
    ):
        lhs + rhs
    lhs = lhs.over("z")
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has missing dimensions ['x']. If this is intentional, use .over()"
        ),
    ):
        lhs + rhs
    rhs = rhs.over("x")
    result = lhs + rhs
    assert_frame_equal(
        result.to_str(return_df=True),
        pl.DataFrame(
            [
                [1, "a", 4, "4 +2 v1[1,a] +4 v2[4,a]"],
                [1, "a", 5, "4 +2 v1[1,a] +4 v2[5,a]"],
                [1, "b", 4, "4 +2 v1[1,b] +4 v2[4,b]"],
                [1, "b", 5, "4 +2 v1[1,b] +4 v2[5,b]"],
                [2, "a", 4, "4 +2 v1[2,a] +4 v2[4,a]"],
                [2, "a", 5, "4 +2 v1[2,a] +4 v2[5,a]"],
                [2, "b", 4, "4 +2 v1[2,b] +4 v2[4,b]"],
                [2, "b", 5, "4 +2 v1[2,b] +4 v2[5,b]"],
            ],
            schema=["x", "y", "z", "expression"],
            orient="row",
        ),
    )


def test_add_expression_with_missing(solver):
    dim2 = Set(y=["a", "b"])
    dim2_large = Set(y=["a", "b", "c"])
    m = Model(solver=solver)
    m.v1 = Variable(dim2)
    m.v2 = Variable(dim2_large)
    lhs = 1 + 2 * m.v1
    rhs = 3 + 4 * m.v2

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()"
        ),
    ):
        lhs + rhs

    result = lhs + rhs.drop_unmatched()
    assert_frame_equal(
        result.to_str(return_df=True),
        pl.DataFrame(
            [
                ["a", "4 +4 v2[a] +2 v1[a]"],
                ["b", "4 +4 v2[b] +2 v1[b]"],
            ],
            schema=["y", "expression"],
            orient="row",
        ),
    )

    result = lhs + rhs.keep_unmatched()
    assert_frame_equal(
        result.to_str(return_df=True),
        pl.DataFrame(
            [
                ["a", "4 +4 v2[a] +2 v1[a]"],
                ["b", "4 +4 v2[b] +2 v1[b]"],
                ["c", "3 +4 v2[c]"],
            ],
            schema=["y", "expression"],
            orient="row",
        ),
    )

    Config.disable_unmatched_checks = True
    result = lhs + rhs
    assert_frame_equal(
        result.to_str(return_df=True),
        pl.DataFrame(
            [
                ["a", "4 +2 v1[a] +4 v2[a]"],
                ["b", "4 +2 v1[b] +4 v2[b]"],
                ["c", "3 +4 v2[c]"],
            ],
            schema=["y", "expression"],
            orient="row",
        ),
    )


def test_add_expressions_with_dims_and_missing(solver):
    m = Model(solver=solver)
    dim = Set(x=[1, 2])
    dim2 = Set(y=["a", "b"])
    dim2_large = Set(y=["a", "b", "c"])
    dim3 = Set(z=[4, 5])
    m.v1 = Variable(dim, dim2)
    m.v2 = Variable(dim2_large, dim3)
    lhs = 1 + 2 * m.v1
    rhs = 3 + 4 * m.v2
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has missing dimensions ['z']. If this is intentional, use .over()",
        ),
    ):
        lhs + rhs
    lhs = lhs.over("z")
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has missing dimensions ['x']. If this is intentional, use .over()",
        ),
    ):
        lhs + rhs
    rhs = rhs.over("x")
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Cannot add dimension ['x'] since it contains unmatched values. If this is intentional, consider using .drop_unmatched()"
        ),
    ):
        lhs + rhs
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Cannot add dimension ['x'] since it contains unmatched values. If this is intentional, consider using .drop_unmatched()"
        ),
    ):
        lhs.drop_unmatched() + rhs

    result = lhs + rhs.drop_unmatched()
    assert_frame_equal(
        result.to_str(return_df=True),
        pl.DataFrame(
            [
                [1, "a", 4, "4 +2 v1[1,a] +4 v2[a,4]"],
                [1, "a", 5, "4 +2 v1[1,a] +4 v2[a,5]"],
                [1, "b", 4, "4 +2 v1[1,b] +4 v2[b,4]"],
                [1, "b", 5, "4 +2 v1[1,b] +4 v2[b,5]"],
                [2, "a", 4, "4 +2 v1[2,a] +4 v2[a,4]"],
                [2, "a", 5, "4 +2 v1[2,a] +4 v2[a,5]"],
                [2, "b", 4, "4 +2 v1[2,b] +4 v2[b,4]"],
                [2, "b", 5, "4 +2 v1[2,b] +4 v2[b,5]"],
            ],
            schema=["x", "y", "z", "expression"],
            orient="row",
        ),
    )


def test_three_way_add():
    df1 = pl.DataFrame({"dim1": [1], "value": [1]}).to_expr()
    df2 = pl.DataFrame({"dim1": [1, 2], "value": [3, 4]}).to_expr()
    df3 = pl.DataFrame({"dim1": [1], "value": [5]}).to_expr()

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()"
        ),
    ):
        df1 + df2 + df3

    # Should not throw any errors
    df2.keep_unmatched() + df1 + df3
    df1 + df3 + df2.keep_unmatched()
    result = df1 + df2.keep_unmatched() + df3
    assert_frame_equal(
        result.data,
        pl.DataFrame(
            {"dim1": [1, 2], VAR_KEY: [CONST_TERM, CONST_TERM], COEF_KEY: [9, 4]}
        ),
        check_dtypes=False,
        check_column_order=False,
    )

    # Should not throw any errors
    df2.drop_unmatched() + df1 + df3
    df1 + df3 + df2.drop_unmatched()
    result = df1 + df2.drop_unmatched() + df3
    assert_frame_equal(
        result.data,
        pl.DataFrame({"dim1": [1], VAR_KEY: [CONST_TERM], COEF_KEY: [9]}),
        check_dtypes=False,
        check_column_order=False,
    )


def test_no_propogate():
    expr1, expr2, expr3 = csvs_to_expr(
        """
    dim1,dim2,value
    1,1,1
    """,
        """
    dim1,dim2,value
    1,1,2
    2,1,3
    """,
        """
    dim2,value
    1,4
    2,4
    """,
    )

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()"
        ),
    ):
        (expr1 + expr2).sum("dim1") + expr3

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "DataFrame has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()"
        ),
    ):
        (expr1 + expr2.keep_unmatched()).sum("dim1") + expr3

    result = (expr1 + expr2.keep_unmatched()).sum("dim1") + expr3.drop_unmatched()
    assert_frame_equal(
        result.to_str(return_df=True),
        pl.DataFrame([[1, "10"]], schema=["dim2", "expression"], orient="row"),
    )


def test_variable_equals(solver):
    if not solver.supports_integer_variables:
        pytest.skip(
            f"Solver {solver.name} does not support integer or binary variables, skipping test."
        )
    m = Model(solver=solver)
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


def test_adding_expressions_that_cancel(solver):
    m = Model(solver=solver)
    m.x = Variable(pl.DataFrame({"t": [0, 1]}))
    m.y = Variable(pl.DataFrame({"t": [0, 1]}))

    coef_1 = pl.DataFrame({"t": [0, 1], "value": [1, -1]})
    coef_2 = pl.DataFrame({"t": [0, 1], "value": [1, 1]})

    m.c = coef_1 * m.x + coef_2 * m.x + m.y >= 0


def test_adding_cancelling_expressions_no_dim(solver):
    m = Model(solver=solver)
    m.X = Variable()
    m.c = m.X - m.X >= 0


def test_adding_empty_expression(solver):
    m = Model(solver=solver)
    m.x = Variable(pl.DataFrame({"t": [0, 1]}))
    m.y = Variable(pl.DataFrame({"t": [0, 1]}))
    m.z = Variable(pl.DataFrame({"t": [0, 1]}))
    m.c = 0 * m.x + m.y >= 0
    m.c_2 = 0 * m.x + 0 * m.y + m.z >= 0
    m.c_3 = m.z + 0 * m.x + 0 * m.y >= 0


def test_to_and_from_quadratic(solver):
    m = Model(solver=solver)
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
