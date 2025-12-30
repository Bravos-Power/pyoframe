"""Tests related to the addition of Expressions in Pyoframe.

Particular attention is paid to addition modifiers. See:
bravos-power.github.io/pyoframe/latest/learn/concepts/addition
"""

import re

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyoframe import Config, Expression, Model, Param, Set, Variable
from pyoframe._arithmetic import PyoframeError
from pyoframe._constants import COEF_KEY, CONST_TERM, VAR_KEY, ExtrasStrategy

from .util import csvs_to_expr

# Matrix of possibilities
# Has multiple dimensions (dim:yes, no)
# Has multiple variable terms (vars:yes, no)
# Requires adding a dimension (add_dim:no, yes_for_left, yes_for_right, yes_for_both, check_raises)
# Has missing values (no, yes_in_left_drop, yes_in_right_drop, yes_in_both_drop, yes_in_left_fill, yes_in_right_fill, yes_in_both_fill, check_raises)


def test_add_expressions():
    expr = Param({"value": [1]})
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
    expr = Param({"value": [1]})
    expr_with_dim = Param({"dim1": [1], "value": [1]})
    expr_with_two_dim = Param({"dim1": [1], "dim2": ["a"], "value": [1]})

    with pytest.raises(
        PyoframeError,
        match=re.escape("If this is intentional, use .over(…)"),
    ):
        expr + expr_with_dim
    with pytest.raises(
        PyoframeError,
        match=re.escape("If this is intentional, use .over(…)"),
    ):
        expr_with_dim + expr
    with pytest.raises(
        PyoframeError,
        match=re.escape("If this is intentional, use .over(…)"),
    ):
        expr_with_dim + expr_with_two_dim
    with pytest.raises(
        PyoframeError,
        match=re.escape("If this is intentional, use .over(…)"),
    ):
        expr_with_two_dim + expr_with_dim
    expr.over("dim1") + expr_with_dim
    expr.over("dim1", "dim2") + expr_with_two_dim
    expr_with_dim.over("dim2") + expr_with_two_dim


def test_add_expression_with_vars_and_over(default_solver):
    m = Model(default_solver)
    m.v = Variable()
    expr_with_dim = Param({"dim1": [1, 2], "value": [3, 4]})
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


def test_add_expression_with_vars_and_over_many(default_solver):
    dim1 = Set(x=[1, 2])
    dim2 = Set(y=["a", "b"])
    dim3 = Set(z=[4, 5])
    m = Model(default_solver)
    m.v1 = Variable(dim1, dim2)
    m.v2 = Variable(dim3, dim2)
    lhs = 1 + 2 * m.v1
    rhs = 3 + 4 * m.v2

    with pytest.raises(
        PyoframeError,
        match=re.escape("If this is intentional, use .over(…)"),
    ):
        lhs + rhs
    lhs = lhs.over("z")
    with pytest.raises(
        PyoframeError,
        match=re.escape("If this is intentional, use .over(…)"),
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


def test_add_expression_with_extras(default_solver):
    dim2 = Set(y=["a", "b"])
    dim2_large = Set(y=["a", "b", "c"])
    m = Model(default_solver)
    m.v1 = Variable(dim2)
    m.v2 = Variable(dim2_large)
    lhs = 1 + 2 * m.v1
    rhs = 3 + 4 * m.v2

    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        lhs + rhs

    result = lhs + rhs.drop_extras()
    assert_frame_equal(
        result.to_str(return_df=True),
        pl.DataFrame(
            [
                ["a", "4 +2 v1[a] +4 v2[a]"],
                ["b", "4 +2 v1[b] +4 v2[b]"],
            ],
            schema=["y", "expression"],
            orient="row",
        ),
    )

    result = lhs + rhs.keep_extras()
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

    Config.disable_extras_checks = True
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


def test_add_expressions_with_keep_and_drop(default_solver):
    m = Model(default_solver)
    x1 = Set(x=[1, 2, 3])
    x2 = Set(x=[2, 3, 4])
    m.v1 = Variable(x1)
    m.v2 = Variable(x2)

    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        m.v1 + m.v2
    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        m.v1.keep_extras() + m.v2
    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        m.v1 + m.v2.keep_extras()
    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        m.v1.drop_extras() + m.v2
    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        m.v1 + m.v2.drop_extras()

    result_kk = m.v1.keep_extras() + m.v2.keep_extras()
    result_kd = m.v1.keep_extras() + m.v2.drop_extras()
    result_dk = m.v1.drop_extras() + m.v2.keep_extras()
    result_dd = m.v1.drop_extras() + m.v2.drop_extras()

    assert_frame_equal(
        result_kk.to_str(return_df=True),
        pl.DataFrame(
            [[1, "v1[1]"], [2, "v1[2] + v2[2]"], [3, "v1[3] + v2[3]"], [4, "v2[4]"]],
            schema=["x", "expression"],
            orient="row",
        ),
    )

    assert_frame_equal(
        result_kd.to_str(return_df=True),
        pl.DataFrame(
            [[1, "v1[1]"], [2, "v1[2] + v2[2]"], [3, "v1[3] + v2[3]"]],
            schema=["x", "expression"],
            orient="row",
        ),
    )

    assert_frame_equal(
        result_dk.to_str(return_df=True),
        pl.DataFrame(
            [[2, "v1[2] + v2[2]"], [3, "v1[3] + v2[3]"], [4, "v2[4]"]],
            schema=["x", "expression"],
            orient="row",
        ),
    )

    assert_frame_equal(
        result_dd.to_str(return_df=True),
        pl.DataFrame(
            [[2, "v1[2] + v2[2]"], [3, "v1[3] + v2[3]"]],
            schema=["x", "expression"],
            orient="row",
        ),
    )


def test_add_expressions_with_dims_and_extras(default_solver):
    m = Model(default_solver)
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
            "If this is intentional, use .over(…)",
        ),
    ):
        lhs + rhs
    lhs = lhs.over("z")
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "If this is intentional, use .over(…)",
        ),
    ):
        lhs + rhs
    rhs = rhs.over("x")
    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        lhs + rhs
    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        lhs.drop_extras() + rhs

    result = lhs + rhs.drop_extras()
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
    df1 = Param({"dim1": [1], "value": [1]})
    df2 = Param({"dim1": [1, 2], "value": [3, 4]})
    df3 = Param({"dim1": [1], "value": [5]})

    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        df1 + df2 + df3

    # Should not throw any errors
    df2.keep_extras() + df1 + df3
    df1 + df3 + df2.keep_extras()
    result = df1 + df2.keep_extras() + df3
    assert_frame_equal(
        result.data,
        pl.DataFrame(
            {"dim1": [1, 2], VAR_KEY: [CONST_TERM, CONST_TERM], COEF_KEY: [9, 4]}
        ),
        check_dtypes=False,
        check_column_order=False,
    )

    # Should not throw any errors
    df2.drop_extras() + df1 + df3
    df1 + df3 + df2.drop_extras()
    result = df1 + df2.drop_extras() + df3
    assert_frame_equal(
        result.data,
        pl.DataFrame({"dim1": [1], VAR_KEY: [CONST_TERM], COEF_KEY: [9]}),
        check_dtypes=False,
        check_column_order=False,
    )


def test_propagation_extras():
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

    assert expr1.keep_extras()._extras_strategy == ExtrasStrategy.KEEP
    assert expr1._extras_strategy == ExtrasStrategy.UNSET, (
        "keep_extras() should not modify the original expression"
    )

    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        (expr1 + expr2).sum("dim1") + expr3

    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        (expr1 + expr2.keep_extras()).sum("dim1") + expr3

    result = (expr1 + expr2.keep_extras()).sum("dim1") + expr3.drop_extras()
    assert_frame_equal(
        result.to_str(return_df=True),
        pl.DataFrame([[1, "10"]], schema=["dim2", "expression"], orient="row"),
    )


def test_propagation_over():
    set_x = Set(x=[1, 2, 3])
    set_xy = Set(x=[1, 2], y=["a", "b"])
    expr1 = set_x.to_expr()
    expr2 = set_xy.to_expr()

    assert "random" in expr1.over("random")._allowed_new_dims
    assert "random" not in expr1._allowed_new_dims, (
        "over() should not modify the original expression"
    )

    with pytest.raises(
        PyoframeError,
        match=re.escape("If this is intentional, use .over(…)"),
    ):
        expr1 + expr2

    with pytest.raises(
        PyoframeError,
        match=re.escape("Use .drop_extras() or .keep_extras()"),
    ):
        expr1.over("y") + expr2

    res1 = expr1.over("y").drop_extras() + expr2
    res2 = expr1.drop_extras().over("y") + expr2
    assert_frame_equal(res1.data, res2.data)

    # check that negation also carries properties
    res1 = -expr1.over("y").drop_extras() + expr2
    res2 = -expr1.drop_extras().over("y") + expr2
    assert_frame_equal(res1.data, res2.data)
