import re
import pandas as pd
import numpy as np
import pytest
from pyoframe._arithmetic import PyoframeError
from polars.testing import assert_frame_equal
import polars as pl

from pyoframe.constants import COEF_KEY, CONST_TERM, VAR_KEY
from pyoframe import Variable, Model, sum, Set, Config, Expression, VType
from .util import csvs_to_expr


def test_set_multiplication():
    dim1 = [1, 2, 3]
    dim2 = ["a", "b"]
    assert_frame_equal(Set(x=dim1, y=dim2).data, (Set(x=dim1) * Set(y=dim2)).data)


def test_set_multiplication_same_name():
    dim1 = [1, 2, 3]
    dim2 = ["a", "b"]
    with pytest.raises(AssertionError, match="columns in common"):
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
        check_dtype=False,
    )


def test_within_set():
    small_set = Set(x=[1, 2], y=["a"])
    large_set = Set(x=[1, 2, 3], y=["a", "b", "c"], z=[1])
    v = Variable(large_set)
    result = v.to_expr().within(small_set)
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
        check_dtype=False,
    )


def test_filter_expression():
    expr = pl.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, 3]}).to_expr()
    result = expr.filter(dim1=2)
    assert isinstance(result, Expression)
    assert_frame_equal(
        result.data,
        pl.DataFrame({"dim1": [2], COEF_KEY: [2], VAR_KEY: [CONST_TERM]}),
        check_dtype=False,
    )


def test_filter_constraint():
    const = pl.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, 3]}).to_expr() >= 0
    result = const.filter(dim1=2)
    assert_frame_equal(
        result,
        pl.DataFrame({"dim1": [2], COEF_KEY: [2], VAR_KEY: [CONST_TERM]}),
        check_dtype=False,
    )


def test_filter_variable():
    v = Variable(pl.DataFrame({"dim1": [1, 2, 3]}))
    result = v.filter(dim1=2)
    assert isinstance(result, Expression)
    assert str(result) == "[2]: x2"


def test_filter_set():
    s = Set(x=[1, 2, 3])
    result = s.filter(x=2)
    assert_frame_equal(result.data, pl.DataFrame({"x": [2]}), check_dtype=False)


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
        check_dtype=False,
        check_column_order=False,
    )


def test_add_expressions_with_vars():
    expr = Expression(pl.DataFrame({VAR_KEY: [1, 2], COEF_KEY: [1, 2]}))
    result = expr + expr
    assert_frame_equal(
        result.data,
        pl.DataFrame({VAR_KEY: [1, 2], COEF_KEY: [2, 4]}),
        check_dtype=False,
        check_column_order=False,
    )


def test_add_expressions_with_vars_and_dims():
    expr = Expression(
        pl.DataFrame(
            {"dim1": [1, 1, 2, 2], VAR_KEY: [1, 2, 1, 2], COEF_KEY: [1, 2, 3, 4]}
        )
    )
    result = expr + expr
    assert_frame_equal(
        result.data,
        pl.DataFrame(
            {"dim1": [1, 1, 2, 2], VAR_KEY: [1, 2, 1, 2], COEF_KEY: [2, 4, 6, 8]}
        ),
        check_dtype=False,
        check_column_order=False,
    )


def test_add_expression_with_add_dim():
    expr = pl.DataFrame({"value": [1]}).to_expr()
    expr_with_dim = pl.DataFrame({"dim1": [1], "value": [1]}).to_expr()
    expr_with_two_dim = pl.DataFrame(
        {"dim1": [1], "dim2": ["a"], "value": [1]}
    ).to_expr()

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has missing dimensions ['dim1']. If this is intentional, use .add_dim()"
        ),
    ):
        expr + expr_with_dim
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has missing dimensions ['dim1']. If this is intentional, use .add_dim()"
        ),
    ):
        expr_with_dim + expr
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has missing dimensions ['dim2']. If this is intentional, use .add_dim()"
        ),
    ):
        expr_with_dim + expr_with_two_dim
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has missing dimensions ['dim2']. If this is intentional, use .add_dim()"
        ),
    ):
        expr_with_two_dim + expr_with_dim
    expr.add_dim("dim1") + expr_with_dim
    expr.add_dim("dim1", "dim2") + expr_with_two_dim
    expr_with_dim.add_dim("dim2") + expr_with_two_dim


def test_add_expression_with_vars_and_add_dim():
    expr_with_dim = pl.DataFrame({"dim1": [1, 2], "value": [3, 4]}).to_expr()
    lhs = (1 + 2 * Variable()).add_dim("dim1")
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
        check_dtype=False,
        check_column_order=False,
        check_row_order=False,
    )

    # Now the other way around
    result = expr_with_dim + lhs
    assert_frame_equal(
        result.data,
        expected_result,
        check_dtype=False,
        check_column_order=False,
        check_row_order=False,
    )


def test_add_expression_with_vars_and_add_dim_many():
    dim1 = Set(x=[1, 2])
    dim2 = Set(y=["a", "b"])
    dim3 = Set(z=[4, 5])
    lhs = 1 + 2 * Variable(dim1, dim2)
    rhs = 3 + 4 * Variable(dim3, dim2)

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has missing dimensions ['z']. If this is intentional, use .add_dim()"
        ),
    ):
        lhs + rhs
    lhs = lhs.add_dim("z")
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has missing dimensions ['x']. If this is intentional, use .add_dim()"
        ),
    ):
        lhs + rhs
    rhs = rhs.add_dim("x")
    result = lhs + rhs
    assert (
        str(result)
        == """[1,a,4]: 4  +2 x1 +4 x5
[1,a,5]: 4  +2 x1 +4 x7
[1,b,4]: 4  +2 x2 +4 x6
[1,b,5]: 4  +2 x2 +4 x8
[2,a,4]: 4  +2 x3 +4 x5
[2,a,5]: 4  +2 x3 +4 x7
[2,b,4]: 4  +2 x4 +4 x6
[2,b,5]: 4  +2 x4 +4 x8"""
    )


def test_add_expression_with_missing():
    dim2 = Set(y=["a", "b"])
    dim2_large = Set(y=["a", "b", "c"])
    lhs = 1 + 2 * Variable(dim2)
    rhs = 3 + 4 * Variable(dim2_large)

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()"
        ),
    ):
        lhs + rhs

    result = lhs + rhs.drop_unmatched()
    assert (
        str(result)
        == """[a]: 4  +4 x3 +2 x1
[b]: 4  +4 x4 +2 x2"""
    )
    result = lhs + rhs.keep_unmatched()
    assert (
        str(result)
        == """[a]: 4  +4 x3 +2 x1
[b]: 4  +4 x4 +2 x2
[c]: 3  +4 x5"""
    )

    Config.disable_unmatched_checks = True
    result = lhs + rhs
    assert (
        str(result)
        == """[a]: 4  +2 x1 +4 x3
[b]: 4  +2 x2 +4 x4
[c]: 3  +4 x5"""
    )


def test_add_expressions_with_dims_and_missing():
    dim = Set(x=[1, 2])
    dim2 = Set(y=["a", "b"])
    dim2_large = Set(y=["a", "b", "c"])
    dim3 = Set(z=[4, 5])
    lhs = 1 + 2 * Variable(dim, dim2)
    rhs = 3 + 4 * Variable(dim2_large, dim3)
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has missing dimensions ['z']. If this is intentional, use .add_dim()",
        ),
    ):
        lhs + rhs
    lhs = lhs.add_dim("z")
    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has missing dimensions ['x']. If this is intentional, use .add_dim()",
        ),
    ):
        lhs + rhs
    rhs = rhs.add_dim("x")
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
    assert (
        str(result)
        == """[1,a,4]: 4  +2 x1 +4 x5
[1,a,5]: 4  +2 x1 +4 x6
[1,b,4]: 4  +2 x2 +4 x7
[1,b,5]: 4  +2 x2 +4 x8
[2,a,4]: 4  +2 x3 +4 x5
[2,a,5]: 4  +2 x3 +4 x6
[2,b,4]: 4  +2 x4 +4 x7
[2,b,5]: 4  +2 x4 +4 x8"""
    )


def test_three_way_add():
    df1 = pl.DataFrame({"dim1": [1], "value": [1]}).to_expr()
    df2 = pl.DataFrame({"dim1": [1, 2], "value": [3, 4]}).to_expr()
    df3 = pl.DataFrame({"dim1": [1], "value": [5]}).to_expr()

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()"
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
        check_dtype=False,
        check_column_order=False,
    )

    # Should not throw any errors
    df2.drop_unmatched() + df1 + df3
    df1 + df3 + df2.drop_unmatched()
    result = df1 + df2.drop_unmatched() + df3
    assert_frame_equal(
        result.data,
        pl.DataFrame({"dim1": [1], VAR_KEY: [CONST_TERM], COEF_KEY: [9]}),
        check_dtype=False,
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
            "Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()"
        ),
    ):
        sum("dim1", expr1 + expr2) + expr3

    with pytest.raises(
        PyoframeError,
        match=re.escape(
            "Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()"
        ),
    ):
        sum("dim1", expr1 + expr2.keep_unmatched()) + expr3

    result = sum("dim1", expr1 + expr2.keep_unmatched()) + expr3.drop_unmatched()
    assert str(result) == "[1]: 10"


def test_variable_equals():
    m = Model("max")
    index = Set(x=[1, 2, 3])
    m.Choose = Variable(index, vtype=VType.BINARY)
    with pytest.raises(
        AssertionError,
        match=re.escape("Cannot specify both 'equals' and 'indexing_sets'"),
    ):
        m.Choose100 = Variable(index, equals=100 * m.Choose)
    m.Choose100 = Variable(equals=100 * m.Choose)
    m.objective = sum(m.Choose100)
    m.solve(log_to_console=False)
    assert m.objective.value == 300
