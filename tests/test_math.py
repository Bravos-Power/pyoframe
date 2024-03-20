import pandas as pd
import numpy as np
import pytest
from pyoframe.constraints import Set
from polars.testing import assert_frame_equal
import polars as pl

from pyoframe.dataframe import COEF_KEY, CONST_TERM, VAR_KEY
from pyoframe import Variable


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
    with pytest.raises(ValueError, match="Cannot add two sets"):
        Set(x=[1, 2, 3]) + Set(x=[1, 2, 3])


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
    large_set = Set(x=[1, 2, 3], y=["a", "b", "c"])
    v = Variable(large_set)
    result = v.to_expr().within(small_set)
    assert_frame_equal(
        result.data,
        pl.DataFrame({"x": [1, 2], "y": ["a", "a"], COEF_KEY: [1, 1], VAR_KEY: [1, 4]}),
        check_dtype=False,
    )


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
        assert constraint == expected_constraint


if __name__ == "__main__":
    pytest.main([__file__])
