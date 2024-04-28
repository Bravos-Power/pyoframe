import pyoframe as pf
from tests.util import csvs_to_expr
import pytest
import pyoframe as pf
import re


def test_sum():
    expr = csvs_to_expr(
        """
    day,water_drank
    1,2
    2,3
    3,4
"""
    )

    with pytest.raises(
        ValueError,
        match=re.escape("Perhaps you meant to use pf.sum() instead of sum()?"),
    ):
        sum("day", expr)

    result = pf.sum("day", expr)
    assert str(result) == "9"


def test_to_str():
    expr = csvs_to_expr(
        """
    day,water_drank
    1,2.00000000001
    2,3
    3,4
"""
    )

    assert str(expr) == "[1]: 2.00000000001\n[2]: 3\n[3]: 4"
    # str() is the same as to_str()
    assert expr.to_str() == str(expr)
    assert expr.to_str(float_precision=6) == "[1]: 2\n[2]: 3\n[3]: 4"
    # repr() is what is used when the object is printed in the console
    assert (
        repr(expr)
        == "<Expression size=3 dimensions={'day': 3} terms=3>\n[1]: 2\n[2]: 3\n[3]: 4"
    )
    pf.Config.print_float_precision = None
    assert (
        repr(expr)
        == "<Expression size=3 dimensions={'day': 3} terms=3>\n[1]: 2.00000000001\n[2]: 3\n[3]: 4"
    )
