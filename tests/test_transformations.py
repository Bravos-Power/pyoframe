"""Tests related to Pyoframe transformations (e.g. .sum())."""

import re

import pytest
from util import csvs_to_expr

import pyoframe as pf


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
