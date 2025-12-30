"""Defines the functions used to monkey patch polars and pandas."""

from functools import wraps

import pandas as pd
import polars as pl

from pyoframe._core import BaseOperableBlock
from pyoframe._param import Param


def _patch_class(cls):
    def _patch_method(func):
        @wraps(func)
        def wrapper(self, other):
            if isinstance(other, BaseOperableBlock):
                return NotImplemented
            return func(self, other)

        return wrapper

    cls.__add__ = _patch_method(cls.__add__)
    cls.__mul__ = _patch_method(cls.__mul__)
    cls.__sub__ = _patch_method(cls.__sub__)
    cls.__le__ = _patch_method(cls.__le__)
    cls.__ge__ = _patch_method(cls.__ge__)
    cls.__lt__ = _patch_method(cls.__lt__)
    cls.__gt__ = _patch_method(cls.__gt__)
    cls.__contains__ = _patch_method(cls.__contains__)


def patch_dataframe_libraries():
    _patch_class(pd.DataFrame)
    _patch_class(pd.Series)
    _patch_class(pl.DataFrame)
    pl.DataFrame.to_expr = lambda self: Param(self)  # type: ignore
    pd.DataFrame.to_expr = lambda self: Param(self)  # type: ignore
    pd.Series.to_expr = lambda self: Param(self)  # type: ignore
