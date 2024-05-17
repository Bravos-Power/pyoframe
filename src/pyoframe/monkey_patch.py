import pandas as pd
import polars as pl
from pyoframe.core import SupportsMath
from pyoframe.core import Expression
from functools import wraps

from pyoframe.constants import COEF_KEY, CONST_TERM, VAR_KEY

# pyright: reportAttributeAccessIssue=false


def _patch_class(cls):
    def _patch_method(func):
        @wraps(func)
        def wrapper(self, other):
            if isinstance(other, SupportsMath):
                return NotImplemented
            return func(self, other)

        return wrapper

    cls.__add__ = _patch_method(cls.__add__)
    cls.__mul__ = _patch_method(cls.__mul__)
    cls.__sub__ = _patch_method(cls.__sub__)
    cls.__le__ = _patch_method(cls.__le__)
    cls.__ge__ = _patch_method(cls.__ge__)
    cls.__contains__ = _patch_method(cls.__contains__)


def _dataframe_to_expr(self: pl.DataFrame) -> Expression:
    return Expression(
        self.rename({self.columns[-1]: COEF_KEY})
        .drop_nulls(COEF_KEY)
        .with_columns(pl.lit(CONST_TERM).alias(VAR_KEY))
    )


def patch_dataframe_libraries():
    """
    Applies two patches to the DataFrame and Series classes of both pandas and polars.
    1) Patches arithmetic operators (e.g. `__add__`) such that operations between DataFrames/Series and `Expressionable`s
        are not supported (i.e. `return NotImplemented`). This leads Python to try the reverse operation (e.g. `__radd__`)
        which is supported by the `Expressionable` class.
    2) Adds a `to_expr` method to DataFrame/Series that allows them to be converted to an `Expression` object.
        Series become dataframes and dataframes become expressions where everything but the last column are treated as dimensions.
    """
    _patch_class(pd.DataFrame)
    _patch_class(pd.Series)
    _patch_class(pl.DataFrame)
    _patch_class(pl.Series)
    pl.DataFrame.to_expr = _dataframe_to_expr
    pl.Series.to_expr = lambda self: self.to_frame().to_expr()
    pd.DataFrame.to_expr = lambda self: pl.from_pandas(self).to_expr()
    pd.Series.to_expr = lambda self: self.to_frame().reset_index().to_expr()
