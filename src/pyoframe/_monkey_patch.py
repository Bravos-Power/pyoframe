"""Defines the functions used to monkey patch polars and pandas."""

from functools import wraps

import pandas as pd
import polars as pl

from pyoframe._constants import COEF_KEY, CONST_TERM, VAR_KEY
from pyoframe._core import BaseOperableBlock, Expression


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


def polars_df_to_expr(self: pl.DataFrame) -> Expression:
    """Converts a [polars](https://pola.rs/) `DataFrame` to a Pyoframe [Expression][pyoframe.Expression] by using the last column for values and the previous columns as dimensions.

    See [Special Functions](../../learn/concepts/special-functions.md#dataframeto_expr) for more details.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        >>> df.to_expr()
        <Expression height=3 terms=3 type=constant>
        ┌─────┬─────┬────────────┐
        │ x   ┆ y   ┆ expression │
        │ (3) ┆ (3) ┆            │
        ╞═════╪═════╪════════════╡
        │ 1   ┆ 4   ┆ 7          │
        │ 2   ┆ 5   ┆ 8          │
        │ 3   ┆ 6   ┆ 9          │
        └─────┴─────┴────────────┘
    """
    name = self.columns[-1]
    return Expression(
        self.rename({name: COEF_KEY})
        .drop_nulls(COEF_KEY)
        .with_columns(pl.lit(CONST_TERM).alias(VAR_KEY)),
        name=name,
    )


def pandas_df_to_expr(self: pd.DataFrame) -> Expression:
    """Same as [`polars.DataFrame.to_expr`](./polars.DataFrame.to_expr.md), but for [pandas](https://pandas.pydata.org/) DataFrames."""
    return polars_df_to_expr(pl.from_pandas(self))


def pandas_series_to_expr(self: pd.Series) -> Expression:
    """Converts a [pandas](https://pandas.pydata.org/) `Series` to a Pyoframe [Expression][pyoframe.Expression], using the index for labels.

    See [Special Functions](../../learn/concepts/special-functions.md#dataframeto_expr) for more details.

    Note that no equivalent method exists for Polars Series, as Polars does not support indexes.
    """
    return pandas_df_to_expr(self.to_frame().reset_index())


def patch_dataframe_libraries():
    _patch_class(pd.DataFrame)
    _patch_class(pd.Series)
    _patch_class(pl.DataFrame)
    _patch_class(pl.Series)
    pl.DataFrame.to_expr = polars_df_to_expr
    pd.DataFrame.to_expr = pandas_df_to_expr
    pd.Series.to_expr = pandas_series_to_expr
