import pandas as pd
import polars as pl
from pyoframe.constraints import Expressionable
from pyoframe.constraints import Expression
from functools import wraps

from pyoframe.dataframe import COEF_KEY, CONST_TERM, VAR_KEY

# pyright: reportAttributeAccessIssue=false


def patch_method(func):
    @wraps(func)
    def wrapper(self, other):
        if isinstance(other, Expressionable):
            return NotImplemented
        return func(self, other)

    return wrapper


def patch_class(cls):
    cls.__add__ = patch_method(cls.__add__)
    cls.__mul__ = patch_method(cls.__mul__)
    cls.__sub__ = patch_method(cls.__sub__)
    cls.__le__ = patch_method(cls.__le__)
    cls.__ge__ = patch_method(cls.__ge__)
    cls.__contains__ = patch_method(cls.__contains__)


patch_class(pd.DataFrame)
patch_class(pd.Series)
patch_class(pl.DataFrame)


def to_expr(self: pl.DataFrame) -> Expression:
    return Expression(
        self.rename({self.columns[-1]: COEF_KEY})
        .drop_nulls(COEF_KEY)
        .with_columns(pl.lit(CONST_TERM).alias(VAR_KEY))
    )


pl.DataFrame.to_expr = to_expr
pd.DataFrame.to_expr = lambda self: pl.from_pandas(self).to_expr()
pd.Series.to_expr = lambda self: self.to_frame().reset_index().to_expr()
