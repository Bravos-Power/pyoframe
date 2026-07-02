"""Defines the functions used to monkey patch polars and pandas."""

import importlib.abc
import importlib.machinery
import sys
from functools import wraps

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


def _patch_polars():
    _patch_class(pl.DataFrame)
    pl.DataFrame.to_expr = lambda self: Param(self)  # type: ignore


def _patch_pandas(pd):
    _patch_class(pd.DataFrame)
    _patch_class(pd.Series)
    pd.DataFrame.to_expr = lambda self: Param(self)  # type: ignore
    pd.Series.to_expr = lambda self: Param(self)  # type: ignore


class PandasFinderInterceptor(importlib.abc.MetaPathFinder):
    class PandasLoaderWrapper(importlib.abc.Loader):
        def __init__(self, original_loader):
            self.original_loader = original_loader

        def exec_module(self, module):
            self.original_loader.exec_module(module)
            _patch_pandas(module)

    def find_spec(self, fullname, path, target=None):
        if path is None and fullname == "pandas":
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = self.PandasLoaderWrapper(spec.loader)
                return spec
        return None


def patch_dataframe_libraries():
    """Will only be called once when Pyoframe is imported."""
    _patch_polars()

    if "pandas" in sys.modules:
        import pandas as pd

        _patch_pandas(pd)
    else:
        sys.meta_path.insert(0, PandasFinderInterceptor())
