"""Contains utility functions and classes."""

from __future__ import annotations

import itertools
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
import polars as pl

from pyoframe._constants import (
    COEF_KEY,
    CONST_TERM,
    RESERVED_COL_KEYS,
    VAR_KEY,
    Config,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe._core import BaseOperableBlock
    from pyoframe._model import Variable

if sys.version_info >= (3, 10):
    pairwise = itertools.pairwise
else:

    def pairwise(iterable):
        iterator = iter(iterable)
        a = next(iterator)

        for b in iterator:
            yield a, b
            a = b


def get_obj_repr(obj: object, *props: str | None, **kwargs):
    """Generates __repr__() strings for classes.

    See usage for examples.
    """
    props_str = " ".join(v for v in props if v is not None)
    if props_str:
        props_str += " "
    kwargs_str = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
    return f"<{obj.__class__.__name__} {props_str}{kwargs_str}>"


def parse_inputs_as_iterable(
    *inputs: Any | Iterable[Any],
) -> Iterable[Any]:
    """Converts a parameter *x: Any | Iterable[Any] to a single Iterable[Any] object.

    This is helpful to support these two ways of passing arguments:
        - foo([1, 2, 3])
        - foo(1, 2, 3)

    Inspired from the polars library.
    """
    if not inputs:
        return []

    # Treat elements of a single iterable as separate inputs
    if len(inputs) == 1 and _is_iterable(inputs[0]):
        return inputs[0]

    return inputs


def _is_iterable(input: Any | Iterable[Any]) -> bool:
    # Inspired from the polars library, TODO: Consider using opposite check, i.e. equals list or tuple
    return isinstance(input, Iterable) and not isinstance(
        input,
        (
            str,
            bytes,
            pl.DataFrame,
            pl.Series,
            pd.DataFrame,
            pd.Series,
            pd.Index,
            dict,
            range,
        ),
    )


def concat_dimensions(
    df: pl.DataFrame, prefix: str, keep_dims: bool = True, to_col: str = "concated_dim"
) -> pl.DataFrame:
    """Returns a new DataFrame with the column 'concated_dim'.

    Reserved columns are ignored. Spaces are replaced with underscores.

    Parameters:
        df:
            The input DataFrame.
        prefix:
            The prefix to be added to the concated dimension.
        keep_dims:
            If `True`, the original dimensions are kept in the new DataFrame.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame(
        ...     {
        ...         "dim1": [1, 2, 3, 1, 2, 3],
        ...         "dim2": ["Y", "Y", "Y", "N", "N", "N"],
        ...     }
        ... )
        >>> concat_dimensions(df, prefix="x")
        shape: (6, 3)
        ┌──────┬──────┬──────────────┐
        │ dim1 ┆ dim2 ┆ concated_dim │
        │ ---  ┆ ---  ┆ ---          │
        │ i64  ┆ str  ┆ str          │
        ╞══════╪══════╪══════════════╡
        │ 1    ┆ Y    ┆ x[1,Y]       │
        │ 2    ┆ Y    ┆ x[2,Y]       │
        │ 3    ┆ Y    ┆ x[3,Y]       │
        │ 1    ┆ N    ┆ x[1,N]       │
        │ 2    ┆ N    ┆ x[2,N]       │
        │ 3    ┆ N    ┆ x[3,N]       │
        └──────┴──────┴──────────────┘
        >>> concat_dimensions(df, prefix="", keep_dims=False)
        shape: (6, 1)
        ┌──────────────┐
        │ concated_dim │
        │ ---          │
        │ str          │
        ╞══════════════╡
        │ [1,Y]        │
        │ [2,Y]        │
        │ [3,Y]        │
        │ [1,N]        │
        │ [2,N]        │
        │ [3,N]        │
        └──────────────┘

        Properly handles cases with no dimensions and ignores reserved columns
        >>> df = pl.DataFrame({VAR_KEY: [1, 2]})
        >>> concat_dimensions(df, prefix="x")
        shape: (2, 2)
        ┌───────────────┬──────────────┐
        │ __variable_id ┆ concated_dim │
        │ ---           ┆ ---          │
        │ i64           ┆ str          │
        ╞═══════════════╪══════════════╡
        │ 1             ┆ x            │
        │ 2             ┆ x            │
        └───────────────┴──────────────┘
    """
    if prefix is None:
        prefix = ""
    dimensions = [col for col in df.columns if col not in RESERVED_COL_KEYS]
    if dimensions:
        query = pl.concat_str(
            pl.lit(prefix + "["),
            pl.concat_str(*dimensions, separator=","),
            pl.lit("]"),
        )
    else:
        query = pl.lit(prefix)

    df = df.with_columns(query.str.replace_all(" ", "_").alias(to_col))

    if not keep_dims:
        df = df.drop(*dimensions)

    return df


def cast_coef_to_string(
    df: pl.DataFrame,
    column_name: str = COEF_KEY,
    drop_ones: bool = True,
    always_show_sign: bool = True,
) -> pl.DataFrame:
    """Converts column `column_name` of the DataFrame `df` to a string. Round to `Config.print_float_precision` decimal places if not None.

    Parameters:
        df:
            The input DataFrame.
        column_name:
            The name of the column to be casted.
        drop_ones:
            If `True`, 1s are replaced with an empty string for non-constant terms.
        always_show_sign:
            If `True`, the sign of the coefficient is always shown, i.e. 1 becomes `+1` not just `1`.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"x": [1.0, -2.0, 1.0, 4.0], VAR_KEY: [1, 2, 0, 4]})
        >>> cast_coef_to_string(df, "x")
        shape: (4, 2)
        ┌─────┬───────────────┐
        │ x   ┆ __variable_id │
        │ --- ┆ ---           │
        │ str ┆ i64           │
        ╞═════╪═══════════════╡
        │ +   ┆ 1             │
        │ -2  ┆ 2             │
        │ +1  ┆ 0             │
        │ +4  ┆ 4             │
        └─────┴───────────────┘
    """
    if Config.float_to_str_precision is not None:
        df = df.with_columns(pl.col(column_name).round(Config.float_to_str_precision))

    if always_show_sign:
        df = df.with_columns(
            pl.col(column_name).abs(),
            _sign=pl.when(pl.col(column_name) < 0)
            .then(pl.lit("-"))
            .otherwise(pl.lit("+")),
        )

    df = df.with_columns(
        pl.when(pl.col(column_name) == pl.col(column_name).round())
        .then(pl.col(column_name).cast(pl.Int64).cast(pl.String))
        .otherwise(pl.col(column_name).cast(pl.String))
        .alias(column_name)
    )

    if drop_ones:
        assert always_show_sign, "drop_ones requires always_show_sign=True"
        condition = pl.col(column_name) == str(1)
        if VAR_KEY in df.columns:
            condition = condition & (pl.col(VAR_KEY) != CONST_TERM)
        df = df.with_columns(
            pl.when(condition)
            .then(pl.lit(""))
            .otherwise(pl.col(column_name))
            .alias(column_name)
        )

    if always_show_sign:
        df = df.with_columns(
            pl.concat_str("_sign", column_name).alias(column_name)
        ).drop("_sign")
    return df


def unwrap_single_values(func) -> pl.DataFrame | Any:
    """Returns the DataFrame unless it is a single value in which case return the value."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pl.DataFrame) and result.shape == (1, 1):
            return result.item()
        return result

    return wrapper


@dataclass
class FuncArgs:
    args: list
    kwargs: dict = field(default_factory=dict)


class Container:
    """A placeholder object that makes it easy to set and get attributes. Used in Model.attr and Model.params, for example.

    Examples:
        >>> x = {}
        >>> params = Container(
        ...     setter=lambda n, v: x.__setitem__(n, v), getter=lambda n: x[n]
        ... )
        >>> params.a = 1
        >>> params.b = 2
        >>> params.a
        1
        >>> params.b
        2
    """

    def __init__(self, setter, getter):
        self._setter = setter
        self._getter = getter

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):  # pragma: no cover
            return super().__setattr__(name, value)
        self._setter(name, value)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):  # pragma: no cover
            return super().__getattribute__(name)
        return self._getter(name)


class NamedVariableMapper:
    """Maps variables to a string representation using the object's name and dimensions.

    Examples:
        >>> import polars as pl
        >>> m = pf.Model()
        >>> m.foo = pf.Variable(pl.DataFrame({"t": range(4)}))
        >>> m.foo.sum()
        <Expression (linear) terms=4>
        foo[0] + foo[1] + foo[2] + foo[3]
    """

    CONST_TERM_NAME = "_ONE"
    NAME_COL = "__name"

    def __init__(self) -> None:
        self._ID_COL = VAR_KEY
        self.mapping_registry = pl.DataFrame(
            {self._ID_COL: [], self.NAME_COL: []},
            schema={self._ID_COL: Config.id_dtype, self.NAME_COL: pl.String},
        )
        self._extend_registry(
            pl.DataFrame(
                {self._ID_COL: [CONST_TERM], self.NAME_COL: [self.CONST_TERM_NAME]},
                schema={self._ID_COL: Config.id_dtype, self.NAME_COL: pl.String},
            )
        )

    def add(self, element: Variable) -> None:
        self._extend_registry(self._element_to_map(element))

    def _extend_registry(self, df: pl.DataFrame) -> None:
        self.mapping_registry = pl.concat([self.mapping_registry, df])

    def apply(
        self,
        df: pl.DataFrame,
        to_col: str,
        id_col: str,
    ) -> pl.DataFrame:
        return df.join(
            self.mapping_registry,
            how="left",
            validate="m:1",
            left_on=id_col,
            right_on=self._ID_COL,
            maintain_order="left" if Config.maintain_order else None,
        ).rename({self.NAME_COL: to_col})

    def _element_to_map(self, element: Variable) -> pl.DataFrame:
        element_name = element.name  # type: ignore
        assert element_name is not None, (
            "Element must have a name to be used in a named mapping."
        )
        element._assert_has_ids()
        return concat_dimensions(
            element.data.select(element._dimensions_unsafe + [VAR_KEY]),
            keep_dims=False,
            prefix=element_name,
            to_col=self.NAME_COL,
        )


def for_solvers(*solvers: str):
    """Limits the decorated function to only be available when the solver is in the `solvers` list."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.solver.name not in solvers:
                raise NotImplementedError(
                    f"Method '{func.__name__}' is not implemented for solver '{self.solver}'."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


# TODO: rename and change to return_expr once Set is split away from BaseOperableBlock
def return_new(func: Callable[..., pl.DataFrame]) -> Callable[..., BaseOperableBlock]:
    """Decorator that upcasts the returned DataFrame to an Expression.

    Requires the first argument (self) to support self._new().
    """

    @wraps(func)
    def wrapper(self: BaseOperableBlock, *args, **kwargs):
        result = func(self, *args, **kwargs)
        return self._new(result, name=f"{self.name}.{func.__name__}(…)")

    return wrapper
