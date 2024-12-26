"""
File containing utility functions and classes.
"""

from dataclasses import dataclass, field
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

import pandas as pd
import polars as pl

from pyoframe.constants import COEF_KEY, CONST_TERM, RESERVED_COL_KEYS, VAR_KEY, Config

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Variable
    from pyoframe.model_element import ModelElementWithId


def get_obj_repr(obj: object, _props: Iterable[str] = (), **kwargs):
    """
    Helper function to generate __repr__ strings for classes. See usage for examples.
    """
    props = {prop: getattr(obj, prop) for prop in _props}
    props_str = " ".join(f"{k}={v}" for k, v in props.items() if v is not None)
    if props_str:
        props_str += " "
    kwargs_str = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
    return f"<{obj.__class__.__name__} {props_str}{kwargs_str}>"


def parse_inputs_as_iterable(
    *inputs: Union[Any, Iterable[Any]],
) -> Iterable[Any]:
    """
    Converts a parameter *x: Any | Iteraable[Any] to a single Iterable[Any] object.
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


def _is_iterable(input: Union[Any, Iterable[Any]]) -> bool:
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
    df: pl.DataFrame,
    prefix: Optional[str] = None,
    keep_dims: bool = True,
    ignore_columns: Sequence[str] = RESERVED_COL_KEYS,
    replace_spaces: bool = True,
    to_col: str = "concated_dim",
) -> pl.DataFrame:
    """
    Returns a new DataFrame with the column 'concated_dim'. Reserved columns are ignored.

    Parameters:
        df:
            The input DataFrame.
        prefix:
            The prefix to be added to the concated dimension.
        keep_dims:
            If True, the original dimensions are kept in the new DataFrame.
        replace_spaces : bool, optional
            If True, replaces spaces with underscores.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame(
        ...     {
        ...         "dim1": [1, 2, 3, 1, 2, 3],
        ...         "dim2": ["Y", "Y", "Y", "N", "N", "N"],
        ...     }
        ... )
        >>> concat_dimensions(df)
        shape: (6, 3)
        ┌──────┬──────┬──────────────┐
        │ dim1 ┆ dim2 ┆ concated_dim │
        │ ---  ┆ ---  ┆ ---          │
        │ i64  ┆ str  ┆ str          │
        ╞══════╪══════╪══════════════╡
        │ 1    ┆ Y    ┆ [1,Y]        │
        │ 2    ┆ Y    ┆ [2,Y]        │
        │ 3    ┆ Y    ┆ [3,Y]        │
        │ 1    ┆ N    ┆ [1,N]        │
        │ 2    ┆ N    ┆ [2,N]        │
        │ 3    ┆ N    ┆ [3,N]        │
        └──────┴──────┴──────────────┘
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
        >>> concat_dimensions(df, keep_dims=False)
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
        >>> # Properly handles cases with no dimensions and ignores reserved columns
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
    dimensions = [col for col in df.columns if col not in ignore_columns]
    if dimensions:
        query = pl.concat_str(
            pl.lit(prefix + "["),
            pl.concat_str(*dimensions, separator=","),
            pl.lit("]"),
        )
    else:
        query = pl.lit(prefix)

    df = df.with_columns(query.alias(to_col))

    if replace_spaces:
        df = df.with_columns(pl.col(to_col).str.replace_all(" ", "_"))

    if not keep_dims:
        df = df.drop(*dimensions)

    return df


def cast_coef_to_string(
    df: pl.DataFrame, column_name: str = COEF_KEY, drop_ones: bool = True
) -> pl.DataFrame:
    """
    Converts column `column_name` of the dataframe `df` to a string. Rounds to `Config.print_float_precision` decimal places if not None.

    Parameters:
        df:
            The input DataFrame.
        column_name:
            The name of the column to be casted.
        drop_ones:
            If True, 1s are replaced with an empty string for non-constant terms.

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
    df = df.with_columns(
        pl.col(column_name).abs(),
        _sign=pl.when(pl.col(column_name) < 0).then(pl.lit("-")).otherwise(pl.lit("+")),
    )

    if Config.float_to_str_precision is not None:
        df = df.with_columns(pl.col(column_name).round(Config.float_to_str_precision))

    df = df.with_columns(
        pl.when(pl.col(column_name) == pl.col(column_name).floor())
        .then(pl.col(column_name).cast(pl.Int64).cast(pl.String))
        .otherwise(pl.col(column_name).cast(pl.String))
        .alias(column_name)
    )

    if drop_ones:
        condition = pl.col(column_name) == str(1)
        if VAR_KEY in df.columns:
            condition = condition & (pl.col(VAR_KEY) != CONST_TERM)
        df = df.with_columns(
            pl.when(condition)
            .then(pl.lit(""))
            .otherwise(pl.col(column_name))
            .alias(column_name)
        )
    else:
        df = df.with_columns(pl.col(column_name).cast(pl.Utf8))
    return df.with_columns(pl.concat_str("_sign", column_name).alias(column_name)).drop(
        "_sign"
    )


def unwrap_single_values(func):
    """Decorator for functions that return DataFrames. Returned dataframes with a single value will instead return the value."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, pl.DataFrame) and result.shape == (1, 1):
            return result.item()
        return result

    return wrapper


def dataframe_to_tupled_list(
    df: pl.DataFrame, num_max_elements: Optional[int] = None
) -> str:
    """
    Converts a dataframe into a list of tuples. Used to print a Set to the console. See examples for behaviour.

    Examples:
        >>> df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
        >>> dataframe_to_tupled_list(df)
        '[1, 2, 3, 4, 5]'
        >>> dataframe_to_tupled_list(df, 3)
        '[1, 2, 3, ...]'

        >>> df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 3, 4, 5, 6]})
        >>> dataframe_to_tupled_list(df, 3)
        '[(1, 2), (2, 3), (3, 4), ...]'
    """
    elipse = False
    if num_max_elements is not None:
        if len(df) > num_max_elements:
            elipse = True
            df = df.head(num_max_elements)

    res = (row for row in df.iter_rows())
    if len(df.columns) == 1:
        res = (row[0] for row in res)

    res = str(list(res))
    if elipse:
        res = res[:-1] + ", ...]"
    return res


@dataclass
class FuncArgs:
    args: List
    kwargs: Dict = field(default_factory=dict)


class Container:
    """
    A placeholder object that makes it easy to set and get attributes. Used in Model.attr and Model.params, for example.

    Examples:
        >>> x = {}
        >>> params = Container(setter=lambda n, v: x.__setitem__(n, v), getter=lambda n: x[n])
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
    """
    Maps variables to a string representation using the object's name and dimensions.

    Examples:
        >>> import polars as pl
        >>> m = pf.Model()
        >>> m.foo = pf.Variable(pl.DataFrame({"t": range(4)}))
        >>> pf.sum(m.foo)
        <Expression size=1 dimensions={} terms=4>
        foo[0] + foo[1] + foo[2] + foo[3]
    """

    CONST_TERM_NAME = "_ONE"
    NAME_COL = "__name"

    def __init__(self, cls: Type["ModelElementWithId"]) -> None:
        self._ID_COL = VAR_KEY
        self.mapping_registry = pl.DataFrame(
            {self._ID_COL: [], self.NAME_COL: []},
            schema={self._ID_COL: pl.UInt32, self.NAME_COL: pl.String},
        )
        self._extend_registry(
            pl.DataFrame(
                {self._ID_COL: [CONST_TERM], self.NAME_COL: [self.CONST_TERM_NAME]},
                schema={self._ID_COL: pl.UInt32, self.NAME_COL: pl.String},
            )
        )

    def add(self, element: "Variable") -> None:
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
        ).rename({self.NAME_COL: to_col})

    def _element_to_map(self, element) -> pl.DataFrame:
        element_name = element.name  # type: ignore
        assert (
            element_name is not None
        ), "Element must have a name to be used in a named mapping."
        element._assert_has_ids()
        return concat_dimensions(
            element.data.select(element.dimensions_unsafe + [VAR_KEY]),
            keep_dims=False,
            prefix=element_name,
            to_col=self.NAME_COL,
        )
