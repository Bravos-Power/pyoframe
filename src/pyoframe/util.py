"""
File containing utility functions and classes.
"""

from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Union

import polars as pl
import pandas as pd

from pyoframe.constants import COEF_KEY, CONST_TERM, RESERVED_COL_KEYS, VAR_KEY


class IdCounterMixin:
    """
    Provides a method that assigns a unique ID to each row in a DataFrame.
    IDs start at 1 and go up consecutively. No zero ID is assigned since it is reserved for the constant variable term.
    IDs are only unique for the subclass since different subclasses have different counters.
    """

    # Keys are the subclass names and values are the next unasigned ID.
    _id_counters: Dict[str, int] = defaultdict(lambda: 1)

    @classmethod
    def _reset_counters(cls):
        """
        Resets all the ID counters.
        This function is called before every unit test to reset the code state.
        """
        cls._id_counters = defaultdict(lambda: 1)

    def _assign_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Adds the column `to_column` to the DataFrame `df` with the next batch
        of unique consecutive IDs.
        """
        assert df.height > 0, "Cannot assign IDs to an empty DataFrame."

        cls_name = self.__class__.__name__
        cur_count = self._id_counters[cls_name]
        df = df.with_columns(
            pl.int_range(cur_count, cur_count + pl.len())
            .alias(self.get_id_column_name())
            .cast(pl.UInt32)
        )
        self._id_counters[cls_name] += df.height
        return df

    @classmethod
    @abstractmethod
    def get_id_column_name(cls) -> str:
        """
        Returns the name of the column containing the IDs.
        """

    @property
    @abstractmethod
    def ids(self) -> pl.DataFrame:
        """
        Returns a dataframe with the IDs and any other relevant columns (i.e. the dimension columns).
        """


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
    # Inspired from the polars library
    return isinstance(input, Iterable) and not isinstance(
        input,
        (str, bytes, pl.DataFrame, pl.Series, pd.DataFrame, pd.Series, pd.Index, dict),
    )


def concat_dimensions(
    df: pl.DataFrame,
    prefix: Optional[str] = None,
    keep_dims: bool = True,
    ignore_columns=RESERVED_COL_KEYS,
    replace_spaces: bool = True,
    to_col: str = "concated_dim",
) -> pl.DataFrame:
    """
    Returns a new DataFrame with the column 'concated_dim'. Reserved columns are ignored.

    Parameters:
        df : pl.DataFrame
            The input DataFrame.
        prefix : str, optional
            The prefix to be added to the concated dimension.
        keep_dims : bool, optional
            If True, the original dimensions are kept in the new DataFrame.

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
    df: pl.DataFrame, column_name: str = COEF_KEY, drop_ones=True, float_precision=None
) -> pl.DataFrame:
    """
    Parameters:
        df : pl.DataFrame
            The input DataFrame.
        column_name : str, optional
            The name of the column to be casted.
        drop_ones : bool, optional
            If True, 1s are replaced with an empty string for non-constant terms.
        float_precision : int, optional
            The number of decimal places to round the coefficients to. If None, no rounding is done (so Polars' default precision is used).
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

    if float_precision is not None:
        df = df.with_columns(pl.col(column_name).round(float_precision))

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
