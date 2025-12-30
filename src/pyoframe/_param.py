"""Defines the function for creating model parameters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from pyoframe._constants import COEF_KEY, CONST_TERM, VAR_KEY
from pyoframe._core import Expression


def Param(
    data: pl.DataFrame | pd.DataFrame | pd.Series | dict | str | Path,
) -> Expression:
    """Creates a model parameter, i.e. an [Expression][pyoframe.Expression] that doesn't involve any variables.

    A Parameter can be created from a DataFrame, CSV file, Parquet file, data dictionary, or a Pandas Series.

    !!! info "`Param` is a function, not a class"
        Technically, `Param(data)` is a function that returns an [Expression][pyoframe.Expression], not a class.
        However, for consistency with other modeling frameworks, we provide it as a class-like function (i.e. an uppercase function).

    !!! tip "Smart naming"
        If a Param is not given a name (i.e. if it is not assigned to a model: `m.my_name = Param(...)`),
        then its [name][pyoframe._model_element.BaseBlock.name] is inferred from the name of the column in `data` that contains the parameter values.
        This makes debugging models with inline parameters easier.

    Args:
        data: The data to use for the parameter.

            If `data` is a polars or pandas `DataFrame`, the last column will be treated as the values of the parameter, and all other columns as labels.

            If `data` is a string or `Path`, it will be interpreted as a path to a CSV or Parquet file that will be read and used as a `DataFrame`. The file extension must be `.csv` or `.parquet`.

            If `data` is a `pandas.Series`, the index(es) will be treated as columns for labels and the series values as the parameter values.

            If `data` is of any other type (e.g. a dictionary), it will be used as if you had called `Param(pl.DataFrame(data))`.

    Returns:
        An Expression representing the parameter.

    Examples:
        >>> m = pf.Model()
        >>> m.fixed_cost = pf.Param({"plant": ["A", "B"], "cost": [1000, 1500]})
        >>> m.fixed_cost
        <Expression (parameter) height=2 terms=2>
        ┌───────┬────────────┐
        │ plant ┆ expression │
        │ (2)   ┆            │
        ╞═══════╪════════════╡
        │ A     ┆ 1000       │
        │ B     ┆ 1500       │
        └───────┴────────────┘

        Since `Param` simply returns an Expression, you can use it in building larger expressions as usual:

        >>> m.variable_cost = pf.Param(
        ...     pl.DataFrame({"plant": ["A", "B"], "cost": [50, 60]})
        ... )
        >>> m.total_cost = m.fixed_cost + m.variable_cost
        >>> m.total_cost
        <Expression (parameter) height=2 terms=2>
        ┌───────┬────────────┐
        │ plant ┆ expression │
        │ (2)   ┆            │
        ╞═══════╪════════════╡
        │ A     ┆ 1050       │
        │ B     ┆ 1560       │
        └───────┴────────────┘
    """
    if isinstance(data, pd.Series):
        data = data.to_frame().reset_index()
    if isinstance(data, pd.DataFrame):
        data = pl.from_pandas(data)

    if isinstance(data, (str, Path)):
        data = Path(data)
        if data.suffix.lower() == ".csv":
            data = pl.read_csv(data)
        elif data.suffix.lower() in {".parquet"}:
            data = pl.read_parquet(data)
        else:
            raise NotImplementedError(
                f"Could not create parameter. Unsupported file format: {data.suffix}"
            )

    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)

    value_col = data.columns[-1]

    return Expression(
        data.rename({value_col: COEF_KEY})
        .drop_nulls(COEF_KEY)
        .with_columns(pl.lit(CONST_TERM).alias(VAR_KEY)),
        name=value_col,
    )
