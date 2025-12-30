"""Defines the function for creating model parameters."""

from pathlib import Path

import pandas as pd
import polars as pl

from pyoframe._constants import COEF_KEY, CONST_TERM, VAR_KEY
from pyoframe._core import Expression


def Param(
    data: pl.DataFrame | pd.DataFrame | pd.Series | dict | str | Path,
    label_cols: list[str] | None = None,
    value_col: str | None = None,
) -> Expression:
    """Creates a model parameter, i.e. an [Expression][pyoframe.Expression] that doesn't involve any variables.

    Parameters can be created from DataFrames, CSV or Parquet files, dictionaries, or Pandas Series.

    !!! note
        Technically, `Param(data)` is a function that returns an [Expression][pyoframe.Expression], not a class.
        However, for consistency with other modeling frameworks, we provide it as a class-like function (i.e. an uppercase function).

    !!! tip "Smart naming"
        If a Param is not given a name (i.e. it is not assigned to a model: `m.my_name = Param(...)`),
        then its [name][pyoframe._model_element.BaseBlock.name] is inferred from the name of the column in `data` that contains the parameter values.
        This makes debugging models with inline parameters easier.

    Args:
        data: The dataframe containing the parameter labels and values. If `data` is a string or `Path`, it will be interpreted as a path to a CSV or Parquet file and loaded accordingly.
            If `data` is a `pandas.Series`, the index(es) will be treated as columns for labels. If `data` is of any other type (e.g. a dictionary), it will be used as if you had called `Param(pl.DataFrame(data))`.

        label_cols: By default, all columns except the last one are used as labels (i.e. dimensions) for the parameter. You can override this by specifying a list of column names to use as labels.
        value_col: By default, the last column is used as the values for the parameter. You can override this by specifying the name of the column to use as values.

    Returns:
        An Expression representing the parameter.

    Examples:
        >>> m = pf.Model()
        >>> m.fixed_cost = Param({"plant": ["A", "B"], "cost": [1000, 1500]})
        >>> m.fixed_cost
        <Expression (parameter) height=2 terms=2>
        ┌───────┬────────────┐
        │ plant ┆ expression │
        │ (2)   ┆            │
        ╞═══════╪════════════╡
        │ A     ┆ 1000       │
        │ B     ┆ 1500       │
        └───────┴────────────┘

        Since `Param` simply returns an Expression, you can use it in operations as usual:

        >>> m.variable_cost = Param(
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
        columns = None
        if value_col is not None and label_cols is not None:
            columns = label_cols + [value_col]

        data = Path(data)
        if data.suffix.lower() == ".csv":
            data = pl.read_csv(data, columns=columns)
        elif data.suffix.lower() in {".parquet"}:
            data = pl.read_parquet(data, columns=columns)
        else:
            raise NotImplementedError(f"Unsupported file format: {data.suffix}")

    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)

    if value_col is None:
        value_col = data.columns[-1]
    if label_cols is None:
        label_cols = [col for col in data.columns if col != value_col]
    assert isinstance(label_cols, list), "labels must be a list of column names"
    data = data.select(label_cols + [value_col])

    return Expression(
        data.rename({value_col: COEF_KEY})
        .drop_nulls(COEF_KEY)
        .with_columns(pl.lit(CONST_TERM).alias(VAR_KEY)),
        name=value_col,
    )
