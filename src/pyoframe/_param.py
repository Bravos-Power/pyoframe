import pandas as pd
import polars as pl

from pyoframe._constants import COEF_KEY, CONST_TERM, VAR_KEY
from pyoframe._core import Expression


def Param(data: pl.DataFrame | pd.DataFrame | pd.Series | dict) -> Expression:
    """Creates a model parameter, i.e. an [Expression][pyoframe.Expression] that doesn't involve any variables.

    !!! note
        Technically, `Param(data)` is a function that returns an [Expression][pyoframe.Expression], not a class.
        However, for consistency with other modeling frameworks, we provide it as a class-like function (i.e. an uppercase function).

    !!! tip "Smart naming"
        If a Param is not given a name (i.e. it is not assigned to a model: `m.my_name = Param(...)`),
        then its [name][pyoframe._model_element.BaseBlock.name] is inferred from the name of the column in `data` that contains the parameter values.
        This makes debugging models with inline parameters easier.

    Args:
        data: The data for the parameter.

            If `data` is a `pandas.DataFrame` or `polars.DataFrame`, the last column in the dataframe will be used for values and all previous columns for labels.

            If `data` is a `pandas.Series`, the index(es) will be used for labels (and the values for values).

            If `data` is of any other type (e.g. a dictionary), it will be used as if you had called `Param(pl.DataFrame(data))`.

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
    if not isinstance(data, pl.DataFrame):
        data = pl.DataFrame(data)

    name = data.columns[-1]
    return Expression(
        data.rename({name: COEF_KEY})
        .drop_nulls(COEF_KEY)
        .with_columns(pl.lit(CONST_TERM).alias(VAR_KEY)),
        name=name,
    )
