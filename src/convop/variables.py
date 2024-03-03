from typing import Iterable
import polars as pl

from convop.expressions import (
    COEFFICIENTS_KEY,
    VARIABLES_KEY,
    Expression,
    Expressionable,
)
from convop.parameters import Parameters

LABELS_DTYPE = pl.UInt32


class Variables(Expressionable):
    def __init__(
        self,
        lb: None | float,
        ub: None | float,
        data: pl.DataFrame,
        name: str | None = None,
    ):
        self.lb = lb
        self.ub = ub
        self.data = data
        self.name = name

    def __repr__(self):
        return f"""
        Variables: {self.name} | Lower Bound: {self.lb} | Upper Bound: {self.ub}
        {self.data}
        """

    def toExpression(self) -> Expression:
        return Expression(
            constants=None,
            variables=self.data.with_columns(pl.lit(1.0).alias(COEFFICIENTS_KEY)),
        )


def add_variables(
    model,
    df: pl.DataFrame | Parameters,
    lb=None,
    ub=None,
    dim: int | None = None,
    name: str | None = None,
):
    """Creates a variable for every row in the dataframe.

    Parameters
    ----------
    df: pl.DataFrame
        The dataframe over which variables should be created.
    lb: float, optional
        The lower bound for all variables.
    ub: float, optional
        The upper bound for all variables.
    dim: int, optional
        The number of starting columns to be used as an index. If None, and df is a Parameters object, the dimension is inferred from the Parameters object. Otherwise, all columns are used as an index.
    name: str, optional
        The name of the variable. If using ModelBuilder this is automatically set to match your variable name.
    """
    if isinstance(df, Parameters):
        if dim is None:
            dim = len(df.index_col_names)
        df = df.data

    if dim is not None:
        df = df.select([pl.col(name) for name in df.columns[:dim]])

    data = df.with_columns(
        pl.int_range(pl.len(), dtype=LABELS_DTYPE).alias(VARIABLES_KEY)
    )
    variables = Variables(lb=lb, ub=ub, data=data)
    model.variables.append(variables)
    return variables
