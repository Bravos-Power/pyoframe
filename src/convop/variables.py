from typing import List
import polars as pl
from convop.expressionable import Expressionable

from convop.expressions import (
    COEF_KEY,
    CONST_KEY,
    VAR_KEY,
    Expression,
    _get_dimensions,
)
from convop.model_element import ModelElement
from convop.parameters import Parameter


class Variable(Expressionable, ModelElement):
    _var_count = 0

    def __init__(
        self,
        df: pl.DataFrame | None | Parameter = None,
        lb: None | float = None,
        ub: None | float = None,
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
        super().__init__()
        if isinstance(df, Parameter):
            df = df.data.drop(CONST_KEY)

        if df is None:
            self.data = pl.DataFrame({VAR_KEY: [Variable._var_count]})
        else:
            self.data = df.with_columns(
                pl.int_range(
                    Variable._var_count,
                    Variable._var_count + pl.len(),
                    dtype=pl.UInt32,
                ).alias(VAR_KEY)
            )
        Variable._var_count += self.data.height
        self.lb = lb
        self.ub = ub

    def __repr__(self):
        return f"""
        Variables: {self.name} | Lower Bound: {self.lb} | Upper Bound: {self.ub}
        {self.data}
        """

    def to_expression(self) -> Expression:
        return Expression(variables=self.data.with_columns(pl.lit(1.0).alias(COEF_KEY)))
    
    @property
    def dimensions(self) -> List[str]:
        return _get_dimensions(self.data)
