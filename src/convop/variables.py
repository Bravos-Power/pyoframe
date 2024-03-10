import polars as pl
from convop.expressionable import Expressionable

from convop.expressions import (
    COEF_KEY,
    VAR_KEY,
    Expression,
)
from convop.parameters import Parameter


# class VariableMapping:
#     VAR_NAME_KEY = "_var_name"

#     def __init__(self) -> None:
#         self.map = pl.DataFrame({VARIABLES_KEY: [], self.VAR_NAME_KEY: []})

#     def create_var_block(self, df: pl.DataFrame, name: str):
#         last_value = self.map[VARIABLES_KEY].max()
#         assert isinstance(last_value, int)

#         df = df.with_columns(
#             pl.concat_str(pl.lit(name), *df.columns, separator="_").alias(
#                 self.VAR_NAME_KEY
#             ),
#             VARIABLES_KEY=pl.int_range(
#                 last_value + 1, last_value + 1 + pl.len(), dtype=pl.UInt32
#             ),
#         )
#         self.map = df.select(VARIABLES_KEY, self.VAR_NAME_KEY)
#         return df.drop(self.VAR_NAME_KEY)


# variable_mapping = VariableMapping()


class Variable(Expressionable):
    _var_count = 0

    def __init__(
        self,
        df: pl.DataFrame | None | Parameter = None,
        lb: None | float = None,
        ub: None | float = None,
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
        if isinstance(df, Parameter):
            df = df.data.drop(df.param_col_name)

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
        self.name = name

    def __repr__(self):
        return f"""
        Variables: {self.name} | Lower Bound: {self.lb} | Upper Bound: {self.ub}
        {self.data}
        """

    def to_expression(self) -> Expression:
        return Expression(variables=self.data.with_columns(pl.lit(1.0).alias(COEF_KEY)))
