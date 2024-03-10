import polars as pl
from convop.expressionable import Expressionable

from convop.expressions import CONST_KEY, Expression


class Parameter(Expressionable):
    def __init__(
        self,
        df: pl.DataFrame,
        index_col_names: list[str],
        param_col_name: str,
        name: str | None = None,
    ):
        self.data = df
        self.index_col_names = index_col_names
        self.param_col_name = param_col_name
        self.name = name

    def _getParamData(self):
        return self.data.select(
            [pl.col(index) for index in self.index_col_names]
            + [pl.col(self.param_col_name).alias(CONST_KEY)]
        )

    def to_expression(self):
        return Expression(constants=self._getParamData())

    def __repr__(self):
        return f"""
        Parameters: {self.name}
        {self._getParamData()}
        """
