import polars as pl
from convop.constraints import Expressionable

from convop.model_element import RESERVED_COL_KEYS, VAR_KEY
from convop.constraints import Expression
from convop.model_element import VAR_KEY, FrameWrapper, ModelElement
import pandas as pd


class Variable(FrameWrapper, Expressionable, ModelElement):
    _var_count = 1  # Must start at 1 since 0 is reserved for constant terms

    @classmethod
    def _reset_count(cls):
        cls._var_count = 1

    def __init__(
        self,
        df: pl.DataFrame | None | pd.Index | pd.DataFrame | Expression = None,
        lb: float = float("-inf"),
        ub: float = float("inf"),
    ):
        """Creates a variable for every row in the dataframe.

        Parameters
        ----------
        df: pl.DataFrame
            The dataframe over which variables should be created.
        lb: float
            The lower bound for all variables.
        ub: float
            The upper bound for all variables.
        dim: int, optional
            The number of starting columns to be used as an index. If None, and df is a Parameters object, the dimension is inferred from the Parameters object. Otherwise, all columns are used as an index.
        name: str, optional
            The name of the variable. If using ModelBuilder this is automatically set to match your variable name.

        Examples
        --------
        >>> import pandas as pd
        >>> from convop import Variable
        >>> df = pd.DataFrame({"dim1": [1, 1, 2, 2, 3, 3], "dim2": ["a", "b", "a", "b", "a", "b"]})
        >>> Variable(df)
        <Variable name=unnamed lb=-inf ub=inf size=6 dimensions={'dim1': 3, 'dim2': 2}>
        >>> Variable(df[["dim1"]])
        Traceback (most recent call last):
        ...
        ValueError: Duplicate rows found in data.
        >>> Variable(df[["dim1"]].drop_duplicates())
        <Variable name=unnamed lb=-inf ub=inf size=3 dimensions={'dim1': 3}>
        """
        if isinstance(df, pd.Index):
            df = pl.from_pandas(pd.DataFrame(index=df).reset_index())
        elif isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df)
        elif isinstance(df, Expression):
            df = df.data.drop(RESERVED_COL_KEYS).unique()

        if df is None:
            data = pl.DataFrame({VAR_KEY: [Variable._var_count]})
        else:
            if df.is_duplicated().any():
                raise ValueError("Duplicate rows found in data.")
            data = df.with_columns(
                pl.int_range(
                    Variable._var_count,
                    Variable._var_count + pl.len(),
                    dtype=pl.UInt32,
                ).alias(VAR_KEY)
            )
        Variable._var_count += data.height
        self.lb = lb
        self.ub = ub
        super().__init__(data)

    def __repr__(self):
        return f"""<Variable name={self.name} lb={self.lb} ub={self.ub} size={self.data.height} dimensions={self.shape}>"""

    def to_expr(self) -> Expression:
        return Expression(self.data)
