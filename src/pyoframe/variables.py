import polars as pl
import pandas as pd
from typing import Iterable, List

from pyoframe.constraints import Expressionable

from pyoframe.dataframe import RESERVED_COL_KEYS, VAR_KEY
from pyoframe.constraints import Expression
from pyoframe.model_element import FrameWrapper, ModelElement
from pyoframe.util import _parse_inputs_as_iterable


VariableSet = pl.DataFrame | pd.Index | pd.DataFrame | Expression


class Variable(FrameWrapper, Expressionable, ModelElement):
    _var_count = 1  # Must start at 1 since 0 is reserved for constant terms

    @classmethod
    def _reset_count(cls):
        cls._var_count = 1

    def __init__(
        self,
        *over: VariableSet | Iterable[VariableSet],
        lb: float = float("-inf"),
        ub: float = float("inf"),
    ):
        """Creates a variable for every row in the dataframe.

        Parameters
        ----------
        over: pl.DataFrame
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
        >>> from pyoframe import Variable
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
        over: pl.DataFrame | None = Variable._parse_over(*over)
        if over is None:
            data = pl.DataFrame(
                {VAR_KEY: [Variable._var_count]}, schema={VAR_KEY: pl.UInt32}
            )
        else:
            if over.is_duplicated().any():
                raise ValueError("Duplicate rows found in data.")
            data = over.with_columns(
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
        return Expression(self.data, model=self._model)

    @staticmethod
    def _parse_over(
        *over: VariableSet | Iterable[VariableSet],
    ) -> pl.DataFrame | None:
        """
        >>> import pandas as pd
        >>> dim1 = pd.Index([1, 2, 3])
        >>> dim2 = pd.Index(["a", "b"])
        >>> Variable._parse_over([dim1, dim2])
        Traceback (most recent call last):
        ...
        AssertionError: All coordinates must have unique column names.
        >>> dim1.name = "dim1"
        >>> dim2.name = "dim2"
        >>> Variable._parse_over([dim1, dim2])
        shape: (6, 2)
        ┌──────┬──────┐
        │ dim1 ┆ dim2 │
        │ ---  ┆ ---  │
        │ i64  ┆ str  │
        ╞══════╪══════╡
        │ 1    ┆ a    │
        │ 1    ┆ b    │
        │ 2    ┆ a    │
        │ 2    ┆ b    │
        │ 3    ┆ a    │
        │ 3    ┆ b    │
        └──────┴──────┘
        """
        if len(over) == 0:
            return None

        over: Iterable[VariableSet] = _parse_inputs_as_iterable(over)

        def _parse_input(input: VariableSet) -> pl.DataFrame:
            if isinstance(input, pd.Index):
                input = pd.DataFrame(index=input).reset_index()
            if isinstance(input, pd.DataFrame):
                input = pl.from_pandas(input)
            if isinstance(input, Expression):
                input = input.data.drop(RESERVED_COL_KEYS).unique()
            return input

        over: List[pl.DataFrame] = [_parse_input(input) for input in over]

        over_merged = over[0]

        for df in over[1:]:
            assert (
                set(over_merged.columns) & set(df.columns) == set()
            ), "All coordinates must have unique column names."
            over_merged = over_merged.join(df, how="cross")
        return over_merged
