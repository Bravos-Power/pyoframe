import polars as pl
from pyoframe.constraints import Expressionable

from pyoframe.model_element import RESERVED_COL_KEYS, VAR_KEY
from pyoframe.constraints import Expression
from pyoframe.model_element import VAR_KEY, FrameWrapper, ModelElement
import pandas as pd
from typing import List


class Variable(FrameWrapper, Expressionable, ModelElement):
    _var_count = 1  # Must start at 1 since 0 is reserved for constant terms

    @classmethod
    def _reset_count(cls):
        cls._var_count = 1

    def __init__(
        self,
        over: (
            pl.DataFrame
            | None
            | pd.Index
            | pd.DataFrame
            | Expression
            | List[pd.Index | pd.DataFrame | pl.DataFrame]
        ) = None,
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
        if isinstance(over, Expression):
            over = over.data.drop(RESERVED_COL_KEYS).unique()

        if over is None:
            data = pl.DataFrame(
                {VAR_KEY: [Variable._var_count]}, schema={VAR_KEY: pl.UInt32}
            )
        else:
            over = Variable._coords_to_df(over)
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
    def _coords_to_df(
        coords: (
            pl.DataFrame
            | pd.DataFrame
            | pd.Index
            | List[pl.DataFrame | pd.DataFrame | pd.Index]
        ),
    ):
        """
        >>> import pandas as pd
        >>> dim1 = pd.Index([1, 2, 3])
        >>> dim2 = pd.Index(["a", "b"])
        >>> Variable._coords_to_df([dim1, dim2])
        Traceback (most recent call last):
        ...
        AssertionError: All coordinates must have unique column names.
        >>> dim1.name = "dim1"
        >>> dim2.name = "dim2"
        >>> Variable._coords_to_df([dim1, dim2])
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
        if not isinstance(coords, list):
            coords = [coords]

        coord_pl: List[pl.DataFrame] = []

        for coord in coords:
            if isinstance(coord, pd.Index):
                coord = pd.DataFrame(index=coord).reset_index()
            if isinstance(coord, pd.DataFrame):
                coord = pl.from_pandas(coord)
            coord_pl.append(coord)

        if len(coord_pl) == 1:
            return coord_pl[0]

        df = coord_pl[0]
        for coord in coord_pl[1:]:
            assert (
                set(coord.columns) & set(df.columns) == set()
            ), "All coordinates must have unique column names."
            df = df.join(coord, how="cross")
        return df