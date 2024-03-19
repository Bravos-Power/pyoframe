from __future__ import annotations
import polars as pl
import pandas as pd
from typing import Iterable, List, Literal

from pyoframe.constraints import Expressionable

from pyoframe.dataframe import COEF_KEY, VAR_KEY
from pyoframe.constraints import Expression, _set_to_polars, Set
from pyoframe.model_element import ModelElement
from pyoframe.util import _parse_inputs_as_iterable


class Variable(ModelElement, Expressionable):
    _var_count = 1  # Must start at 1 since 0 is reserved for constant terms

    @classmethod
    def _reset_count(cls):
        cls._var_count = 1

    def __init__(
        self,
        *sets: Set | Iterable[Set],
        lb: float = float("-inf"),
        ub: float = float("inf"),
        vtype: Literal["continuous", "binary", "integer"] = "continuous",
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
        <Variable lb=-inf ub=inf size=6 dimensions={'dim1': 3, 'dim2': 2}>
        [1,a]: x1
        [1,b]: x2
        [2,a]: x3
        [2,b]: x4
        [3,a]: x5
        [3,b]: x6
        >>> Variable(df[["dim1"]])
        Traceback (most recent call last):
        ...
        ValueError: Duplicate rows found in data.
        >>> Variable(df[["dim1"]].drop_duplicates())
        <Variable lb=-inf ub=inf size=3 dimensions={'dim1': 3}>
        [1]: x7
        [2]: x8
        [3]: x9
        """
        indexing_set: pl.DataFrame | None = Variable._parse_set(*sets)
        if indexing_set is None:
            data = pl.DataFrame({VAR_KEY: [Variable._var_count]})
        else:
            if indexing_set.is_duplicated().any():
                raise ValueError("Duplicate rows found in data.")
            data = indexing_set.with_columns(
                pl.int_range(Variable._var_count, Variable._var_count + pl.len()).alias(
                    VAR_KEY
                )
            )

        super().__init__(data)
        Variable._var_count += data.height
        self.lb = lb
        self.ub = ub
        assert vtype in (
            "continuous",
            "binary",
            "integer",
        ), "type must be one of 'continuous', 'binary', or 'integer'"
        self._vtype: Literal["continuous", "binary", "integer"] = vtype

    @property
    def vtype(self) -> Literal["continuous", "binary", "integer"]:
        return self._vtype

    def __repr__(self):
        return f"""<Variable{' name='+self.name if self.name is not None else ''} lb={self.lb} ub={self.ub} size={self.data.height} dimensions={self.shape}>\n{self.to_expr().to_str(max_line_len=80, max_rows=10)}"""

    def to_expr(self) -> Expression:
        return Expression(
            self.data.with_columns(pl.lit(1.0).alias(COEF_KEY)), model=self._model
        )

    @staticmethod
    def _parse_set(
        *over: Set | Iterable[Set],
    ) -> pl.DataFrame | None:
        """
        >>> import pandas as pd
        >>> dim1 = pd.Index([1, 2, 3], name="dim1")
        >>> dim2 = pd.Index(["a", "b"], name="dim1")
        >>> Variable._parse_set([dim1, dim2])
        Traceback (most recent call last):
        ...
        AssertionError: All coordinates must have unique column names.
        >>> dim2.name = "dim2"
        >>> Variable._parse_set([dim1, dim2])
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

        over_iter: Iterable[Set] = _parse_inputs_as_iterable(over)

        over_frames: List[pl.DataFrame] = [_set_to_polars(set) for set in over_iter]

        over_merged = over_frames[0]

        for df in over_frames[1:]:
            assert (
                set(over_merged.columns) & set(df.columns) == set()
            ), "All coordinates must have unique column names."
            over_merged = over_merged.join(df, how="cross")
        return over_merged

    def next(self, dim: str, wrap_around=False) -> Expression:
        """
        TODO add documentation

        >>> import pandas as pd
        >>> from pyoframe import Variable, Model
        >>> time_dim = pd.DataFrame({"time": ["00:00", "06:00", "12:00", "18:00"]})
        >>> space_dim = pd.DataFrame({"city": ["Toronto", "Berlin"]})
        >>> m = Model()
        >>> m.bat_charge = Variable(time_dim, space_dim)
        >>> m.bat_flow = Variable(time_dim, space_dim)
        >>> (m.bat_charge + m.bat_flow).within({"time": ["00:00", "06:00", "12:00"]}) == m.bat_charge.next("time")
        <Constraint sense='=' size=6 dimensions={'time': 3, 'city': 2} terms=18>
        [00:00,Berlin]: bat_charge[00:00,Berlin] + bat_flow[00:00,Berlin] - bat_charge[06:00,Berlin] = 0
        [00:00,Toronto]: bat_charge[00:00,Toronto] + bat_flow[00:00,Toronto] - bat_charge[06:00,Toronto] = 0
        [06:00,Berlin]: bat_charge[06:00,Berlin] + bat_flow[06:00,Berlin] - bat_charge[12:00,Berlin] = 0
        [06:00,Toronto]: bat_charge[06:00,Toronto] + bat_flow[06:00,Toronto] - bat_charge[12:00,Toronto] = 0
        [12:00,Berlin]: bat_charge[12:00,Berlin] + bat_flow[12:00,Berlin] - bat_charge[18:00,Berlin] = 0
        [12:00,Toronto]: bat_charge[12:00,Toronto] + bat_flow[12:00,Toronto] - bat_charge[18:00,Toronto] = 0

        >>> (m.bat_charge + m.bat_flow) == m.bat_charge.next("time", wrap_around=True)
        <Constraint sense='=' size=8 dimensions={'time': 4, 'city': 2} terms=24>
        [00:00,Berlin]: bat_charge[00:00,Berlin] + bat_flow[00:00,Berlin] - bat_charge[06:00,Berlin] = 0
        [00:00,Toronto]: bat_charge[00:00,Toronto] + bat_flow[00:00,Toronto] - bat_charge[06:00,Toronto] = 0
        [06:00,Berlin]: bat_charge[06:00,Berlin] + bat_flow[06:00,Berlin] - bat_charge[12:00,Berlin] = 0
        [06:00,Toronto]: bat_charge[06:00,Toronto] + bat_flow[06:00,Toronto] - bat_charge[12:00,Toronto] = 0
        [12:00,Berlin]: bat_charge[12:00,Berlin] + bat_flow[12:00,Berlin] - bat_charge[18:00,Berlin] = 0
        [12:00,Toronto]: bat_charge[12:00,Toronto] + bat_flow[12:00,Toronto] - bat_charge[18:00,Toronto] = 0
        [18:00,Berlin]: bat_charge[18:00,Berlin] + bat_flow[18:00,Berlin] - bat_charge[00:00,Berlin] = 0
        [18:00,Toronto]: bat_charge[18:00,Toronto] + bat_flow[18:00,Toronto] - bat_charge[00:00,Toronto] = 0
        """

        wrapped = self.data.select(dim).unique().sort(by=dim)
        wrapped = wrapped.with_columns(pl.col(dim).shift(-1).alias("__next"))
        if wrap_around:
            wrapped = wrapped.with_columns(pl.col("__next").fill_null(pl.first(dim)))
        else:
            wrapped = wrapped.drop_nulls(dim)

        expr = self.to_expr()
        data = expr.data.rename({dim: "__prev"})
        data = data.join(
            wrapped, left_on="__prev", right_on="__next", how="inner"
        ).drop(["__prev", "__next"])
        return expr._new(data)
