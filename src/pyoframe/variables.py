"""
File containing Variable class representing decision variables in optimization models.
"""

from __future__ import annotations
from typing import Iterable

import polars as pl

from pyoframe.constraints import SupportsMath, Set

from pyoframe.constants import COEF_KEY, VAR_KEY, VType, VTypeValue
from pyoframe.constraints import Expression
from pyoframe.model_element import ModelElement
from pyoframe.constraints import SetTypes
from pyoframe.util import get_obj_repr


class Variable(ModelElement, SupportsMath):
    """
    Represents one or many decision variable in an optimization model.

    Parameters
    ----------
    *indexing_sets: SetTypes (typically a DataFrame or Set)
        If no indexing_sets are provided, a single variable with no dimensions is created.
        Otherwise, a variable is created for each element in the Cartesian product of the indexing_sets (see Set for details on behaviour).
    lb: float
        The lower bound for all variables.
    ub: float
        The upper bound for all variables.
    vtype: VType | VTypeValue
        The type of the variable. Can be either a VType enum or a string. Default is VType.CONTINUOUS.

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
    ValueError: Duplicate rows found in input data.
    >>> Variable(df[["dim1"]].drop_duplicates())
    <Variable lb=-inf ub=inf size=3 dimensions={'dim1': 3}>
    [1]: x7
    [2]: x8
    [3]: x9
    """

    _var_count = 1  # Must start at 1 since 0 is reserved for constant terms

    @classmethod
    def reset_var_count(cls):
        """Resets the variable count. Useful to ensure consistency in unit tests."""
        cls._var_count = 1

    # TODO: Breaking change, remove support for Iterable[AcceptableSets]
    def __init__(
        self,
        *indexing_sets: SetTypes | Iterable[SetTypes],
        lb: float = float("-inf"),
        ub: float = float("inf"),
        vtype: VType | VTypeValue = VType.CONTINUOUS,
    ):
        if len(indexing_sets) == 0:
            data = pl.DataFrame({VAR_KEY: [Variable._var_count]})
        else:
            data = Set(*indexing_sets).data.with_columns(
                pl.int_range(Variable._var_count, Variable._var_count + pl.len()).alias(
                    VAR_KEY
                )
            )
        super().__init__(data)

        Variable._var_count += data.height

        self.lb = lb
        self.ub = ub
        self.vtype: VType = VType(vtype)

    def __repr__(self):
        return (
            get_obj_repr(
                self, ("name", "lb", "ub"), size=self.data.height, dimensions=self.shape
            )
            + "\n"
            + self.to_expr().to_str(max_line_len=80, max_rows=10)
        )

    def to_expr(self) -> Expression:
        return self._new(self.data)

    def _new(self, data: pl.DataFrame):
        e = Expression(data.with_columns(pl.lit(1.0).alias(COEF_KEY)))
        e._model = self._model
        e.unmatched_strategy = self.unmatched_strategy
        e.allowed_new_dims = self.allowed_new_dims
        return e

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
        >>> # Fails because the dimensions are not the same
        >>> m.bat_charge + m.bat_flow == m.bat_charge.next("time")
        Traceback (most recent call last):
        ...
        pyoframe.arithmetic.PyoframeError: Failed to add expressions:
        <Expression size=8 dimensions={'time': 4, 'city': 2} terms=16> + <Expression size=6 dimensions={'city': 2, 'time': 3} terms=6>
        Due to error:
        Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()
        shape: (2, 4)
        ┌───────┬─────────┬────────────┬────────────┐
        │ time  ┆ city    ┆ time_right ┆ city_right │
        │ ---   ┆ ---     ┆ ---        ┆ ---        │
        │ str   ┆ str     ┆ str        ┆ str        │
        ╞═══════╪═════════╪════════════╪════════════╡
        │ 18:00 ┆ Toronto ┆ null       ┆ null       │
        │ 18:00 ┆ Berlin  ┆ null       ┆ null       │
        └───────┴─────────┴────────────┴────────────┘
        >>> (m.bat_charge + m.bat_flow).drop_unmatched() == m.bat_charge.next("time")
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

        wrapped = self.data.select(dim).unique(maintain_order=True).sort(by=dim)
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
