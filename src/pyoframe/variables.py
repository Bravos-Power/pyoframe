"""
File containing Variable class representing decision variables in optimization models.
"""

from __future__ import annotations
from typing import Iterable, TYPE_CHECKING

import polars as pl

from pyoframe.constraints import SupportsMath, Set

from pyoframe.constants import (
    COEF_KEY,
    RC_COL,
    SOLUTION_KEY,
    VAR_KEY,
    VType,
    VTypeValue,
)
from pyoframe.constraints import Expression, SupportsToExpr
from pyoframe.constraints import SetTypes
from pyoframe.util import get_obj_repr, unwrap_single_values
from pyoframe.model_element import CountableModelElement, SupportPolarsMethodMixin

if TYPE_CHECKING:
    from pyoframe.model import Model


class Variable(CountableModelElement, SupportsMath, SupportPolarsMethodMixin):
    """
    Represents one or many decision variable in an optimization model.

    Parameters:
        *indexing_sets: SetTypes (typically a DataFrame or Set)
            If no indexing_sets are provided, a single variable with no dimensions is created.
            Otherwise, a variable is created for each element in the Cartesian product of the indexing_sets (see Set for details on behaviour).
        lb: float
            The lower bound for all variables.
        ub: float
            The upper bound for all variables.
        vtype: VType | VTypeValue
            The type of the variable. Can be either a VType enum or a string. Default is VType.CONTINUOUS.

    Examples:
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

    # TODO: Breaking change, remove support for Iterable[AcceptableSets]
    def __init__(
        self,
        *indexing_sets: SetTypes | Iterable[SetTypes],
        lb: float | int | SupportsToExpr = float("-inf"),
        ub: float | int | SupportsToExpr = float("inf"),
        vtype: VType | VTypeValue = VType.CONTINUOUS,
    ):
        data = Set(*indexing_sets).data if len(indexing_sets) > 0 else pl.DataFrame()
        super().__init__(data)

        self.vtype: VType = VType(vtype)
        self._fixed_to = None

        # Tightening the bounds is not strictly necessary, but it adds clarity
        if self.vtype == VType.BINARY:
            lb, ub = 0, 1

        if isinstance(lb, (float, int)):
            self.lb, self.lb_constraint = lb, None
        else:
            self.lb, self.lb_constraint = float("-inf"), lb <= self

        if isinstance(ub, (float, int)):
            self.ub, self.ub_constraint = ub, None
        else:
            self.ub, self.ub_constraint = float("inf"), self <= ub

    def on_add_to_model(self, model: "Model", name: str):
        super().on_add_to_model(model, name)
        if self.lb_constraint is not None:
            setattr(model, f"{name}_lb", self.lb_constraint)
        if self.ub_constraint is not None:
            setattr(model, f"{name}_ub", self.ub_constraint)
        if self._fixed_to is not None:
            setattr(model, f"{name}_fixed", self == self._fixed_to)

    @classmethod
    def create_fixed(cls, expr: SupportsToExpr):
        v = Variable(expr)
        v._fixed_to = expr
        return v

    @classmethod
    def get_id_column_name(cls):
        return VAR_KEY

    @property
    @unwrap_single_values
    def solution(self):
        if SOLUTION_KEY not in self.data.columns:
            raise ValueError(f"No solution solution found for Variable '{self.name}'.")

        return self.data.select(self.dimensions_unsafe + [SOLUTION_KEY])

    @property
    @unwrap_single_values
    def RC(self):
        """
        The reduced cost of the variable.
        Will raise an error if the model has not already been solved.
        The first call to this property will load the reduced costs from the solver (lazy loading).
        """
        if RC_COL not in self.data.columns:
            if self._model.solver is None:
                raise ValueError("The model has not been solved yet.")
            self._model.solver.load_rc()
        return self.data.select(self.dimensions_unsafe + [RC_COL])

    @RC.setter
    def RC(self, value):
        self._extend_dataframe_by_id(value)

    @solution.setter
    def solution(self, value):
        self._extend_dataframe_by_id(value)

    def __repr__(self):
        return (
            get_obj_repr(
                self, ("name", "lb", "ub"), size=self.data.height, dimensions=self.shape
            )
            + "\n"
            + self.to_expr().to_str(max_line_len=80, max_rows=10)
        )

    def to_expr(self) -> Expression:
        return self._new(self.data.drop(SOLUTION_KEY))

    def _new(self, data: pl.DataFrame):
        e = Expression(data.with_columns(pl.lit(1.0).alias(COEF_KEY)))
        e._model = self._model
        # We propogate the unmatched strategy intentionally. Without this a .keep_unmatched() on a variable would always be lost.
        e.unmatched_strategy = self.unmatched_strategy
        e.allowed_new_dims = self.allowed_new_dims
        return e

    def next(self, dim: str, wrap_around: bool = False) -> Expression:
        """
        Creates an expression where the variable at each index is the next variable in the specified dimension.

        Parameters:
            dim:
                The dimension over which to shift the variable.
            wrap_around:
                If True, the last index in the dimension is connected to the first index.

        Examples:
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
            pyoframe.constants.PyoframeError: Failed to add expressions:
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
