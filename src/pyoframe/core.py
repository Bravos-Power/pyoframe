from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    overload,
)

import numpy as np
import pandas as pd
import polars as pl
import pyoptinterface as poi

from pyoframe._arithmetic import (
    _add_expressions,
    _get_dimensions,
    _multiply_expressions,
)
from pyoframe.constants import (
    COEF_KEY,
    CONST_TERM,
    CONSTRAINT_KEY,
    DUAL_KEY,
    KEY_TYPE,
    POLARS_VERSION,
    QUAD_VAR_KEY,
    RESERVED_COL_KEYS,
    SOLUTION_KEY,
    VAR_KEY,
    Config,
    ConstraintSense,
    ObjSense,
    PyoframeError,
    UnmatchedStrategy,
    VType,
    VTypeValue,
)
from pyoframe.model_element import (
    ModelElement,
    ModelElementWithId,
    SupportPolarsMethodMixin,
)
from pyoframe.util import (
    Container,
    FuncArgs,
    cast_coef_to_string,
    concat_dimensions,
    dataframe_to_tupled_list,
    get_obj_repr,
    parse_inputs_as_iterable,
    unwrap_single_values,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Model


def _forward_to_expression(func_name: str):
    def wrapper(self: "SupportsMath", *args, **kwargs) -> "Expression":
        expr = self.to_expr()
        return getattr(expr, func_name)(*args, **kwargs)

    return wrapper


class SupportsToExpr(Protocol):
    def to_expr(self) -> "Expression": ...


class SupportsMath(ABC, SupportsToExpr):
    """Any object that can be converted into an expression."""

    def __init__(self, **kwargs):
        self.unmatched_strategy = UnmatchedStrategy.UNSET
        self.allowed_new_dims: List[str] = []
        super().__init__(**kwargs)

    def keep_unmatched(self):
        self.unmatched_strategy = UnmatchedStrategy.KEEP
        return self

    def drop_unmatched(self):
        self.unmatched_strategy = UnmatchedStrategy.DROP
        return self

    def add_dim(self, *dims: str):
        self.allowed_new_dims.extend(dims)
        return self

    @abstractmethod
    def to_expr(self) -> "Expression": ...

    __add__ = _forward_to_expression("__add__")
    __mul__ = _forward_to_expression("__mul__")
    sum = _forward_to_expression("sum")
    map = _forward_to_expression("map")

    def __pow__(self, power: int):
        """
        Support squaring expressions:
        >>> m = pf.Model()
        >>> m.v = pf.Variable()
        >>> m.v ** 2
        <Expression size=1 dimensions={} terms=1 degree=2>
        v * v
        >>> m.v ** 3
        Traceback (most recent call last):
        ...
        ValueError: Raising an expressions to **3 is not supported. Expressions can only be squared (**2).
        """
        if power == 2:
            return self * self
        raise ValueError(
            f"Raising an expressions to **{power} is not supported. Expressions can only be squared (**2)."
        )

    def __neg__(self):
        res = self.to_expr() * -1
        # Negating a constant term should keep the unmatched strategy
        res.unmatched_strategy = self.unmatched_strategy
        return res

    def __sub__(self, other):
        """
        >>> import polars as pl
        >>> m = pf.Model()
        >>> df = pl.DataFrame({"dim1": [1,2,3], "value": [1,2,3]})
        >>> m.v = pf.Variable(df["dim1"])
        >>> m.v - df
        <Expression size=3 dimensions={'dim1': 3} terms=6>
        [1]: v[1] -1
        [2]: v[2] -2
        [3]: v[3] -3
        """
        if not isinstance(other, (int, float)):
            other = other.to_expr()
        return self.to_expr() + (-other)

    def __rmul__(self, other):
        return self.to_expr() * other

    def __radd__(self, other):
        return self.to_expr() + other

    def __truediv__(self, other):
        """

        Examples:
            Support division.
            >>> m = pf.Model()
            >>> m.v = Variable({"dim1": [1,2,3]})
            >>> m.v / 2
            <Expression size=3 dimensions={'dim1': 3} terms=3>
            [1]: 0.5 v[1]
            [2]: 0.5 v[2]
            [3]: 0.5 v[3]
        """
        return self.to_expr() * (1 / other)

    def __rsub__(self, other):
        """
        Support right subtraction.

        Examples:
            >>> m = pf.Model()
            >>> m.v = Variable({"dim1": [1,2,3]})
            >>> 1 - m.v
            <Expression size=3 dimensions={'dim1': 3} terms=6>
            [1]: 1  - v[1]
            [2]: 1  - v[2]
            [3]: 1  - v[3]
        """
        return other + (-self.to_expr())

    def __le__(self, other):
        """Equality constraint.

        Examples:
            >>> m = pf.Model()
            >>> m.v = pf.Variable()
            >>> m.v <= 1
            <Constraint sense='<=' size=1 dimensions={} terms=2>
            v <= 1
        """
        return Constraint(self - other, ConstraintSense.LE)

    def __ge__(self, other):
        """Equality constraint.

        Examples:
            >>> m = pf.Model()
            >>> m.v = pf.Variable()
            >>> m.v >= 1
            <Constraint sense='>=' size=1 dimensions={} terms=2>
            v >= 1
        """
        return Constraint(self - other, ConstraintSense.GE)

    def __eq__(self, value: object):  # type: ignore
        """Equality constraint.

        Examples:
            >>> m = pf.Model()
            >>> m.v = pf.Variable()
            >>> m.v == 1
            <Constraint sense='=' size=1 dimensions={} terms=2>
            v = 1
        """
        return Constraint(self - value, ConstraintSense.EQ)


SetTypes = Union[
    pl.DataFrame,
    pd.Index,
    pd.DataFrame,
    SupportsMath,
    Mapping[str, Sequence[object]],
    "Set",
    "Constraint",
]


class Set(ModelElement, SupportsMath, SupportPolarsMethodMixin):
    """
    A set which can then be used to index variables.

    Examples:
        >>> pf.Set(x=range(2), y=range(3))
        <Set size=6 dimensions={'x': 2, 'y': 3}>
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    """

    def __init__(self, *data: SetTypes | Iterable[SetTypes], **named_data):
        data_list = list(data)
        for name, set in named_data.items():
            data_list.append({name: set})
        df = self._parse_acceptable_sets(*data_list)
        if not df.is_empty() and df.is_duplicated().any():
            raise ValueError("Duplicate rows found in input data.")
        super().__init__(df)

    def _new(self, data: pl.DataFrame):
        s = Set(data)
        s._model = self._model
        # Copy over the unmatched strategy on operations like .rename(), .with_columns(), etc.
        s.unmatched_strategy = self.unmatched_strategy
        return s

    @staticmethod
    def _parse_acceptable_sets(
        *over: SetTypes | Iterable[SetTypes],
    ) -> pl.DataFrame:
        """
        Examples:
            >>> import pandas as pd
            >>> dim1 = pd.Index([1, 2, 3], name="dim1")
            >>> dim2 = pd.Index(["a", "b"], name="dim1")
            >>> Set._parse_acceptable_sets([dim1, dim2])
            Traceback (most recent call last):
            ...
            AssertionError: All coordinates must have unique column names.
            >>> dim2.name = "dim2"
            >>> Set._parse_acceptable_sets([dim1, dim2])
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
        assert len(over) > 0, "At least one set must be provided."
        over_iter: Iterable[SetTypes] = parse_inputs_as_iterable(*over)

        over_frames: List[pl.DataFrame] = [Set._set_to_polars(set) for set in over_iter]

        over_merged = over_frames[0]

        for df in over_frames[1:]:
            assert (
                set(over_merged.columns) & set(df.columns) == set()
            ), "All coordinates must have unique column names."
            over_merged = over_merged.join(df, how="cross")
        return over_merged

    def to_expr(self) -> Expression:
        return Expression(
            self.data.with_columns(
                pl.lit(1).alias(COEF_KEY), pl.lit(CONST_TERM).alias(VAR_KEY)
            )
        )

    def __mul__(self, other):
        if isinstance(other, Set):
            assert (
                set(self.data.columns) & set(other.data.columns) == set()
            ), "Cannot multiply two sets with columns in common."
            return Set(self.data, other.data)
        return super().__mul__(other)

    def __add__(self, other):
        if isinstance(other, Set):
            try:
                return self._new(
                    pl.concat([self.data, other.data]).unique(maintain_order=True)
                )
            except pl.exceptions.ShapeError as e:
                if "unable to vstack, column names don't match" in str(e):
                    raise PyoframeError(
                        f"Failed to add sets '{self.friendly_name}' and '{other.friendly_name}' because dimensions do not match ({self.dimensions} != {other.dimensions}) "
                    ) from e
                raise e

        return super().__add__(other)

    def __repr__(self):
        return (
            get_obj_repr(self, ("name",), size=self.data.height, dimensions=self.shape)
            + "\n"
            + dataframe_to_tupled_list(
                self.data, num_max_elements=Config.print_max_set_elements
            )
        )

    @staticmethod
    def _set_to_polars(set: "SetTypes") -> pl.DataFrame:
        if isinstance(set, dict):
            df = pl.DataFrame(set)
        elif isinstance(set, Constraint):
            df = set.data.select(set.dimensions_unsafe)
        elif isinstance(set, SupportsMath):
            if POLARS_VERSION.major < 1:
                df = (
                    set.to_expr()
                    .data.drop(RESERVED_COL_KEYS)
                    .unique(maintain_order=True)
                )
            else:
                df = (
                    set.to_expr()
                    .data.drop(RESERVED_COL_KEYS, strict=False)
                    .unique(maintain_order=True)
                )
        elif isinstance(set, pd.Index):
            df = pl.from_pandas(pd.DataFrame(index=set).reset_index())
        elif isinstance(set, pd.DataFrame):
            df = pl.from_pandas(set)
        elif isinstance(set, pl.DataFrame):
            df = set
        elif isinstance(set, pl.Series):
            df = set.to_frame()
        elif isinstance(set, Set):
            df = set.data
        elif isinstance(set, range):
            raise ValueError(
                "Cannot convert a range to a set without a dimension name. Try Set(dim_name=range(...))"
            )
        else:
            raise ValueError(f"Cannot convert type {type(set)} to a polars DataFrame")

        if "index" in df.columns:
            raise ValueError(
                "Please specify a custom dimension name rather than using 'index' to avoid confusion."
            )

        for reserved_key in RESERVED_COL_KEYS:
            if reserved_key in df.columns:
                raise ValueError(
                    f"Cannot use reserved column names {reserved_key} as dimensions."
                )

        return df


class Expression(ModelElement, SupportsMath, SupportPolarsMethodMixin):
    """A linear or quadratic expression."""

    def __init__(self, data: pl.DataFrame):
        """
        A linear expression.

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"item" : [1, 1, 1, 2, 2], "time": ["mon", "tue", "wed", "mon", "tue"], "cost": [1, 2, 3, 4, 5]}).set_index(["item", "time"])
            >>> m = pf.Model()
            >>> m.Time = pf.Variable(df.index)
            >>> m.Size = pf.Variable(df.index)
            >>> expr = df["cost"] * m.Time + df["cost"] * m.Size
            >>> expr
            <Expression size=5 dimensions={'item': 2, 'time': 3} terms=10>
            [1,mon]: Time[1,mon] + Size[1,mon]
            [1,tue]: 2 Time[1,tue] +2 Size[1,tue]
            [1,wed]: 3 Time[1,wed] +3 Size[1,wed]
            [2,mon]: 4 Time[2,mon] +4 Size[2,mon]
            [2,tue]: 5 Time[2,tue] +5 Size[2,tue]
        """
        # Sanity checks, VAR_KEY and COEF_KEY must be present
        assert VAR_KEY in data.columns, "Missing variable column."
        assert COEF_KEY in data.columns, "Missing coefficient column."

        # Sanity check no duplicates indices
        if Config.enable_is_duplicated_expression_safety_check:
            duplicated_mask = data.drop(COEF_KEY).is_duplicated()
            # In theory this should never happen unless there's a bug in the library
            if duplicated_mask.any():  # pragma: no cover
                duplicated_data = data.filter(duplicated_mask)
                raise ValueError(
                    f"Cannot create an expression with duplicate indices:\n{duplicated_data}."
                )

        super().__init__(data)

    # Might add this in later
    # @classmethod
    # def empty(cls, dimensions=[], type=None):
    #     data = {COEF_KEY: [], VAR_KEY: []}
    #     data.update({d: [] for d in dimensions})
    #     schema = {COEF_KEY: pl.Float64, VAR_KEY: pl.UInt32}
    #     if type is not None:
    #         schema.update({d: t for d, t in zip(dimensions, type)})
    #     return Expression(
    #         pl.DataFrame(data).with_columns(
    #             *[pl.col(c).cast(t) for c, t in schema.items()]
    #         )
    #     )

    @classmethod
    def constant(cls, constant: int | float) -> "Expression":
        """
        Examples:
            >>> pf.Expression.constant(5)
            <Expression size=1 dimensions={} terms=1>
            5
        """
        return cls(
            pl.DataFrame(
                {
                    COEF_KEY: [constant],
                    VAR_KEY: [CONST_TERM],
                },
                schema={COEF_KEY: pl.Float64, VAR_KEY: KEY_TYPE},
            )
        )

    def sum(self, over: Union[str, Iterable[str]]):
        """
        Examples:
            >>> import pandas as pd
            >>> m = pf.Model()
            >>> df = pd.DataFrame({"item" : [1, 1, 1, 2, 2], "time": ["mon", "tue", "wed", "mon", "tue"], "cost": [1, 2, 3, 4, 5]}).set_index(["item", "time"])
            >>> m.quantity = Variable(df.reset_index()[["item"]].drop_duplicates())
            >>> expr = (m.quantity * df["cost"]).sum("time")
            >>> expr.data
            shape: (2, 3)
            ┌──────┬─────────┬───────────────┐
            │ item ┆ __coeff ┆ __variable_id │
            │ ---  ┆ ---     ┆ ---           │
            │ i64  ┆ f64     ┆ u32           │
            ╞══════╪═════════╪═══════════════╡
            │ 1    ┆ 6.0     ┆ 1             │
            │ 2    ┆ 9.0     ┆ 2             │
            └──────┴─────────┴───────────────┘
        """
        if isinstance(over, str):
            over = [over]
        dims = self.dimensions
        if not dims:
            raise ValueError(
                f"Cannot sum over dimensions {over} since the current expression has no dimensions."
            )
        assert set(over) <= set(dims), f"Cannot sum over {over} as it is not in {dims}"
        remaining_dims = [dim for dim in dims if dim not in over]

        return self._new(
            self.data.drop(over)
            .group_by(remaining_dims + self._variable_columns, maintain_order=True)
            .sum()
        )

    @property
    def _variable_columns(self) -> List[str]:
        if self.is_quadratic:
            return [VAR_KEY, QUAD_VAR_KEY]
        else:
            return [VAR_KEY]

    def map(self, mapping_set: SetTypes, drop_shared_dims: bool = True) -> Expression:
        """
        Replaces the dimensions that are shared with mapping_set with the other dimensions found in mapping_set.

        This is particularly useful to go from one type of dimensions to another. For example, to convert data that
        is indexed by city to data indexed by country (see example).

        Parameters:
            mapping_set:
                The set to map the expression to. This can be a DataFrame, Index, or another Set.
            drop_shared_dims:
                If True, the dimensions shared between the expression and the mapping set are dropped from the resulting expression and
                    repeated rows are summed.
                If False, the shared dimensions are kept in the resulting expression.

        Returns:
            A new Expression containing the result of the mapping operation.

        Examples:
            >>> import polars as pl
            >>> pop_data = pl.DataFrame({"city": ["Toronto", "Vancouver", "Boston"], "year": [2024, 2024, 2024], "population": [10, 2, 8]}).to_expr()
            >>> cities_and_countries = pl.DataFrame({"city": ["Toronto", "Vancouver", "Boston"], "country": ["Canada", "Canada", "USA"]})
            >>> pop_data.map(cities_and_countries)
            <Expression size=2 dimensions={'year': 1, 'country': 2} terms=2>
            [2024,Canada]: 12
            [2024,USA]: 8
            >>> pop_data.map(cities_and_countries, drop_shared_dims=False)
            <Expression size=3 dimensions={'city': 3, 'year': 1, 'country': 2} terms=3>
            [Toronto,2024,Canada]: 10
            [Vancouver,2024,Canada]: 2
            [Boston,2024,USA]: 8
        """
        mapping_set = Set(mapping_set)

        dims = self.dimensions
        if dims is None:
            raise ValueError("Cannot use .map() on an expression with no dimensions.")

        mapping_dims = mapping_set.dimensions
        if mapping_dims is None:
            raise ValueError(
                "Cannot use .map() with a mapping set containing no dimensions."
            )

        shared_dims = [dim for dim in dims if dim in mapping_dims]
        if not shared_dims:
            raise ValueError(
                f"Cannot apply .map() as there are no shared dimensions between the expression (dims={self.dimensions}) and the mapping set (dims={mapping_set.dimensions})."
            )

        mapped_expression = self * mapping_set

        if drop_shared_dims:
            return sum(shared_dims, mapped_expression)

        return mapped_expression

    def rolling_sum(self, over: str, window_size: int) -> Expression:
        """
        Calculates the rolling sum of the Expression over a specified window size for a given dimension.

        This method applies a rolling sum operation over the dimension specified by `over`,
        using a window defined by `window_size`.


        Parameters:
            over :
                The name of the dimension (column) over which the rolling sum is calculated.
                This dimension must exist within the Expression's dimensions.
            window_size :
                The size of the moving window in terms of number of records.
                The rolling sum is calculated over this many consecutive elements.

        Returns:
            A new Expression instance containing the result of the rolling sum operation.
                This new Expression retains all dimensions (columns) of the original data,
                with the rolling sum applied over the specified dimension.

        Examples:
            >>> import polars as pl
            >>> cost = pl.DataFrame({"item" : [1, 1, 1, 2, 2], "time": [1, 2, 3, 1, 2], "cost": [1, 2, 3, 4, 5]})
            >>> m = pf.Model()
            >>> m.quantity = pf.Variable(cost[["item", "time"]])
            >>> (m.quantity * cost).rolling_sum(over="time", window_size=2)
            <Expression size=5 dimensions={'item': 2, 'time': 3} terms=8>
            [1,1]: quantity[1,1]
            [1,2]: quantity[1,1] +2 quantity[1,2]
            [1,3]: 2 quantity[1,2] +3 quantity[1,3]
            [2,1]: 4 quantity[2,1]
            [2,2]: 4 quantity[2,1] +5 quantity[2,2]
        """
        dims = self.dimensions
        if dims is None:
            raise ValueError(
                "Cannot use rolling_sum() with an expression with no dimensions."
            )
        assert over in dims, f"Cannot sum over {over} as it is not in {dims}"
        remaining_dims = [dim for dim in dims if dim not in over]

        return self._new(
            pl.concat(
                [
                    df.with_columns(pl.col(over).max())
                    for _, df in self.data.rolling(
                        index_column=over,
                        period=f"{window_size}i",
                        group_by=remaining_dims,
                    )
                ]
            )
        )

    def within(self, set: "SetTypes") -> Expression:
        """
        Examples:
            >>> import pandas as pd
            >>> general_expr = pd.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, 3]}).to_expr()
            >>> filter_expr = pd.DataFrame({"dim1": [1, 3], "value": [5, 6]}).to_expr()
            >>> general_expr.within(filter_expr).data
            shape: (2, 3)
            ┌──────┬─────────┬───────────────┐
            │ dim1 ┆ __coeff ┆ __variable_id │
            │ ---  ┆ ---     ┆ ---           │
            │ i64  ┆ f64     ┆ u32           │
            ╞══════╪═════════╪═══════════════╡
            │ 1    ┆ 1.0     ┆ 0             │
            │ 3    ┆ 3.0     ┆ 0             │
            └──────┴─────────┴───────────────┘
        """
        df: pl.DataFrame = Set(set).data
        set_dims = _get_dimensions(df)
        assert (
            set_dims is not None
        ), "Cannot use .within() with a set with no dimensions."
        dims = self.dimensions
        assert (
            dims is not None
        ), "Cannot use .within() with an expression with no dimensions."
        dims_in_common = [dim for dim in dims if dim in set_dims]
        by_dims = df.select(dims_in_common).unique(maintain_order=True)
        return self._new(self.data.join(by_dims, on=dims_in_common))

    @property
    def is_quadratic(self) -> bool:
        """
        Returns True if the expression is quadratic, False otherwise.

        Computes in O(1) since expressions are quadratic if and
        only if self.data contain the QUAD_VAR_KEY column.

        Examples:
            >>> import pandas as pd
            >>> m = pf.Model()
            >>> m.v = Variable()
            >>> expr = pd.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, 3]}) * m.v
            >>> expr *= m.v
            >>> expr.is_quadratic
            True
        """
        return QUAD_VAR_KEY in self.data.columns

    def degree(self) -> int:
        """
        Returns the degree of the expression (0=constant, 1=linear, 2=quadratic).

        Examples:
            >>> import pandas as pd
            >>> m = pf.Model()
            >>> m.v1 = pf.Variable()
            >>> m.v2 = pf.Variable()
            >>> expr = pd.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, 3]}).to_expr()
            >>> expr.degree()
            0
            >>> expr *= m.v1
            >>> expr.degree()
            1
            >>> expr += (m.v2 ** 2).add_dim("dim1")
            >>> expr.degree()
            2
        """
        if self.is_quadratic:
            return 2
        elif (self.data.get_column(VAR_KEY) != CONST_TERM).any():
            return 1
        else:
            return 0

    def __add__(self, other):
        """
        Examples:
            >>> import pandas as pd
            >>> m = pf.Model()
            >>> add = pd.DataFrame({"dim1": [1,2,3], "add": [10, 20, 30]}).to_expr()
            >>> m.v = Variable(add)
            >>> m.v + add
            <Expression size=3 dimensions={'dim1': 3} terms=6>
            [1]: v[1] +10
            [2]: v[2] +20
            [3]: v[3] +30
            >>> m.v + add + 2
            <Expression size=3 dimensions={'dim1': 3} terms=6>
            [1]: v[1] +12
            [2]: v[2] +22
            [3]: v[3] +32
            >>> m.v + pd.DataFrame({"dim1": [1,2], "add": [10, 20]})
            Traceback (most recent call last):
            ...
            pyoframe.constants.PyoframeError: Failed to add expressions:
            <Expression size=3 dimensions={'dim1': 3} terms=3> + <Expression size=2 dimensions={'dim1': 2} terms=2>
            Due to error:
            Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()
            shape: (1, 2)
            ┌──────┬────────────┐
            │ dim1 ┆ dim1_right │
            │ ---  ┆ ---        │
            │ i64  ┆ i64        │
            ╞══════╪════════════╡
            │ 3    ┆ null       │
            └──────┴────────────┘
            >>> m.v2 = Variable()
            >>> 5 + 2 * m.v2
            <Expression size=1 dimensions={} terms=2>
            2 v2 +5
        """
        if isinstance(other, str):
            raise ValueError(
                "Cannot add a string to an expression. Perhaps you meant to use pf.sum() instead of sum()?"
            )
        if isinstance(other, (int, float)):
            return self._add_const(other)
        other = other.to_expr()
        self._learn_from_other(other)
        return _add_expressions(self, other)

    def __mul__(
        self: "Expression", other: int | float | SupportsToExpr
    ) -> "Expression":
        if isinstance(other, (int, float)):
            return self.with_columns(pl.col(COEF_KEY) * other)

        other = other.to_expr()
        self._learn_from_other(other)
        return _multiply_expressions(self, other)

    def to_expr(self) -> Expression:
        return self

    def _learn_from_other(self, other: Expression):
        if self._model is None and other._model is not None:
            self._model = other._model

    def _new(self, data: pl.DataFrame) -> Expression:
        e = Expression(data)
        e._model = self._model
        # Note: We intentionally don't propogate the unmatched strategy to the new expression
        e.allowed_new_dims = self.allowed_new_dims
        return e

    def _add_const(self, const: int | float) -> Expression:
        """
        Examples:
            >>> m = pf.Model()
            >>> m.x1 = Variable()
            >>> m.x2 = Variable()
            >>> m.x1 + 5
            <Expression size=1 dimensions={} terms=2>
            x1 +5
            >>> m.x1 ** 2 + 5
            <Expression size=1 dimensions={} terms=2 degree=2>
            x1 * x1 +5
            >>> m.x1 ** 2 + m.x2 + 5
            <Expression size=1 dimensions={} terms=3 degree=2>
            x1 * x1 + x2 +5

            It also works with dimensions

            >>> m = pf.Model()
            >>> m.v = Variable({"dim1": [1, 2, 3]})
            >>> m.v * m.v + 5
            <Expression size=3 dimensions={'dim1': 3} terms=6 degree=2>
            [1]: 5  + v[1] * v[1]
            [2]: 5  + v[2] * v[2]
            [3]: 5  + v[3] * v[3]
        """
        dim = self.dimensions
        data = self.data
        # Fill in missing constant terms
        if not dim:
            if CONST_TERM not in data[VAR_KEY]:
                const_df = pl.DataFrame(
                    {COEF_KEY: [0.0], VAR_KEY: [CONST_TERM]},
                    schema={COEF_KEY: pl.Float64, VAR_KEY: KEY_TYPE},
                )
                if self.is_quadratic:
                    const_df = const_df.with_columns(
                        pl.lit(CONST_TERM).alias(QUAD_VAR_KEY).cast(KEY_TYPE)
                    )
                data = pl.concat(
                    [data, const_df],
                    how="vertical_relaxed",
                )
        else:
            keys = (
                data.select(dim)
                .unique(maintain_order=True)
                .with_columns(pl.lit(CONST_TERM).alias(VAR_KEY).cast(KEY_TYPE))
            )
            if self.is_quadratic:
                keys = keys.with_columns(
                    pl.lit(CONST_TERM).alias(QUAD_VAR_KEY).cast(KEY_TYPE)
                )
            if POLARS_VERSION.major >= 1:
                data = data.join(
                    keys, on=dim + self._variable_columns, how="full", coalesce=True
                )
            else:
                data = data.join(
                    keys, on=dim + self._variable_columns, how="outer_coalesce"
                )
            data = data.with_columns(pl.col(COEF_KEY).fill_null(0.0))

        data = data.with_columns(
            pl.when(pl.col(VAR_KEY) == CONST_TERM)
            .then(pl.col(COEF_KEY) + const)
            .otherwise(pl.col(COEF_KEY))
        )

        return self._new(data)

    @property
    def constant_terms(self):
        dims = self.dimensions
        constant_terms = self.data.filter(pl.col(VAR_KEY) == CONST_TERM).drop(VAR_KEY)
        if self.is_quadratic:
            constant_terms = constant_terms.drop(QUAD_VAR_KEY)
        if dims is not None:
            dims_df = self.data.select(dims).unique(maintain_order=True)
            if POLARS_VERSION.major >= 1:
                df = constant_terms.join(dims_df, on=dims, how="full", coalesce=True)
            else:
                df = constant_terms.join(dims_df, on=dims, how="outer_coalesce")
            return df.with_columns(pl.col(COEF_KEY).fill_null(0.0))
        else:
            if len(constant_terms) == 0:
                return pl.DataFrame(
                    {COEF_KEY: [0.0], VAR_KEY: [CONST_TERM]},
                    schema={COEF_KEY: pl.Float64, VAR_KEY: KEY_TYPE},
                )
            return constant_terms

    @property
    def variable_terms(self):
        return self.data.filter(pl.col(VAR_KEY) != CONST_TERM)

    @unwrap_single_values
    def evaluate(self) -> pl.DataFrame:
        """
        The value of the expression. Only available after the model has been solved.

        Examples:
            >>> m = pf.Model()
            >>> m.X = pf.Variable({"dim1": [1, 2, 3]}, ub=10)
            >>> m.expr_1 = 2 * m.X + 1
            >>> m.expr_2 = pf.sum(m.expr_1)
            >>> m.maximize = m.expr_2 - 3
            >>> m.attr.Silent = True
            >>> m.optimize()
            >>> m.expr_1.evaluate()
            shape: (3, 2)
            ┌──────┬──────────┐
            │ dim1 ┆ solution │
            │ ---  ┆ ---      │
            │ i64  ┆ f64      │
            ╞══════╪══════════╡
            │ 1    ┆ 21.0     │
            │ 2    ┆ 21.0     │
            │ 3    ┆ 21.0     │
            └──────┴──────────┘
            >>> m.expr_2.evaluate()
            63.0
        """
        assert (
            self._model is not None
        ), "Expression must be added to the model to use .value"

        df = self.data
        sm = self._model.poi
        attr = poi.VariableAttribute.Value
        for var_col in self._variable_columns:
            df = df.with_columns(
                (
                    pl.col(COEF_KEY)
                    * pl.col(var_col).map_elements(
                        lambda v_id: (
                            sm.get_variable_attribute(poi.VariableIndex(v_id), attr)
                            if v_id != CONST_TERM
                            else 1
                        ),
                        return_dtype=pl.Float64,
                    )
                ).alias(COEF_KEY)
            ).drop(var_col)

        df = df.rename({COEF_KEY: SOLUTION_KEY})

        dims = self.dimensions
        if dims is not None:
            df = df.group_by(dims, maintain_order=True)
        return df.sum()

    def to_poi(self) -> poi.ScalarAffineFunction:
        if self.dimensions is not None:
            raise ValueError(
                "Only non-dimensioned expressions can be converted to PyOptInterface."
            )  # pragma: no cover

        return poi.ScalarAffineFunction(
            coefficients=self.data.get_column(COEF_KEY).to_numpy(),
            variables=self.data.get_column(VAR_KEY).to_numpy(),
        )

    def to_str_table(self, include_const_term=True):
        data = self.data if include_const_term else self.variable_terms
        data = cast_coef_to_string(data)

        for var_col in self._variable_columns:
            temp_var_column = f"{var_col}_temp"
            if self._model is not None and self._model.var_map is not None:
                data = self._model.var_map.apply(
                    data, to_col=temp_var_column, id_col=var_col
                )
            else:
                data = data.with_columns(
                    pl.concat_str(pl.lit("x"), var_col).alias(temp_var_column)
                )
            data = data.with_columns(
                pl.when(pl.col(var_col) == CONST_TERM)
                .then(pl.lit(""))
                .otherwise(temp_var_column)
                .alias(var_col)
            ).drop(temp_var_column)
        if self.is_quadratic:
            data = data.with_columns(
                pl.when(pl.col(QUAD_VAR_KEY) == "")
                .then(pl.col(VAR_KEY))
                .otherwise(pl.concat_str(VAR_KEY, pl.lit(" * "), pl.col(QUAD_VAR_KEY)))
                .alias(VAR_KEY)
            ).drop(QUAD_VAR_KEY)

        dimensions = self.dimensions

        # Create a string for each term
        data = data.with_columns(
            expr=pl.concat_str(
                COEF_KEY,
                pl.lit(" "),
                VAR_KEY,
            )
        ).drop(COEF_KEY, VAR_KEY)

        if dimensions is not None:
            data = data.group_by(dimensions, maintain_order=True).agg(
                pl.col("expr").str.concat(delimiter=" ")
            )
        else:
            data = data.select(pl.col("expr").str.concat(delimiter=" "))

        # Remove leading +
        data = data.with_columns(pl.col("expr").str.strip_chars(characters=" +"))

        # TODO add vertical ... if too many rows, in the middle of the table
        if Config.print_max_lines:
            data = data.head(Config.print_max_lines)

        if Config.print_max_line_length:
            data = data.with_columns(
                pl.when(pl.col("expr").str.len_chars() > Config.print_max_line_length)
                .then(
                    pl.concat_str(
                        pl.col("expr").str.slice(0, Config.print_max_line_length),
                        pl.lit("..."),
                    )
                )
                .otherwise(pl.col("expr"))
            )
        return data

    def to_str_create_prefix(self, data):
        if self.name is None and self.dimensions is None:
            return data

        return (
            concat_dimensions(data, prefix=self.name, ignore_columns=["expr"])
            .with_columns(
                pl.concat_str("concated_dim", pl.lit(": "), "expr").alias("expr")
            )
            .drop("concated_dim")
        )

    def to_str(
        self,
        include_const_term=True,
        include_header=False,
        include_data=True,
    ):
        result = ""
        if include_header:
            result += get_obj_repr(
                self,
                size=len(self),
                dimensions=self.shape,
                terms=len(self.data),
                degree=2 if self.degree() == 2 else None,
            )
        if include_header and include_data:
            result += "\n"
        if include_data:
            str_table = self.to_str_table(
                include_const_term=include_const_term,
            )
            str_table = self.to_str_create_prefix(str_table)
            result += str_table.select(pl.col("expr").str.concat(delimiter="\n")).item()

        return result

    def __repr__(self) -> str:
        return self.to_str(include_header=True)

    def __str__(self) -> str:
        return self.to_str()


@overload
def sum(over: Union[str, Sequence[str]], expr: SupportsToExpr) -> "Expression": ...


@overload
def sum(over: SupportsToExpr) -> "Expression": ...


def sum(
    over: Union[str, Sequence[str], SupportsToExpr],
    expr: Optional[SupportsToExpr] = None,
) -> "Expression":
    if expr is None:
        assert isinstance(over, SupportsMath)
        over = over.to_expr()
        all_dims = over.dimensions
        if all_dims is None:
            raise ValueError(
                "Cannot sum over dimensions with an expression with no dimensions."
            )
        return over.sum(all_dims)
    else:
        assert isinstance(over, (str, Sequence))
        return expr.to_expr().sum(over)


def sum_by(by: Union[str, Sequence[str]], expr: SupportsToExpr) -> "Expression":
    if isinstance(by, str):
        by = [by]
    expr = expr.to_expr()
    dimensions = expr.dimensions
    assert (
        dimensions is not None
    ), "Cannot sum by dimensions with an expression with no dimensions."
    remaining_dims = [dim for dim in dimensions if dim not in by]
    return sum(over=remaining_dims, expr=expr)


class Constraint(ModelElementWithId):
    """A linear programming constraint."""

    def __init__(self, lhs: Expression, sense: ConstraintSense):
        """Initialize a constraint.

        Parameters:
            lhs:
                The left hand side of the constraint.
            sense:
                The sense of the constraint.
        """
        self.lhs = lhs
        self._model = lhs._model
        self.sense = sense
        self.to_relax: Optional[FuncArgs] = None
        self.attr = Container(self._set_attribute, self._get_attribute)

        dims = self.lhs.dimensions
        data = pl.DataFrame() if dims is None else self.lhs.data.select(dims).unique()

        super().__init__(data)

    def _set_attribute(self, name, value):
        self._assert_has_ids()
        col_name = name
        try:
            name = poi.ConstraintAttribute[name]
            setter = self._model.poi.set_constraint_attribute
        except KeyError:
            setter = self._model.poi.set_constraint_raw_attribute

        if self.dimensions is None:
            for key in self.data.get_column(CONSTRAINT_KEY):
                setter(poi.ConstraintIndex(poi.ConstraintType.Linear, key), name, value)
        else:
            for key, value in (
                self.data.join(value, on=self.dimensions, how="inner")
                .select(pl.col(CONSTRAINT_KEY), pl.col(col_name))
                .iter_rows()
            ):
                setter(poi.ConstraintIndex(poi.ConstraintType.Linear, key), name, value)

    @unwrap_single_values
    def _get_attribute(self, name):
        self._assert_has_ids()
        col_name = name
        try:
            name = poi.ConstraintAttribute[name]
            getter = self._model.poi.get_constraint_attribute
        except KeyError:
            getter = self._model.poi.get_constraint_raw_attribute

        with (
            warnings.catch_warnings()
        ):  # map_elements without return_dtype= gives a warning
            warnings.filterwarnings(
                action="ignore", category=pl.exceptions.MapWithoutReturnDtypeWarning
            )
            return self.data.with_columns(
                pl.col(CONSTRAINT_KEY)
                .map_elements(
                    lambda v_id: getter(
                        poi.ConstraintIndex(poi.ConstraintType.Linear, v_id), name
                    )
                )
                .alias(col_name)
            ).select(self.dimensions_unsafe + [col_name])

    def on_add_to_model(self, model: "Model", name: str):
        super().on_add_to_model(model, name)
        if self.to_relax is not None:
            self.relax(*self.to_relax.args, **self.to_relax.kwargs)
        self._assign_ids()

    def _assign_ids(self):
        assert self._model is not None

        is_quadratic = self.lhs.is_quadratic
        use_var_names = self._model.use_var_names
        kwargs: Dict[str, Any] = dict(sense=self.sense.to_poi(), rhs=0)

        key_cols = [COEF_KEY] + self.lhs._variable_columns
        key_cols_polars = [pl.col(c) for c in key_cols]

        add_constraint = (
            self._model.poi.add_quadratic_constraint
            if is_quadratic
            else self._model.poi.add_linear_constraint
        )
        ScalarFunction = (
            poi.ScalarQuadraticFunction if is_quadratic else poi.ScalarAffineFunction
        )

        if self.dimensions is None:
            if self._model.use_var_names:
                kwargs["name"] = self.name
            df = self.data.with_columns(
                pl.lit(
                    add_constraint(
                        ScalarFunction(
                            *[self.lhs.data.get_column(c).to_numpy() for c in key_cols]
                        ),
                        **kwargs,
                    ).index
                )
                .alias(CONSTRAINT_KEY)
                .cast(KEY_TYPE)
            )
        else:
            df = self.lhs.data.group_by(self.dimensions, maintain_order=True).agg(
                *key_cols_polars
            )
            if use_var_names:
                df = (
                    concat_dimensions(df, prefix=self.name)
                    .with_columns(
                        pl.struct(*key_cols_polars, pl.col("concated_dim"))
                        .map_elements(
                            lambda x: add_constraint(
                                ScalarFunction(*[np.array(x[c]) for c in key_cols]),
                                name=x["concated_dim"],
                                **kwargs,
                            ).index,
                            return_dtype=KEY_TYPE,
                        )
                        .alias(CONSTRAINT_KEY)
                    )
                    .drop("concated_dim")
                )
            else:
                df = df.with_columns(
                    pl.struct(*key_cols_polars)
                    .map_elements(
                        lambda x: add_constraint(
                            ScalarFunction(*[np.array(x[c]) for c in key_cols]),
                            **kwargs,
                        ).index,
                        return_dtype=KEY_TYPE,
                    )
                    .alias(CONSTRAINT_KEY)
                )
            df = df.drop(key_cols)

        self._data = df

    @property
    @unwrap_single_values
    def dual(self) -> Union[pl.DataFrame, float]:
        dual = self.attr.Dual
        if isinstance(dual, pl.DataFrame):
            dual = dual.rename({"Dual": DUAL_KEY})
        return dual

    @classmethod
    def get_id_column_name(cls):
        return CONSTRAINT_KEY

    def filter(self, *args, **kwargs) -> pl.DataFrame:
        return self.lhs.data.filter(*args, **kwargs)

    def relax(
        self, cost: SupportsToExpr, max: Optional[SupportsToExpr] = None
    ) -> Constraint:
        """
        Relaxes the constraint by adding a variable to the constraint that can be non-zero at a cost.

        Parameters:
            cost:
                The cost of relaxing the constraint. Costs should be positives as they will automatically
                become negative for maximization problems.
            max:
                The maximum value of the relaxation variable.

        Returns:
            The same constraint

        Examples:
            >>> m = pf.Model()
            >>> m.hours_sleep = pf.Variable(lb=0)
            >>> m.hours_day = pf.Variable(lb=0)
            >>> m.hours_in_day = m.hours_sleep + m.hours_day == 24
            >>> m.maximize = m.hours_day
            >>> m.must_sleep = (m.hours_sleep >= 8).relax(cost=2, max=3)
            >>> m.optimize()
            >>> m.hours_day.solution
            16.0
            >>> m.maximize += 2 * m.hours_day
            >>> m.optimize()
            >>> m.hours_day.solution
            19.0

            Note: .relax() can only be called after the sense of the model has been defined.

            >>> m = pf.Model()
            >>> m.hours_sleep = pf.Variable(lb=0)
            >>> m.hours_day = pf.Variable(lb=0)
            >>> m.hours_in_day = m.hours_sleep + m.hours_day == 24
            >>> m.must_sleep = (m.hours_sleep >= 8).relax(cost=2, max=3)
            Traceback (most recent call last):
            ...
            ValueError: Cannot relax a constraint before the objective sense has been set. Try setting the objective first or using Model(sense=...).

            One way to solve this is by setting the sense directly on the model. See how this works fine:

            >>> m = pf.Model(sense="max")
            >>> m.hours_sleep = pf.Variable(lb=0)
            >>> m.hours_day = pf.Variable(lb=0)
            >>> m.hours_in_day = m.hours_sleep + m.hours_day == 24
            >>> m.must_sleep = (m.hours_sleep >= 8).relax(cost=2, max=3)

            And now an example with dimensions:

            >>> homework_due_tomorrow = pl.DataFrame({"project": ["A", "B", "C"], "cost_per_hour_underdelivered": [10, 20, 30], "hours_to_finish": [9, 9, 9], "max_underdelivered": [1, 9, 9]})
            >>> m.hours_spent = pf.Variable(homework_due_tomorrow[["project"]], lb=0)
            >>> m.must_finish_project = (m.hours_spent >= homework_due_tomorrow[["project", "hours_to_finish"]]).relax(homework_due_tomorrow[["project", "cost_per_hour_underdelivered"]], max=homework_due_tomorrow[["project", "max_underdelivered"]])
            >>> m.only_one_day = sum("project", m.hours_spent) <= 24
            >>> # Relaxing a constraint after it has already been assigned will give an error
            >>> m.only_one_day.relax(1)
            Traceback (most recent call last):
            ...
            ValueError: .relax() must be called before the Constraint is added to the model
            >>> m.attr.Silent = True
            >>> m.optimize()
            >>> m.maximize.value
            -50.0
            >>> m.hours_spent.solution
            shape: (3, 2)
            ┌─────────┬──────────┐
            │ project ┆ solution │
            │ ---     ┆ ---      │
            │ str     ┆ f64      │
            ╞═════════╪══════════╡
            │ A       ┆ 8.0      │
            │ B       ┆ 7.0      │
            │ C       ┆ 9.0      │
            └─────────┴──────────┘
        """
        if self._has_ids:
            raise ValueError(
                ".relax() must be called before the Constraint is added to the model"
            )

        m = self._model
        if m is None or self.name is None:
            self.to_relax = FuncArgs(args=[cost, max])
            return self

        var_name = f"{self.name}_relaxation"
        assert not hasattr(
            m, var_name
        ), "Conflicting names, relaxation variable already exists on the model."
        var = Variable(self, lb=0, ub=max)
        setattr(m, var_name, var)

        if self.sense == ConstraintSense.LE:
            self.lhs -= var
        elif self.sense == ConstraintSense.GE:
            self.lhs += var
        else:  # pragma: no cover
            # TODO
            raise NotImplementedError(
                "Relaxation for equalities has not yet been implemented. Submit a pull request!"
            )

        penalty = var * cost
        if self.dimensions:
            penalty = sum(self.dimensions, penalty)
        if m.sense is None:
            raise ValueError(
                "Cannot relax a constraint before the objective sense has been set. Try setting the objective first or using Model(sense=...)."
            )
        elif m.sense == ObjSense.MAX:
            penalty *= -1
        if m.objective is None:
            m.objective = penalty
        else:
            m.objective += penalty

        return self

    def to_str(self) -> str:
        dims = self.dimensions
        str_table = self.lhs.to_str_table(include_const_term=False)
        str_table = self.lhs.to_str_create_prefix(str_table)
        rhs = self.lhs.constant_terms.with_columns(pl.col(COEF_KEY) * -1)
        rhs = cast_coef_to_string(rhs, drop_ones=False)
        # Remove leading +
        rhs = rhs.with_columns(pl.col(COEF_KEY).str.strip_chars(characters=" +"))
        rhs = rhs.rename({COEF_KEY: "rhs"})
        constr_str = pl.concat(
            [str_table, rhs], how=("align" if dims else "horizontal")
        )
        constr_str = constr_str.select(
            pl.concat_str("expr", pl.lit(f" {self.sense.value} "), "rhs").str.concat(
                delimiter="\n"
            )
        ).item()
        return constr_str

    def __repr__(self) -> str:
        return (
            get_obj_repr(
                self,
                ("name",),
                sense=f"'{self.sense.value}'",
                size=len(self),
                dimensions=self.shape,
                terms=len(self.lhs.data),
            )
            + "\n"
            + self.to_str()
        )


class Variable(ModelElementWithId, SupportsMath, SupportPolarsMethodMixin):
    """
    Represents one or many decision variable in an optimization model.

    Parameters:
        *indexing_sets:
            If no indexing_sets are provided, a single variable with no dimensions is created.
            Otherwise, a variable is created for each element in the Cartesian product of the indexing_sets (see Set for details on behaviour).
        lb:
            The lower bound for all variables.
        ub:
            The upper bound for all variables.
        vtype:
            The type of the variable. Can be either a VType enum or a string. Default is VType.CONTINUOUS.
        equals:
            When specified, a variable is created and a constraint is added to make the variable equal to the provided expression.

    Examples:
        >>> import pandas as pd
        >>> m = pf.Model()
        >>> df = pd.DataFrame({"dim1": [1, 1, 2, 2, 3, 3], "dim2": ["a", "b", "a", "b", "a", "b"]})
        >>> v = Variable(df)
        >>> v
        <Variable size=6 dimensions={'dim1': 3, 'dim2': 2} added_to_model=False>

        Variables cannot be used until they're added to the model.

        >>> m.constraint = v <= 3
        Traceback (most recent call last):
        ...
        ValueError: Cannot use 'Variable' before it has beed added to a model.
        >>> m.v = v
        >>> m.constraint = m.v <= 3

        >>> m.v
        <Variable name=v size=6 dimensions={'dim1': 3, 'dim2': 2}>
        [1,a]: v[1,a]
        [1,b]: v[1,b]
        [2,a]: v[2,a]
        [2,b]: v[2,b]
        [3,a]: v[3,a]
        [3,b]: v[3,b]
        >>> m.v2 = Variable(df[["dim1"]])
        Traceback (most recent call last):
        ...
        ValueError: Duplicate rows found in input data.
        >>> m.v3 = Variable(df[["dim1"]].drop_duplicates())
        >>> m.v3
        <Variable name=v3 size=3 dimensions={'dim1': 3}>
        [1]: v3[1]
        [2]: v3[2]
        [3]: v3[3]
    """

    # TODO: Breaking change, remove support for Iterable[AcceptableSets]
    def __init__(
        self,
        *indexing_sets: SetTypes | Iterable[SetTypes],
        lb: float | int | SupportsToExpr | None = None,
        ub: float | int | SupportsToExpr | None = None,
        vtype: VType | VTypeValue = VType.CONTINUOUS,
        equals: Optional[SupportsMath] = None,
    ):
        if equals is not None:
            assert (
                len(indexing_sets) == 0
            ), "Cannot specify both 'equals' and 'indexing_sets'"
            indexing_sets = (equals,)

        data = Set(*indexing_sets).data if len(indexing_sets) > 0 else pl.DataFrame()
        super().__init__(data)

        self.vtype: VType = VType(vtype)
        self.attr = Container(self._set_attribute, self._get_attribute)
        self._equals = equals

        if lb is not None and not isinstance(lb, (float, int)):
            self._lb_expr, self.lb = lb, None
        else:
            self._lb_expr, self.lb = None, lb
        if ub is not None and not isinstance(ub, (float, int)):
            self._ub_expr, self.ub = ub, None
        else:
            self._ub_expr, self.ub = None, ub

    def _set_attribute(self, name, value):
        self._assert_has_ids()
        col_name = name
        try:
            name = poi.VariableAttribute[name]
            setter = self._model.poi.set_variable_attribute
        except KeyError:
            setter = self._model.poi.set_variable_raw_attribute

        if self.dimensions is None:
            for key in self.data.get_column(VAR_KEY):
                setter(poi.VariableIndex(key), name, value)
        else:
            for key, v in (
                self.data.join(value, on=self.dimensions, how="inner")
                .select(pl.col(VAR_KEY), pl.col(col_name))
                .iter_rows()
            ):
                setter(poi.VariableIndex(key), name, v)

    @unwrap_single_values
    def _get_attribute(self, name):
        self._assert_has_ids()
        col_name = name
        try:
            name = poi.VariableAttribute[name]
            getter = self._model.poi.get_variable_attribute
        except KeyError:
            getter = self._model.poi.get_variable_raw_attribute

        with (
            warnings.catch_warnings()
        ):  # map_elements without return_dtype= gives a warning
            warnings.filterwarnings(
                action="ignore", category=pl.exceptions.MapWithoutReturnDtypeWarning
            )
            return self.data.with_columns(
                pl.col(VAR_KEY)
                .map_elements(lambda v_id: getter(poi.VariableIndex(v_id), name))
                .alias(col_name)
            ).select(self.dimensions_unsafe + [col_name])

    def _assign_ids(self):
        kwargs = dict(domain=self.vtype.to_poi())
        if self.lb is not None:
            kwargs["lb"] = self.lb
        if self.ub is not None:
            kwargs["ub"] = self.ub

        if self.dimensions is not None and self._model.use_var_names:
            df = (
                concat_dimensions(self.data, prefix=self.name)
                .with_columns(
                    pl.col("concated_dim")
                    .map_elements(
                        lambda name: self._model.poi.add_variable(
                            name=name, **kwargs
                        ).index,
                        return_dtype=KEY_TYPE,
                    )
                    .alias(VAR_KEY)
                )
                .drop("concated_dim")
            )
        else:
            if self._model.use_var_names:
                kwargs["name"] = self.name

            df = self.data.with_columns(
                pl.lit(0).alias(VAR_KEY).cast(KEY_TYPE)
            ).with_columns(
                pl.col(VAR_KEY).map_elements(
                    lambda _: self._model.poi.add_variable(**kwargs).index,
                    return_dtype=KEY_TYPE,
                )
            )

        self._data = df

    def on_add_to_model(self, model, name):
        super().on_add_to_model(model, name)
        self._assign_ids()
        if self._lb_expr is not None:
            setattr(model, f"{name}_lb", self._lb_expr <= self)

        if self._ub_expr is not None:
            setattr(model, f"{name}_ub", self <= self._ub_expr)

        if self._equals is not None:
            setattr(model, f"{name}_equals", self == self._equals)

    @classmethod
    def get_id_column_name(cls):
        return VAR_KEY

    @property
    @unwrap_single_values
    def solution(self):
        solution = self.attr.Value
        if isinstance(solution, pl.DataFrame):
            solution = solution.rename({"Value": SOLUTION_KEY})
        return solution

    def __repr__(self):
        if self._has_ids:
            return (
                get_obj_repr(
                    self,
                    ("name", "lb", "ub"),
                    size=self.data.height,
                    dimensions=self.shape,
                )
                + "\n"
                + self.to_expr().to_str()
            )
        else:
            return get_obj_repr(
                self,
                ("name", "lb", "ub"),
                size=self.data.height,
                dimensions=self.shape,
                added_to_model=False,
            )

    def to_expr(self) -> Expression:
        self._assert_has_ids()
        if POLARS_VERSION.major < 1:
            return self._new(self.data.drop(SOLUTION_KEY))
        else:
            return self._new(self.data.drop(SOLUTION_KEY, strict=False))

    def _new(self, data: pl.DataFrame):
        self._assert_has_ids()
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
            >>> time_dim = pd.DataFrame({"time": ["00:00", "06:00", "12:00", "18:00"]})
            >>> space_dim = pd.DataFrame({"city": ["Toronto", "Berlin"]})
            >>> m = pf.Model()
            >>> m.bat_charge = pf.Variable(time_dim, space_dim)
            >>> m.bat_flow = pf.Variable(time_dim, space_dim)
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

        if POLARS_VERSION.major < 1:
            data = data.join(
                wrapped, left_on="__prev", right_on="__next", how="inner"
            ).drop(["__prev", "__next"])
        else:
            data = data.join(
                wrapped, left_on="__prev", right_on="__next", how="inner"
            ).drop(["__prev", "__next"], strict=False)
        return expr._new(data)
