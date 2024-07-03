from __future__ import annotations
from typing import (
    Iterable,
    List,
    Mapping,
    Protocol,
    Sequence,
    overload,
    Union,
    Optional,
    TYPE_CHECKING,
)
from abc import ABC, abstractmethod

import pandas as pd
import polars as pl

from pyoframe._arithmetic import _add_expressions, _get_dimensions
from pyoframe.constants import (
    COEF_KEY,
    CONST_TERM,
    CONSTRAINT_KEY,
    DUAL_KEY,
    RESERVED_COL_KEYS,
    SLACK_COL,
    VAR_KEY,
    SOLUTION_KEY,
    RC_COL,
    VType,
    VTypeValue,
    Config,
    ConstraintSense,
    UnmatchedStrategy,
    PyoframeError,
    ObjSense,
)
from pyoframe.util import (
    cast_coef_to_string,
    concat_dimensions,
    get_obj_repr,
    parse_inputs_as_iterable,
    unwrap_single_values,
    dataframe_to_tupled_list,
    FuncArgs,
)

from pyoframe.model_element import (
    ModelElement,
    ModelElementWithId,
    SupportPolarsMethodMixin,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Model

VAR_TYPE = pl.UInt32


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

    def __neg__(self):
        res = self.to_expr() * -1
        # Negating a constant term should keep the unmatched strategy
        res.unmatched_strategy = self.unmatched_strategy
        return res

    def __sub__(self, other):
        """
        >>> import polars as pl
        >>> from pyoframe import Variable
        >>> df = pl.DataFrame({"dim1": [1,2,3], "value": [1,2,3]})
        >>> var = Variable(df["dim1"])
        >>> var - df
        <Expression size=3 dimensions={'dim1': 3} terms=6>
        [1]: x1 -1
        [2]: x2 -2
        [3]: x3 -3
        """
        if not isinstance(other, (int, float)):
            other = other.to_expr()
        return self.to_expr() + (-other)

    def __rmul__(self, other):
        return self.to_expr() * other

    def __radd__(self, other):
        return self.to_expr() + other

    def __le__(self, other):
        """Equality constraint.
        Examples
        >>> from pyoframe import Variable
        >>> Variable() <= 1
        <Constraint sense='<=' size=1 dimensions={} terms=2>
        x1 <= 1
        """
        return Constraint(self - other, ConstraintSense.LE)

    def __ge__(self, other):
        """Equality constraint.
        Examples
        >>> from pyoframe import Variable
        >>> Variable() >= 1
        <Constraint sense='>=' size=1 dimensions={} terms=2>
        x1 >= 1
        """
        return Constraint(self - other, ConstraintSense.GE)

    def __eq__(self, value: object):
        """Equality constraint.
        Examples
        >>> from pyoframe import Variable
        >>> Variable() == 1
        <Constraint sense='=' size=1 dimensions={} terms=2>
        x1 = 1
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
            except pl.ShapeError as e:
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
    """A linear expression."""

    def __init__(self, data: pl.DataFrame):
        """
        >>> import pandas as pd
        >>> from pyoframe import Variable, Model
        >>> df = pd.DataFrame({"item" : [1, 1, 1, 2, 2], "time": ["mon", "tue", "wed", "mon", "tue"], "cost": [1, 2, 3, 4, 5]}).set_index(["item", "time"])
        >>> m = Model("min")
        >>> m.Time = Variable(df.index)
        >>> m.Size = Variable(df.index)
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

    def sum(self, over: Union[str, Iterable[str]]):
        """
        Examples:
            >>> import pandas as pd
            >>> from pyoframe import Variable
            >>> df = pd.DataFrame({"item" : [1, 1, 1, 2, 2], "time": ["mon", "tue", "wed", "mon", "tue"], "cost": [1, 2, 3, 4, 5]}).set_index(["item", "time"])
            >>> quantity = Variable(df.reset_index()[["item"]].drop_duplicates())
            >>> expr = (quantity * df["cost"]).sum("time")
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
            .group_by(remaining_dims + [VAR_KEY], maintain_order=True)
            .sum()
        )

    def map(self, mapping_set: SetTypes, drop_shared_dims: bool = True):
        """
        Replaces the dimensions that are shared with mapping_set with the other dimensions found in mapping_set.

        This is particularly useful to go from one type of dimensions to another. For example, to convert data that
        is indexed by city to data indexed by country (see example).

        Parameters:
            mapping_set : SetTypes
                The set to map the expression to. This can be a DataFrame, Index, or another Set.
            drop_shared_dims : bool, default True
                If True, the dimensions shared between the expression and the mapping set are dropped from the resulting expression and
                    repeated rows are summed.
                If False, the shared dimensions are kept in the resulting expression.

        Returns:
            Expression
                A new Expression containing the result of the mapping operation.

        Examples:

        >>> import polars as pl
        >>> from pyoframe import Variable, Model
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

    def rolling_sum(self, over: str, window_size: int):
        """
        Calculates the rolling sum of the Expression over a specified window size for a given dimension.

        This method applies a rolling sum operation over the dimension specified by `over`,
        using a window defined by `window_size`.


        Parameters:
            over : str
                The name of the dimension (column) over which the rolling sum is calculated.
                This dimension must exist within the Expression's dimensions.
            window_size : int
                The size of the moving window in terms of number of records.
                The rolling sum is calculated over this many consecutive elements.

        Returns:
            Expression
                A new Expression instance containing the result of the rolling sum operation.
                This new Expression retains all dimensions (columns) of the original data,
                with the rolling sum applied over the specified dimension.

        Examples:
            >>> import polars as pl
            >>> from pyoframe import Variable, Model
            >>> cost = pl.DataFrame({"item" : [1, 1, 1, 2, 2], "time": [1, 2, 3, 1, 2], "cost": [1, 2, 3, 4, 5]})
            >>> m = Model("min")
            >>> m.quantity = Variable(cost[["item", "time"]])
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
                        index_column=over, period=f"{window_size}i", by=remaining_dims
                    )
                ]
            )
        )

    def within(self, set: "SetTypes") -> Expression:
        """
        Examples
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

    def __add__(self, other):
        """
        Examples:
            >>> import pandas as pd
            >>> from pyoframe import Variable
            >>> add = pd.DataFrame({"dim1": [1,2,3], "add": [10, 20, 30]}).to_expr()
            >>> var = Variable(add)
            >>> var + add
            <Expression size=3 dimensions={'dim1': 3} terms=6>
            [1]: x1 +10
            [2]: x2 +20
            [3]: x3 +30
            >>> var + add + 2
            <Expression size=3 dimensions={'dim1': 3} terms=6>
            [1]: x1 +12
            [2]: x2 +22
            [3]: x3 +32
            >>> var + pd.DataFrame({"dim1": [1,2], "add": [10, 20]})
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
            >>> 5 + 2 * Variable()
            <Expression size=1 dimensions={} terms=2>
            2 x4 +5
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

        if (other.data.get_column(VAR_KEY) != CONST_TERM).any():
            self, other = other, self

        if (other.data.get_column(VAR_KEY) != CONST_TERM).any():
            raise ValueError(
                "Multiplication of two expressions with variables is non-linear and not supported."
            )
        multiplier = other.data.drop(VAR_KEY)

        dims = self.dimensions_unsafe
        other_dims = other.dimensions_unsafe
        dims_in_common = [dim for dim in dims if dim in other_dims]

        data = (
            self.data.join(
                multiplier,
                on=dims_in_common,
                how="inner" if dims_in_common else "cross",
            )
            .with_columns(pl.col(COEF_KEY) * pl.col(COEF_KEY + "_right"))
            .drop(COEF_KEY + "_right")
        )

        return self._new(data)

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
        dim = self.dimensions
        data = self.data
        # Fill in missing constant terms
        if not dim:
            if CONST_TERM not in data[VAR_KEY]:
                data = pl.concat(
                    [
                        data,
                        pl.DataFrame(
                            {COEF_KEY: [0.0], VAR_KEY: [CONST_TERM]},
                            schema={COEF_KEY: pl.Float64, VAR_KEY: VAR_TYPE},
                        ),
                    ],
                    how="vertical_relaxed",
                )
        else:
            keys = (
                data.select(dim)
                .unique(maintain_order=True)
                .with_columns(pl.lit(CONST_TERM).alias(VAR_KEY).cast(VAR_TYPE))
            )
            data = data.join(keys, on=dim + [VAR_KEY], how="outer_coalesce")
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
        if dims is not None:
            return constant_terms.join(
                self.data.select(dims).unique(maintain_order=True),
                on=dims,
                how="outer_coalesce",
            ).with_columns(pl.col(COEF_KEY).fill_null(0.0))
        else:
            if len(constant_terms) == 0:
                return pl.DataFrame(
                    {COEF_KEY: [0.0], VAR_KEY: [CONST_TERM]},
                    schema={COEF_KEY: pl.Float64, VAR_KEY: VAR_TYPE},
                )
            return constant_terms

    @property
    def variable_terms(self):
        return self.data.filter(pl.col(VAR_KEY) != CONST_TERM)

    @property
    @unwrap_single_values
    def value(self) -> pl.DataFrame:
        """
        The value of the expression. Only available after the model has been solved.

        Examples:
            >>> import pyoframe as pf
            >>> m = pf.Model("max")
            >>> m.X = pf.Variable({"dim1": [1, 2, 3]}, ub=10)
            >>> m.expr_1 = 2 * m.X + 1
            >>> m.expr_2 = pf.sum(m.expr_1)
            >>> m.objective = m.expr_2 - 3
            >>> result = m.solve(log_to_console=False)
            >>> m.expr_1.value
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
            >>> m.expr_2.value
            63.0
        """
        assert (
            self._model is not None
        ), "Expression must be added to the model to use .value"
        if self._model.result is None or self._model.result.solution is None:
            raise ValueError(
                "Can't obtain value of expression since the model has not been solved."
            )

        df = (
            self.data.join(self._model.result.solution.primal, on=VAR_KEY, how="left")
            .with_columns(
                (
                    pl.when(pl.col(VAR_KEY) == CONST_TERM)
                    .then(1)
                    .otherwise(pl.col(SOLUTION_KEY))
                    * pl.col(COEF_KEY)
                ).alias(SOLUTION_KEY)
            )
            .drop(VAR_KEY)
            .drop(COEF_KEY)
        )

        dims = self.dimensions
        if dims is not None:
            df = df.group_by(dims, maintain_order=True)
        return df.sum()

    def to_str_table(
        self,
        max_line_len=None,
        max_rows=None,
        include_const_term=True,
        include_const_variable=False,
        var_map=None,
        float_precision=None,
    ):
        data = self.data if include_const_term else self.variable_terms
        data = cast_coef_to_string(data, float_precision=float_precision)

        if var_map is not None:
            data = var_map.apply(data, to_col="str_var")
        elif self._model is not None and self._model.var_map is not None:
            var_map = self._model.var_map
            data = var_map.apply(data, to_col="str_var")
        else:
            data = data.with_columns(
                pl.concat_str(pl.lit("x"), VAR_KEY).alias("str_var")
            )
        if include_const_variable:
            data = data.drop(VAR_KEY).rename({"str_var": VAR_KEY})
        else:
            data = data.with_columns(
                pl.when(pl.col(VAR_KEY) == CONST_TERM)
                .then(pl.lit(""))
                .otherwise("str_var")
                .alias(VAR_KEY)
            ).drop("str_var")

        dimensions = self.dimensions

        # Create a string for each term
        data = data.with_columns(
            expr=pl.concat_str(
                COEF_KEY,
                pl.lit(" "),
                VAR_KEY,
            )
        ).drop(COEF_KEY, VAR_KEY)

        # Combine terms into one string
        if dimensions is not None:
            data = data.group_by(dimensions, maintain_order=True).agg(
                pl.col("expr").str.concat(delimiter=" ")
            )
        else:
            data = data.select(pl.col("expr").str.concat(delimiter=" "))

        # Remove leading +
        data = data.with_columns(pl.col("expr").str.strip_chars(characters=" +"))

        # TODO add vertical ... if too many rows, in the middle of the table
        if max_rows:
            data = data.head(max_rows)

        if max_line_len:
            data = data.with_columns(
                pl.when(pl.col("expr").str.len_chars() > max_line_len)
                .then(
                    pl.concat_str(
                        pl.col("expr").str.slice(0, max_line_len),
                        pl.lit("..."),
                    )
                )
                .otherwise(pl.col("expr"))
            )
        return data

    def to_str_create_prefix(self, data):
        if self.name is not None or self.dimensions:
            data = concat_dimensions(data, prefix=self.name, ignore_columns=["expr"])
            data = data.with_columns(
                pl.concat_str("concated_dim", pl.lit(": "), "expr").alias("expr")
            ).drop("concated_dim")
        return data

    def to_str(
        self,
        max_line_len=None,
        max_rows=None,
        include_const_term=True,
        include_const_variable=False,
        var_map=None,
        include_prefix=True,
        include_header=False,
        include_data=True,
        float_precision=None,
    ):
        result = ""
        if include_header:
            result += get_obj_repr(
                self, size=len(self), dimensions=self.shape, terms=len(self.data)
            )
        if include_header and include_data:
            result += "\n"
        if include_data:
            str_table = self.to_str_table(
                max_line_len=max_line_len,
                max_rows=max_rows,
                include_const_term=include_const_term,
                include_const_variable=include_const_variable,
                var_map=var_map,
                float_precision=float_precision,
            )
            if include_prefix:
                str_table = self.to_str_create_prefix(str_table)
            result += str_table.select(pl.col("expr").str.concat(delimiter="\n")).item()

        return result

    def __repr__(self) -> str:
        return self.to_str(
            max_line_len=80,
            max_rows=15,
            include_header=True,
            float_precision=Config.print_float_precision,
        )

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
            lhs: Expression
                The left hand side of the constraint.
            sense: Sense
                The sense of the constraint.
        """
        self.lhs = lhs
        self._model = lhs._model
        self.sense = sense
        self.to_relax: Optional[FuncArgs] = None

        dims = self.lhs.dimensions
        data = pl.DataFrame() if dims is None else self.lhs.data.select(dims).unique()

        super().__init__(data)

    def on_add_to_model(self, model: "Model", name: str):
        super().on_add_to_model(model, name)
        if self.to_relax is not None:
            self.relax(*self.to_relax.args, **self.to_relax.kwargs)

    @property
    @unwrap_single_values
    def slack(self):
        """
        The slack of the constraint.
        Will raise an error if the model has not already been solved.
        The first call to this property will load the slack values from the solver (lazy loading).
        """
        if SLACK_COL not in self.data.columns:
            assert (
                self._model is not None
            ), "Constraint must be added to a model to get the slack."
            if self._model.solver is None:
                raise ValueError("The model has not been solved yet.")
            self._model.solver.load_slack()
        return self.data.select(self.dimensions_unsafe + [SLACK_COL])

    @slack.setter
    def slack(self, value):
        self._extend_dataframe_by_id(value)

    @property
    @unwrap_single_values
    def dual(self) -> Union[pl.DataFrame, float]:
        if DUAL_KEY not in self.data.columns:
            raise ValueError(f"No dual values founds for constraint '{self.name}'")
        return self.data.select(self.dimensions_unsafe + [DUAL_KEY])

    @dual.setter
    def dual(self, value):
        self._extend_dataframe_by_id(value)

    @classmethod
    def get_id_column_name(cls):
        return CONSTRAINT_KEY

    def to_str_create_prefix(self, data, const_map=None):
        if const_map is None:
            return self.lhs.to_str_create_prefix(data)

        data_map = const_map.apply(self.ids, to_col=None)

        if self.dimensions is None:
            assert data.height == 1
            prefix = data_map.select(pl.col(CONSTRAINT_KEY)).item()
            return data.select(
                pl.concat_str(pl.lit(f"{prefix}: "), "expr").alias("expr")
            )

        data = data.join(data_map, on=self.dimensions)
        return data.with_columns(
            pl.concat_str(CONSTRAINT_KEY, pl.lit(": "), "expr").alias("expr")
        ).drop(CONSTRAINT_KEY)

    def filter(self, *args, **kwargs) -> pl.DataFrame:
        return self.lhs.data.filter(*args, **kwargs)

    def relax(
        self, cost: SupportsToExpr, max: Optional[SupportsToExpr] = None
    ) -> Constraint:
        """
        Relaxes the constraint by adding a variable to the constraint that can be non-zero at a cost.

        Parameters:
            cost: SupportsToExpr
                The cost of relaxing the constraint. Costs should be positives as they will automatically
                become negative for maximization problems.
            max: SupportsToExpr, default None
                The maximum value of the relaxation variable.

        Returns:
            The same constraint

        Examples:
            >>> import pyoframe as pf
            >>> m = pf.Model("max")
            >>> homework_due_tomorrow = pl.DataFrame({"project": ["A", "B", "C"], "cost_per_hour_underdelivered": [10, 20, 30], "hours_to_finish": [9, 9, 9], "max_underdelivered": [1, 9, 9]})
            >>> m.hours_spent = pf.Variable(homework_due_tomorrow[["project"]], lb=0)
            >>> m.must_finish_project = m.hours_spent >= homework_due_tomorrow[["project", "hours_to_finish"]]
            >>> m.only_one_day = sum("project", m.hours_spent) <= 24
            >>> m.solve(log_to_console=False)
            Status: warning
            Termination condition: infeasible
            <BLANKLINE>

            >>> _ = m.must_finish_project.relax(homework_due_tomorrow[["project", "cost_per_hour_underdelivered"]], max=homework_due_tomorrow[["project", "max_underdelivered"]])
            >>> result = m.solve(log_to_console=False)
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


            >>> # It can also be done all in one go!
            >>> m = pf.Model("max")
            >>> homework_due_tomorrow = pl.DataFrame({"project": ["A", "B", "C"], "cost_per_hour_underdelivered": [10, 20, 30], "hours_to_finish": [9, 9, 9], "max_underdelivered": [1, 9, 9]})
            >>> m.hours_spent = pf.Variable(homework_due_tomorrow[["project"]], lb=0)
            >>> m.must_finish_project = (m.hours_spent >= homework_due_tomorrow[["project", "hours_to_finish"]]).relax(5)
            >>> m.only_one_day = (sum("project", m.hours_spent) <= 24).relax(1)
            >>> _ = m.solve(log_to_console=False)
            >>> m.objective.value
            -3.0
            >>> m.hours_spent.solution
            shape: (3, 2)
            ┌─────────┬──────────┐
            │ project ┆ solution │
            │ ---     ┆ ---      │
            │ str     ┆ f64      │
            ╞═════════╪══════════╡
            │ A       ┆ 9.0      │
            │ B       ┆ 9.0      │
            │ C       ┆ 9.0      │
            └─────────┴──────────┘
        """
        m = self._model
        if m is None or self.name is None:
            self.to_relax = FuncArgs(args=[cost, max])
            return self

        var_name = f"{self.name}_relaxation"
        assert not hasattr(
            m, var_name
        ), "Conflicting names, relaxation variable already exists on the model."
        var = Variable(self, lb=0, ub=max)

        if self.sense == ConstraintSense.LE:
            self.lhs -= var
        elif self.sense == ConstraintSense.GE:
            self.lhs += var
        else:  # pragma: no cover
            # TODO
            raise NotImplementedError(
                "Relaxation for equalities has not yet been implemented. Submit a pull request!"
            )

        setattr(m, var_name, var)
        penalty = var * cost
        if self.dimensions:
            penalty = sum(self.dimensions, penalty)
        if m.sense == ObjSense.MAX:
            penalty *= -1
        if m.objective is None:
            m.objective = penalty
        else:
            m.objective += penalty

        return self

    def to_str(
        self,
        max_line_len=None,
        max_rows=None,
        var_map=None,
        float_precision=None,
        const_map=None,
    ) -> str:
        dims = self.dimensions
        str_table = self.lhs.to_str_table(
            max_line_len=max_line_len,
            max_rows=max_rows,
            include_const_term=False,
            var_map=var_map,
        )
        str_table = self.to_str_create_prefix(str_table, const_map=const_map)
        rhs = self.lhs.constant_terms.with_columns(pl.col(COEF_KEY) * -1)
        rhs = cast_coef_to_string(rhs, drop_ones=False, float_precision=float_precision)
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
            + self.to_str(max_line_len=80, max_rows=15)
        )


class Variable(ModelElementWithId, SupportsMath, SupportPolarsMethodMixin):
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
        equals: SupportsToExpr
            When specified, a variable is created and a constraint is added to make the variable equal to the provided expression.

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
        lb: float | int | SupportsToExpr | None = None,
        ub: float | int | SupportsToExpr | None = None,
        vtype: VType | VTypeValue = VType.CONTINUOUS,
        equals: Optional[SupportsMath] = None,
    ):
        if lb is None:
            lb = float("-inf")
        if ub is None:
            ub = float("inf")
        if equals is not None:
            assert (
                len(indexing_sets) == 0
            ), "Cannot specify both 'equals' and 'indexing_sets'"
            indexing_sets = (equals,)

        data = Set(*indexing_sets).data if len(indexing_sets) > 0 else pl.DataFrame()
        super().__init__(data)

        self.vtype: VType = VType(vtype)
        self._equals = equals

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
        if self._equals is not None:
            setattr(model, f"{name}_equals", self == self._equals)

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
            assert (
                self._model is not None
            ), "Variable must be added to a model to get the reduced cost."
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
        return self._new(self.data.drop(SOLUTION_KEY, strict=False))

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
            >>> m = Model("min")
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
        ).drop(["__prev", "__next"], strict=False)
        return expr._new(data)
