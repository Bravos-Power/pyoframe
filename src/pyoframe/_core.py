"""Defines several core Pyoframe objects including Set, Constraint, Variable, and Expression."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Union, overload

import pandas as pd
import polars as pl
import pyoptinterface as poi

from pyoframe._arithmetic import (
    _get_dimensions,
    _simplify_expr_df,
    add,
    multiply,
)
from pyoframe._constants import (
    COEF_KEY,
    CONST_TERM,
    CONSTRAINT_KEY,
    DUAL_KEY,
    QUAD_VAR_KEY,
    RESERVED_COL_KEYS,
    SOLUTION_KEY,
    VAR_KEY,
    Config,
    ConstraintSense,
    ExtrasStrategy,
    ObjSense,
    PyoframeError,
    VType,
    VTypeValue,
)
from pyoframe._model_element import BaseBlock
from pyoframe._utils import (
    Container,
    FuncArgs,
    cast_coef_to_string,
    concat_dimensions,
    get_obj_repr,
    pairwise,
    parse_inputs_as_iterable,
    return_new,
    unwrap_single_values,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe._model import Model

Operable = Union["BaseOperableBlock", pl.DataFrame, pd.DataFrame, pd.Series, int, float]
"""Any of the following objects: `int`, `float`, [Variable][pyoframe.Variable], [Expression][pyoframe.Expression], [Set][pyoframe.Set], polars or pandas DataFrame, or pandas Series."""


class BaseOperableBlock(BaseBlock):
    """Any object that can be converted into an expression."""

    def __init__(self, *args, **kwargs):
        self._extras_strategy = ExtrasStrategy.UNSET
        self._allowed_new_dims: list[str] = []
        super().__init__(*args, **kwargs)

    @abstractmethod
    def _new(self, data: pl.DataFrame, name: str) -> BaseOperableBlock:
        """Helper method to create a new instance of the same (or for Variable derivative) class."""

    def _copy_flags(self, other: BaseOperableBlock):
        """Copies the flags from another BaseOperableBlock object."""
        self._extras_strategy = other._extras_strategy
        self._allowed_new_dims = other._allowed_new_dims.copy()

    def keep_extras(self):
        """Indicates that labels not present in the other expression should be kept during addition, subtraction, or constraint creation.

        [Learn more](../../learn/concepts/addition.md) about addition modifiers.

        See Also:
            [`drop_extras`][pyoframe.Expression.drop_extras].
        """
        new = self._new(self.data, name=f"{self.name}.keep_extras()")
        new._copy_flags(self)
        new._extras_strategy = ExtrasStrategy.KEEP
        return new

    def drop_extras(self):
        """Indicates that labels not present in the other expression should be discarded during addition, subtraction, or constraint creation.

        [Learn more](../../learn/concepts/addition.md) about addition modifiers.

        See Also:
            [`keep_extras`][pyoframe.Expression.keep_extras].
        """
        new = self._new(self.data, name=f"{self.name}.drop_extras()")
        new._copy_flags(self)
        new._extras_strategy = ExtrasStrategy.DROP
        return new

    def keep_unmatched(self):  # pragma: no cover
        """Deprecated, use [`keep_extras`][pyoframe.Expression.keep_extras] instead."""
        warnings.warn(
            "'keep_unmatched' has been renamed to 'keep_extras'. Please use 'keep_extras' instead.",
            DeprecationWarning,
        )
        return self.keep_extras()

    def drop_unmatched(self):  # pragma: no cover
        """Deprecated, use [`drop_extras`][pyoframe.Expression.drop_extras] instead."""
        warnings.warn(
            "'drop_unmatched' has been renamed to 'drop_extras'. Please use 'drop_extras' instead.",
            DeprecationWarning,
        )
        return self.drop_extras()

    def raise_extras(self):
        """Indicates that labels not present in the other expression should raise an error during addition, subtraction, or constraint creation.

        This is the default behavior and, as such, this addition modifier should only be used in the rare cases where you want to override a previous use of `keep_extras()` or `drop_extras()`.

        [Learn more](../../learn/concepts/addition.md) about addition modifiers.

        See Also:
            [`keep_extras`][pyoframe.Expression.keep_extras] and [`drop_extras`][pyoframe.Expression.drop_extras].
        """
        new = self._new(self.data, name=f"{self.name}.raise_extras()")
        new._copy_flags(self)
        new._extras_strategy = ExtrasStrategy.UNSET
        return new

    def over(self, *dims: str):
        """Indicates that the expression can be broadcasted over the given dimensions during addition and subtraction."""
        new = self._new(self.data, name=f"{self.name}.over(…)")
        new._copy_flags(self)
        new._allowed_new_dims.extend(dims)
        return new

    @return_new
    def rename(self, *args, **kwargs):
        """Renames one or several of the object's dimensions.

        Takes the same arguments as [`polars.DataFrame.rename`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.rename.html).

        See the [portfolio optimization example](../../examples/portfolio_optimization.md) for a usage example.

        Examples:
            >>> m = pf.Model()
            >>> m.v = pf.Variable(
            ...     {"hour": ["00:00", "06:00", "12:00", "18:00"]},
            ...     {"city": ["Toronto", "Berlin", "Paris"]},
            ... )
            >>> m.v
            <Variable 'v' height=12>
            ┌───────┬─────────┬──────────────────┐
            │ hour  ┆ city    ┆ variable         │
            │ (4)   ┆ (3)     ┆                  │
            ╞═══════╪═════════╪══════════════════╡
            │ 00:00 ┆ Toronto ┆ v[00:00,Toronto] │
            │ 00:00 ┆ Berlin  ┆ v[00:00,Berlin]  │
            │ 00:00 ┆ Paris   ┆ v[00:00,Paris]   │
            │ 06:00 ┆ Toronto ┆ v[06:00,Toronto] │
            │ 06:00 ┆ Berlin  ┆ v[06:00,Berlin]  │
            │ …     ┆ …       ┆ …                │
            │ 12:00 ┆ Berlin  ┆ v[12:00,Berlin]  │
            │ 12:00 ┆ Paris   ┆ v[12:00,Paris]   │
            │ 18:00 ┆ Toronto ┆ v[18:00,Toronto] │
            │ 18:00 ┆ Berlin  ┆ v[18:00,Berlin]  │
            │ 18:00 ┆ Paris   ┆ v[18:00,Paris]   │
            └───────┴─────────┴──────────────────┘

            >>> m.v.rename({"city": "location"})
            <Expression (linear) height=12 terms=12>
            ┌───────┬──────────┬──────────────────┐
            │ hour  ┆ location ┆ expression       │
            │ (4)   ┆ (3)      ┆                  │
            ╞═══════╪══════════╪══════════════════╡
            │ 00:00 ┆ Toronto  ┆ v[00:00,Toronto] │
            │ 00:00 ┆ Berlin   ┆ v[00:00,Berlin]  │
            │ 00:00 ┆ Paris    ┆ v[00:00,Paris]   │
            │ 06:00 ┆ Toronto  ┆ v[06:00,Toronto] │
            │ 06:00 ┆ Berlin   ┆ v[06:00,Berlin]  │
            │ …     ┆ …        ┆ …                │
            │ 12:00 ┆ Berlin   ┆ v[12:00,Berlin]  │
            │ 12:00 ┆ Paris    ┆ v[12:00,Paris]   │
            │ 18:00 ┆ Toronto  ┆ v[18:00,Toronto] │
            │ 18:00 ┆ Berlin   ┆ v[18:00,Berlin]  │
            │ 18:00 ┆ Paris    ┆ v[18:00,Paris]   │
            └───────┴──────────┴──────────────────┘

        """
        return self.data.rename(*args, **kwargs)

    @return_new
    def with_columns(self, *args, **kwargs):
        """Creates a new object with modified columns.

        Takes the same arguments as [`polars.DataFrame.with_columns`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_columns.html).

        !!! warning
            Only use this function if you know what you're doing. It is not recommended to manually modify the columns
            within a Pyoframe object.
        """
        return self.data.with_columns(*args, **kwargs)

    @return_new
    def filter(self, *args, **kwargs):
        """Creates a copy of the object containing only a subset of the original rows.

        Takes the same arguments as [`polars.DataFrame.filter`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.filter.html).

        See Also:
            [`Expression.pick`][pyoframe.Expression.pick] or [`Variable.pick`][pyoframe.Variable.pick] if you wish to drop the filtered
            column in the process.

        """
        return self.data.filter(*args, **kwargs)

    @return_new
    def pick(self, **kwargs):
        """Filters elements by the given criteria and then drops the filtered dimensions.

        Examples:
            >>> m = pf.Model()
            >>> m.v = pf.Variable(
            ...     [
            ...         {"hour": ["00:00", "06:00", "12:00", "18:00"]},
            ...         {"city": ["Toronto", "Berlin", "Paris"]},
            ...     ]
            ... )
            >>> m.v.pick(hour="06:00")
            <Expression (linear) height=3 terms=3>
            ┌─────────┬──────────────────┐
            │ city    ┆ expression       │
            │ (3)     ┆                  │
            ╞═════════╪══════════════════╡
            │ Toronto ┆ v[06:00,Toronto] │
            │ Berlin  ┆ v[06:00,Berlin]  │
            │ Paris   ┆ v[06:00,Paris]   │
            └─────────┴──────────────────┘
            >>> m.v.pick(hour="06:00", city="Toronto")
            <Expression (linear) terms=1>
            v[06:00,Toronto]

        See Also:
            [`Expression.filter`][pyoframe.Expression.filter] or [`Variable.filter`][pyoframe.Variable.filter] if you don't wish to drop the filtered column.
        """
        return self.data.filter(**kwargs).drop(kwargs.keys())

    def _add_allowed_new_dims_to_df(self, df):
        cols = df.columns
        df = df.with_columns(*(pl.lit("*").alias(c) for c in self._allowed_new_dims))
        df = df.select(cols[:-1] + self._allowed_new_dims + [cols[-1]])  # reorder
        return df

    def add_dim(self, *dims: str):  # pragma: no cover
        """Deprecated, use [`over`][pyoframe.Expression.over] instead."""
        warnings.warn(
            "'add_dim' has been renamed to 'over'. Please use 'over' instead.",
            DeprecationWarning,
        )
        return self.over(*dims)

    @abstractmethod
    def to_expr(self) -> Expression:
        """Converts the object to a Pyoframe Expression."""
        ...

    def sum(self, *args, **kwargs):
        """Converts the object to an expression (see `.to_expr()`) and then applies [`Expression.sum`][pyoframe.Expression.sum]."""
        return self.to_expr().sum(*args, **kwargs)

    def sum_by(self, *args, **kwargs):
        """Converts the object to an expression (see `.to_expr()`) and then applies [`Expression.sum_by`][pyoframe.Expression.sum_by]."""
        return self.to_expr().sum_by(*args, **kwargs)

    def map(self, *args, **kwargs):
        """Converts the object to an expression (see `.to_expr()`) and then applies [`Expression.map`][pyoframe.Expression.map]."""
        return self.to_expr().map(*args, **kwargs)

    def __add__(self, *args, **kwargs):
        return self.to_expr().__add__(*args, **kwargs)

    def __mul__(self, *args, **kwargs):
        return self.to_expr().__mul__(*args, **kwargs)

    def __pow__(self, power: int):
        """Supports squaring expressions.

        Examples:
            >>> m = pf.Model()
            >>> m.v = pf.Variable()
            >>> m.v**2
            <Expression (quadratic) terms=1>
            v * v
            >>> m.v**3
            Traceback (most recent call last):
            ...
            ValueError: Raising an expressions to **3 is not supported. Expressions can only be squared (**2).
        """
        if power == 2:
            res = self * self
            res.name = f"({self.name}**2)"
            return res
        raise ValueError(
            f"Raising an expressions to **{power} is not supported. Expressions can only be squared (**2)."
        )

    def __neg__(self):
        res = self.to_expr() * -1
        res.name = f"-{self.name}"
        res._copy_flags(self)
        return res

    def __sub__(self, other):
        """Subtracts a value from this Expression.

        Examples:
            >>> import polars as pl
            >>> m = pf.Model()
            >>> df = pl.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, 3]})
            >>> m.v = pf.Variable(df["dim1"])
            >>> m.v - df
            <Expression (linear) height=3 terms=6>
            ┌──────┬────────────┐
            │ dim1 ┆ expression │
            │ (3)  ┆            │
            ╞══════╪════════════╡
            │ 1    ┆ v[1] -1    │
            │ 2    ┆ v[2] -2    │
            │ 3    ┆ v[3] -3    │
            └──────┴────────────┘
        """
        if not isinstance(other, (int, float)):
            other = other.to_expr()  # TODO don't rely on monkey patch
        return self.to_expr() + (-other)

    def __rmul__(self, other):
        return self.to_expr() * other

    def __radd__(self, other):
        return self.to_expr() + other

    def __truediv__(self, other):
        """Divides this expression.

        Examples:
            Support division.
            >>> m = pf.Model()
            >>> m.v = Variable({"dim1": [1, 2, 3]})
            >>> m.v / 2
            <Expression (linear) height=3 terms=3>
            ┌──────┬────────────┐
            │ dim1 ┆ expression │
            │ (3)  ┆            │
            ╞══════╪════════════╡
            │ 1    ┆ 0.5 v[1]   │
            │ 2    ┆ 0.5 v[2]   │
            │ 3    ┆ 0.5 v[3]   │
            └──────┴────────────┘
        """
        return self.to_expr() * (1 / other)

    def __rtruediv__(self, other):
        # This just improves error messages when trying to divide by a Set or Variable.
        # When dividing by an Expression, see the Expression.__rtruediv__ method.
        raise PyoframeError(
            f"Cannot divide by '{self.name}' because it is not a number or parameter."
        )

    def __rsub__(self, other):
        """Supports right subtraction.

        Examples:
            >>> m = pf.Model()
            >>> m.v = Variable({"dim1": [1, 2, 3]})
            >>> 1 - m.v
            <Expression (linear) height=3 terms=6>
            ┌──────┬────────────┐
            │ dim1 ┆ expression │
            │ (3)  ┆            │
            ╞══════╪════════════╡
            │ 1    ┆ 1 - v[1]   │
            │ 2    ┆ 1 - v[2]   │
            │ 3    ┆ 1 - v[3]   │
            └──────┴────────────┘
        """
        return other + (-self.to_expr())

    def __le__(self, other):
        return Constraint(self - other, ConstraintSense.LE)

    def __lt__(self, _):
        raise PyoframeError(
            "Constraints cannot be created with the '<' or '>' operators. Did you mean to use '<=' or '>=' instead?"
        )

    def __ge__(self, other):
        return Constraint(self - other, ConstraintSense.GE)

    def __gt__(self, _):
        raise PyoframeError(
            "Constraints cannot be created with the '<' or '>' operator. Did you mean to use '<=' or '>=' instead?"
        )

    def __eq__(self, value: object):  # type: ignore
        return Constraint(self - value, ConstraintSense.EQ)


SetTypes = Union[
    pl.DataFrame,
    pd.Index,
    pd.DataFrame,
    BaseOperableBlock,
    Mapping[str, Sequence[object]],
    "Set",
    "Constraint",
]


class Set(BaseOperableBlock):
    """A set which can then be used to index variables.

    Examples:
        >>> pf.Set(x=range(2), y=range(3))
        <Set 'unnamed' height=6>
        ┌─────┬─────┐
        │ x   ┆ y   │
        │ (2) ┆ (3) │
        ╞═════╪═════╡
        │ 0   ┆ 0   │
        │ 0   ┆ 1   │
        │ 0   ┆ 2   │
        │ 1   ┆ 0   │
        │ 1   ┆ 1   │
        │ 1   ┆ 2   │
        └─────┴─────┘
    """

    def __init__(self, *data: SetTypes | Iterable[SetTypes], **named_data):
        data_list = list(data)
        for name, set in named_data.items():
            data_list.append({name: set})
        df = self._parse_acceptable_sets(*data_list)
        if not df.is_empty() and df.is_duplicated().any():
            raise ValueError("Duplicate rows found in input data.")
        super().__init__(df, name="unnamed_set")

    def _new(self, data: pl.DataFrame, name: str) -> Set:
        s = Set(data)
        s.name = name
        s._model = self._model
        return s

    @staticmethod
    def _parse_acceptable_sets(
        *over: SetTypes | Iterable[SetTypes],
    ) -> pl.DataFrame:
        """Computes the cartesian product of the given sets.

        Examples:
            >>> import pandas as pd
            >>> dim1 = pd.Index([1, 2, 3], name="dim1")
            >>> dim2 = pd.Index(["a", "b"], name="dim1")
            >>> Set._parse_acceptable_sets([dim1, dim2])
            Traceback (most recent call last):
            ...
            AssertionError: Dimension 'dim1' is not unique.
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

        over_frames: list[pl.DataFrame] = [Set._set_to_polars(set) for set in over_iter]

        over_merged = over_frames[0]

        for df in over_frames[1:]:
            overlap_dims = set(over_merged.columns) & set(df.columns)
            assert not overlap_dims, (
                f"Dimension '{tuple(overlap_dims)[0]}' is not unique."
            )
            over_merged = over_merged.join(df, how="cross")
        return over_merged

    def to_expr(self) -> Expression:
        """Converts the Set to an Expression equal to 1 for each index.

        Useful when multiplying a Set by an Expression.
        """
        return Expression(
            self.data.with_columns(
                pl.lit(1).alias(COEF_KEY), pl.lit(CONST_TERM).alias(VAR_KEY)
            ),
            name=self.name,
        )

    def drop(self, *dims: str) -> Set:
        """Returns a new Set with the given dimensions dropped.

        Only unique rows are kept in the resulting Set.

        Examples:
            >>> xy = pf.Set(x=range(3), y=range(2))
            >>> xy
            <Set 'unnamed' height=6>
            ┌─────┬─────┐
            │ x   ┆ y   │
            │ (3) ┆ (2) │
            ╞═════╪═════╡
            │ 0   ┆ 0   │
            │ 0   ┆ 1   │
            │ 1   ┆ 0   │
            │ 1   ┆ 1   │
            │ 2   ┆ 0   │
            │ 2   ┆ 1   │
            └─────┴─────┘
            >>> x = xy.drop("y")
            >>> x
            <Set 'unnamed_set.drop(…)' height=3>
            ┌─────┐
            │ x   │
            │ (3) │
            ╞═════╡
            │ 0   │
            │ 1   │
            │ 2   │
            └─────┘
        """
        if not dims:
            raise ValueError("At least one dimension must be provided to drop.")
        return self._new(
            self.data.drop(dims).unique(maintain_order=Config.maintain_order),
            name=f"{self.name}.drop(…)",
        )

    def __mul__(self, other):
        if isinstance(other, Set):
            overlap_dims = set(self.data.columns) & set(other.data.columns)
            assert not overlap_dims, (
                f"Cannot multiply the two sets because dimension '{tuple(overlap_dims)[0]}' is present in both sets."
            )
            return Set(self.data, other.data)
        return super().__mul__(other)

    def __add__(self, other):
        # TODO replace with bitwise or
        if isinstance(other, Set):
            try:
                return self._new(
                    pl.concat([self.data, other.data]).unique(
                        maintain_order=Config.maintain_order
                    ),
                    name=f"({self.name} + {other.name})",
                )
            except pl.exceptions.ShapeError as e:
                if "unable to vstack, column names don't match" in str(e):
                    raise PyoframeError(
                        f"Failed to add sets '{self.name}' and '{other.name}' because dimensions do not match ({self.dimensions} != {other.dimensions}) "
                    ) from e
                raise e  # pragma: no cover

        return super().__add__(other)

    def __repr__(self):
        header = get_obj_repr(
            self,
            "'unnamed'" if self.name == "unnamed_set" else f"'{self.name}'",
            height=self.data.height,
        )
        data = self._add_shape_to_columns(self.data)
        data = self._add_allowed_new_dims_to_df(data)
        with Config.print_polars_config:
            table = repr(data)

        return header + "\n" + table

    @staticmethod
    def _set_to_polars(set: SetTypes) -> pl.DataFrame:
        if isinstance(set, dict):
            df = pl.DataFrame(set)
        elif isinstance(set, Constraint):
            df = set.data.select(set._dimensions_unsafe)
        elif isinstance(set, BaseOperableBlock):
            df = (
                set.to_expr()
                .data.drop(RESERVED_COL_KEYS, strict=False)
                .unique(maintain_order=Config.maintain_order)
            )
        elif isinstance(set, pd.Index):
            df = pl.from_pandas(pd.DataFrame(index=set).reset_index())
        elif isinstance(set, pd.DataFrame):
            df = pl.from_pandas(set)
        elif isinstance(set, pl.DataFrame):
            df = set
        elif isinstance(set, pl.Series):
            df = set.to_frame()
        elif isinstance(set, pd.Series):
            if not set.name:
                raise ValueError("Cannot convert an unnamed Pandas Series to a Set.")
            df = pl.from_pandas(set).to_frame()
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


class Expression(BaseOperableBlock):
    """Represents a linear or quadratic mathematical expression.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "item": [1, 1, 1, 2, 2],
        ...         "time": ["mon", "tue", "wed", "mon", "tue"],
        ...         "cost": [1, 2, 3, 4, 5],
        ...     }
        ... ).set_index(["item", "time"])
        >>> m = pf.Model()
        >>> m.Time = pf.Variable(df.index)
        >>> m.Size = pf.Variable(df.index)
        >>> expr = df["cost"] * m.Time + df["cost"] * m.Size
        >>> expr
        <Expression (linear) height=5 terms=10>
        ┌──────┬──────┬──────────────────────────────┐
        │ item ┆ time ┆ expression                   │
        │ (2)  ┆ (3)  ┆                              │
        ╞══════╪══════╪══════════════════════════════╡
        │ 1    ┆ mon  ┆ Time[1,mon] + Size[1,mon]    │
        │ 1    ┆ tue  ┆ 2 Time[1,tue] +2 Size[1,tue] │
        │ 1    ┆ wed  ┆ 3 Time[1,wed] +3 Size[1,wed] │
        │ 2    ┆ mon  ┆ 4 Time[2,mon] +4 Size[2,mon] │
        │ 2    ┆ tue  ┆ 5 Time[2,tue] +5 Size[2,tue] │
        └──────┴──────┴──────────────────────────────┘
    """

    def __init__(self, data: pl.DataFrame, name: str | None = None):
        # Sanity checks, VAR_KEY and COEF_KEY must be present
        assert VAR_KEY in data.columns, "Missing variable column."
        assert COEF_KEY in data.columns, "Missing coefficient column."

        # Sanity check no duplicates labels
        if Config.enable_is_duplicated_expression_safety_check:
            duplicated_mask = data.drop(COEF_KEY).is_duplicated()
            # In theory this should never happen unless there's a bug in the library
            if duplicated_mask.any():
                duplicated_data = data.filter(duplicated_mask)
                raise ValueError(
                    f"Cannot create an expression with duplicate labels:\n{duplicated_data}."
                )

        data = _simplify_expr_df(data)

        if name is None:
            warnings.warn(
                "Expression should be given a name to support troubleshooting.",
                UserWarning,
            )

            super().__init__(data)
        else:
            super().__init__(data, name=name)

    @classmethod
    def constant(cls, constant: int | float) -> Expression:
        """Creates a new expression equal to the given constant.

        Examples:
            >>> pf.Expression.constant(5)
            <Expression (parameter) terms=1>
            5
        """
        return cls(
            pl.DataFrame(
                {
                    COEF_KEY: [constant],
                    VAR_KEY: [CONST_TERM],
                },
                schema={COEF_KEY: pl.Float64, VAR_KEY: Config.id_dtype},
            ),
            name=str(constant),
        )

    @return_new
    def sum(self, *over: str):
        """Sums an expression over specified dimensions.

        If no dimensions are specified, the sum is taken over all of the expression's dimensions.

        Examples:
            >>> expr = pf.Param(
            ...     {
            ...         "time": ["mon", "tue", "wed", "mon", "tue"],
            ...         "place": [
            ...             "Toronto",
            ...             "Toronto",
            ...             "Toronto",
            ...             "Vancouver",
            ...             "Vancouver",
            ...         ],
            ...         "tiktok_posts": [1e6, 3e6, 2e6, 1e6, 2e6],
            ...     }
            ... )
            >>> expr
            <Expression (parameter) height=5 terms=5>
            ┌──────┬───────────┬────────────┐
            │ time ┆ place     ┆ expression │
            │ (3)  ┆ (2)       ┆            │
            ╞══════╪═══════════╪════════════╡
            │ mon  ┆ Toronto   ┆ 1000000    │
            │ tue  ┆ Toronto   ┆ 3000000    │
            │ wed  ┆ Toronto   ┆ 2000000    │
            │ mon  ┆ Vancouver ┆ 1000000    │
            │ tue  ┆ Vancouver ┆ 2000000    │
            └──────┴───────────┴────────────┘
            >>> expr.sum("time")
            <Expression (parameter) height=2 terms=2>
            ┌───────────┬────────────┐
            │ place     ┆ expression │
            │ (2)       ┆            │
            ╞═══════════╪════════════╡
            │ Toronto   ┆ 6000000    │
            │ Vancouver ┆ 3000000    │
            └───────────┴────────────┘
            >>> expr.sum()
            <Expression (parameter) terms=1>
            9000000

            If the given dimensions don't exist, an error will be raised:

            >>> expr.sum("city")
            Traceback (most recent call last):
            ...
            AssertionError: Cannot sum over ['city'] as it is not in ['time', 'place']

        See Also:
            [pyoframe.Expression.sum_by][] for summing over all dimensions _except_ those that are specified.
        """
        dims = self.dimensions
        if dims is None:
            raise ValueError("Cannot sum a dimensionless expression.")
        if not over:
            over = tuple(dims)
        assert set(over) <= set(dims), (
            f"Cannot sum over {list(over)} as it is not in {dims}"
        )
        remaining_dims = [dim for dim in dims if dim not in over]

        return (
            self.data.drop(over)
            .group_by(
                remaining_dims + self._variable_columns,
                maintain_order=Config.maintain_order,
            )
            .sum()
        )

    def sum_by(self, *by: str):
        """Like [`Expression.sum`][pyoframe.Expression.sum], but the sum is taken over all dimensions *except* those specified in `by` (just like a `group_by().sum()` operation).

        Examples:
            >>> expr = pf.Param(
            ...     {
            ...         "time": ["mon", "tue", "wed", "mon", "tue"],
            ...         "place": [
            ...             "Toronto",
            ...             "Toronto",
            ...             "Toronto",
            ...             "Vancouver",
            ...             "Vancouver",
            ...         ],
            ...         "tiktok_posts": [1e6, 3e6, 2e6, 1e6, 2e6],
            ...     }
            ... )
            >>> expr
            <Expression (parameter) height=5 terms=5>
            ┌──────┬───────────┬────────────┐
            │ time ┆ place     ┆ expression │
            │ (3)  ┆ (2)       ┆            │
            ╞══════╪═══════════╪════════════╡
            │ mon  ┆ Toronto   ┆ 1000000    │
            │ tue  ┆ Toronto   ┆ 3000000    │
            │ wed  ┆ Toronto   ┆ 2000000    │
            │ mon  ┆ Vancouver ┆ 1000000    │
            │ tue  ┆ Vancouver ┆ 2000000    │
            └──────┴───────────┴────────────┘

            >>> expr.sum_by("place")
            <Expression (parameter) height=2 terms=2>
            ┌───────────┬────────────┐
            │ place     ┆ expression │
            │ (2)       ┆            │
            ╞═══════════╪════════════╡
            │ Toronto   ┆ 6000000    │
            │ Vancouver ┆ 3000000    │
            └───────────┴────────────┘

            If the specified dimensions don't exist, an error will be raised:

            >>> expr.sum_by("city")
            Traceback (most recent call last):
            ...
            ValueError: Cannot sum by ['city'] because it is not a valid dimension. The expression's dimensions are: ['time', 'place'].

            >>> total_sum = expr.sum()
            >>> total_sum.sum_by("time")
            Traceback (most recent call last):
            ...
            ValueError: Cannot sum a dimensionless expression.

        See Also:
            [pyoframe.Expression.sum][] for summing over specified dimensions.
        """
        if not by:
            raise ValueError("sum_by requires at least 1 argument.")
        dims = self.dimensions
        if dims is None:
            raise ValueError("Cannot sum a dimensionless expression.")
        if not set(by) <= set(dims):
            raise ValueError(
                f"Cannot sum by {list(set(by) - set(dims))} because it is not a valid dimension. The expression's dimensions are: {list(dims)}."
            )
        remaining_dims = [dim for dim in dims if dim not in by]
        return self.sum(*remaining_dims)

    @property
    def _variable_columns(self) -> list[str]:
        if self.is_quadratic:
            return [VAR_KEY, QUAD_VAR_KEY]
        else:
            return [VAR_KEY]

    def map(self, mapping_set: SetTypes, drop_shared_dims: bool = True) -> Expression:
        """Replaces the dimensions that are shared with mapping_set with the other dimensions found in mapping_set.

        This is particularly useful to go from one type of dimensions to another. For example, to convert data that
        is indexed by city to data indexed by country (see example).

        Parameters:
            mapping_set:
                The set to map the expression to. This can be a DataFrame, Index, or another Set.
            drop_shared_dims:
                If `True`, the dimensions shared between the expression and the mapping set are dropped from the resulting expression and
                    repeated rows are summed.
                If `False`, the shared dimensions are kept in the resulting expression.

        Returns:
            A new Expression containing the result of the mapping operation.

        Examples:
            >>> import polars as pl
            >>> pop_data = pf.Param(
            ...     {
            ...         "city": ["Toronto", "Vancouver", "Boston"],
            ...         "year": [2024, 2024, 2024],
            ...         "population": [10, 2, 8],
            ...     }
            ... )
            >>> cities_and_countries = pl.DataFrame(
            ...     {
            ...         "city": ["Toronto", "Vancouver", "Boston"],
            ...         "country": ["Canada", "Canada", "USA"],
            ...     }
            ... )
            >>> pop_data.map(cities_and_countries)
            <Expression (parameter) height=2 terms=2>
            ┌──────┬─────────┬────────────┐
            │ year ┆ country ┆ expression │
            │ (1)  ┆ (2)     ┆            │
            ╞══════╪═════════╪════════════╡
            │ 2024 ┆ Canada  ┆ 12         │
            │ 2024 ┆ USA     ┆ 8          │
            └──────┴─────────┴────────────┘

            >>> pop_data.map(cities_and_countries, drop_shared_dims=False)
            <Expression (parameter) height=3 terms=3>
            ┌───────────┬──────┬─────────┬────────────┐
            │ city      ┆ year ┆ country ┆ expression │
            │ (3)       ┆ (1)  ┆ (2)     ┆            │
            ╞═══════════╪══════╪═════════╪════════════╡
            │ Toronto   ┆ 2024 ┆ Canada  ┆ 10         │
            │ Vancouver ┆ 2024 ┆ Canada  ┆ 2          │
            │ Boston    ┆ 2024 ┆ USA     ┆ 8          │
            └───────────┴──────┴─────────┴────────────┘
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
            mapped_expression = mapped_expression.sum(*shared_dims)

        mapped_expression.name = f"{self.name}.map(…)"

        return mapped_expression

    @return_new
    def rolling_sum(self, over: str, window_size: int):
        """Calculates the rolling sum of the Expression over a specified window size for a given dimension.

        This method applies a rolling sum operation over the dimension specified by `over`,
        using a window defined by `window_size`.


        Parameters:
            over:
                The name of the dimension (column) over which the rolling sum is calculated.
                This dimension must exist within the Expression's dimensions.
            window_size:
                The size of the moving window in terms of number of records.
                The rolling sum is calculated over this many consecutive elements.

        Returns:
            A new Expression instance containing the result of the rolling sum operation.
                This new Expression retains all dimensions (columns) of the original data,
                with the rolling sum applied over the specified dimension.

        Examples:
            >>> import polars as pl
            >>> cost = pl.DataFrame(
            ...     {
            ...         "item": [1, 1, 1, 2, 2],
            ...         "time": [1, 2, 3, 1, 2],
            ...         "cost": [1, 2, 3, 4, 5],
            ...     }
            ... )
            >>> m = pf.Model()
            >>> m.quantity = pf.Variable(cost[["item", "time"]])
            >>> (m.quantity * cost).rolling_sum(over="time", window_size=2)
            <Expression (linear) height=5 terms=8>
            ┌──────┬──────┬──────────────────────────────────┐
            │ item ┆ time ┆ expression                       │
            │ (2)  ┆ (3)  ┆                                  │
            ╞══════╪══════╪══════════════════════════════════╡
            │ 1    ┆ 1    ┆ quantity[1,1]                    │
            │ 1    ┆ 2    ┆ quantity[1,1] +2 quantity[1,2]   │
            │ 1    ┆ 3    ┆ 2 quantity[1,2] +3 quantity[1,3] │
            │ 2    ┆ 1    ┆ 4 quantity[2,1]                  │
            │ 2    ┆ 2    ┆ 4 quantity[2,1] +5 quantity[2,2] │
            └──────┴──────┴──────────────────────────────────┘
        """
        dims = self.dimensions
        if dims is None:
            raise ValueError(
                "Cannot use rolling_sum() with an expression with no dimensions."
            )
        assert over in dims, f"Cannot sum over {over} as it is not in {dims}"
        remaining_dims = [dim for dim in dims if dim not in over]

        return pl.concat(
            [
                df.with_columns(pl.col(over).max())
                for _, df in self.data.rolling(
                    index_column=over,
                    period=f"{window_size}i",
                    group_by=remaining_dims,
                )
            ]
        )

    @return_new
    def within(self, set: SetTypes):
        """Filters this expression to only include the dimensions within the provided set.

        Examples:
            >>> general_expr = pf.Param({"dim1": [1, 2, 3], "value": [1, 2, 3]})
            >>> filter_expr = pf.Param({"dim1": [1, 3], "value": [5, 6]})
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
        assert set_dims is not None, (
            "Cannot use .within() with a set with no dimensions."
        )
        dims = self.dimensions
        assert dims is not None, (
            "Cannot use .within() with an expression with no dimensions."
        )
        dims_in_common = [dim for dim in dims if dim in set_dims]
        by_dims = df.select(dims_in_common).unique(maintain_order=Config.maintain_order)
        return self.data.join(
            by_dims,
            on=dims_in_common,
            maintain_order="left" if Config.maintain_order else None,
        )

    @property
    def is_quadratic(self) -> bool:
        """Returns `True` if the expression is quadratic, False otherwise.

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

    @overload
    def degree(self, return_str: Literal[False] = False) -> int: ...

    @overload
    def degree(self, return_str: Literal[True] = True) -> str: ...

    def degree(self, return_str: bool = False) -> int | str:
        """Returns the degree of the expression (0=constant, 1=linear, 2=quadratic).

        Parameters:
            return_str: If `True`, returns the degree as a string (`"constant"`, `"linear"`, or `"quadratic"`).
                If `False`, returns the degree as an integer (0, 1, or 2).

        Examples:
            >>> m = pf.Model()
            >>> m.v1 = pf.Variable()
            >>> m.v2 = pf.Variable()
            >>> expr = pf.Param({"dim1": [1, 2, 3], "value": [1, 2, 3]})
            >>> expr.degree()
            0
            >>> expr *= m.v1
            >>> expr.degree()
            1
            >>> expr += (m.v2**2).over("dim1")
            >>> expr.degree()
            2
            >>> expr.degree(return_str=True)
            'quadratic'
        """
        if self.is_quadratic:
            return "quadratic" if return_str else 2
        # TODO improve performance of .evaluate() by ensuring early exit if linear
        elif (self.data.get_column(VAR_KEY) != CONST_TERM).any():
            return "linear" if return_str else 1
        else:
            return "parameter" if return_str else 0

    def __add__(self, other):
        """Adds another expression or a constant to this expression.

        Examples:
            >>> m = pf.Model()
            >>> add = pf.Param({"dim1": [1, 2, 3], "add": [10, 20, 30]})
            >>> m.v = Variable(add)
            >>> m.v + add
            <Expression (linear) height=3 terms=6>
            ┌──────┬────────────┐
            │ dim1 ┆ expression │
            │ (3)  ┆            │
            ╞══════╪════════════╡
            │ 1    ┆ v[1] +10   │
            │ 2    ┆ v[2] +20   │
            │ 3    ┆ v[3] +30   │
            └──────┴────────────┘

            >>> m.v + add + 2
            <Expression (linear) height=3 terms=6>
            ┌──────┬────────────┐
            │ dim1 ┆ expression │
            │ (3)  ┆            │
            ╞══════╪════════════╡
            │ 1    ┆ 12 + v[1]  │
            │ 2    ┆ 22 + v[2]  │
            │ 3    ┆ 32 + v[3]  │
            └──────┴────────────┘

            >>> m.v + pd.DataFrame({"dim1": [1, 2], "add": [10, 20]})
            Traceback (most recent call last):
            ...
            pyoframe._constants.PyoframeError: Cannot add the two expressions below because expression 1 has extra labels.
            Expression 1:	v
            Expression 2:	add
            Extra labels in expression 1:
            ┌──────┐
            │ dim1 │
            ╞══════╡
            │ 3    │
            └──────┘
            Use .drop_extras() or .keep_extras() to indicate how the extra labels should be handled. Learn more at
                https://bravos-power.github.io/pyoframe/latest/learn/concepts/addition
            >>> m.v2 = Variable()
            >>> 5 + 2 * m.v2
            <Expression (linear) terms=2>
            2 v2 +5
        """
        if isinstance(other, (int, float)):
            return self._add_const(other)
        other = other.to_expr()  # TODO don't rely on monkey patch
        self._learn_from_other(other)
        return add(self, other)

    def __mul__(self: Expression, other: Operable) -> Expression:
        if isinstance(other, (int, float)):
            if other == 1:
                return self
            return self._new(
                self.data.with_columns(pl.col(COEF_KEY) * other),
                name=f"({other} * {self.name})",
            )

        other: Expression = other.to_expr()  # TODO don't rely on monkey patch
        self._learn_from_other(other)
        return multiply(self, other)

    def __rtruediv__(self, other):
        """Support dividing by an expression when that expression is a constant."""
        assert isinstance(other, (int, float)), (
            f"Expected a number not a {type(other)} when dividing by an expression."
        )
        if self.degree() != 0:
            raise PyoframeError(
                f"Cannot divide by '{self.name}' because denominators cannot contain variables."
            )

        return self._new(
            self.data.with_columns((pl.lit(other) / pl.col(COEF_KEY)).alias(COEF_KEY)),
            name=f"({other} / {self.name})",
        )

    def to_expr(self) -> Expression:
        """Returns the expression itself."""
        return self

    def _learn_from_other(self, other: Expression):
        if self._model is None and other._model is not None:
            self._model = other._model

    def _new(self, data: pl.DataFrame, name: str) -> Expression:
        e = Expression(data, name)
        e._model = self._model
        return e

    def _add_const(self, const: int | float) -> Expression:
        """Adds a constant to the expression.

        Examples:
            >>> m = pf.Model()
            >>> m.x1 = Variable()
            >>> m.x2 = Variable()
            >>> m.x1 + 5
            <Expression (linear) terms=2>
            x1 +5
            >>> m.x1**2 + 5
            <Expression (quadratic) terms=2>
            x1 * x1 +5
            >>> m.x1**2 + m.x2 + 5
            <Expression (quadratic) terms=3>
            x1 * x1 + x2 +5

            It also works with dimensions

            >>> m = pf.Model()
            >>> m.v = Variable({"dim1": [1, 2, 3]})
            >>> m.v * m.v + 5
            <Expression (quadratic) height=3 terms=6>
            ┌──────┬─────────────────┐
            │ dim1 ┆ expression      │
            │ (3)  ┆                 │
            ╞══════╪═════════════════╡
            │ 1    ┆ 5 + v[1] * v[1] │
            │ 2    ┆ 5 + v[2] * v[2] │
            │ 3    ┆ 5 + v[3] * v[3] │
            └──────┴─────────────────┘
        """
        if const == 0:
            return self
        dim = self.dimensions
        data = self.data
        # Fill in missing constant terms
        if not dim:
            if CONST_TERM not in data[VAR_KEY]:
                const_df = pl.DataFrame(
                    {COEF_KEY: [0.0], VAR_KEY: [CONST_TERM]},
                    schema={COEF_KEY: pl.Float64, VAR_KEY: Config.id_dtype},
                )
                if self.is_quadratic:
                    const_df = const_df.with_columns(
                        pl.lit(CONST_TERM).alias(QUAD_VAR_KEY).cast(Config.id_dtype)
                    )
                data = pl.concat(
                    [data, const_df],
                    how="vertical_relaxed",
                )
        else:
            keys = (
                data.select(dim)
                .unique(maintain_order=Config.maintain_order)
                .with_columns(pl.lit(CONST_TERM).alias(VAR_KEY).cast(Config.id_dtype))
            )
            if self.is_quadratic:
                keys = keys.with_columns(
                    pl.lit(CONST_TERM).alias(QUAD_VAR_KEY).cast(Config.id_dtype)
                )
            data = data.join(
                keys,
                on=dim + self._variable_columns,
                how="full",
                coalesce=True,
                # We use right_left not left_right to bring the constants near the front for better readability
                maintain_order="right_left" if Config.maintain_order else None,
            ).with_columns(pl.col(COEF_KEY).fill_null(0.0))

        data = data.with_columns(
            pl.when(pl.col(VAR_KEY) == CONST_TERM)
            .then(pl.col(COEF_KEY) + const)
            .otherwise(pl.col(COEF_KEY))
        )

        name = f"({self.name} + {const})" if const >= 0 else f"({self.name} - {-const})"
        return self._new(data, name=name)

    @property
    def constant_terms(self) -> pl.DataFrame:
        """Returns all the constant terms in the expression."""
        dims = self.dimensions
        constant_terms = self.data.filter(pl.col(VAR_KEY) == CONST_TERM).drop(VAR_KEY)
        if self.is_quadratic:
            constant_terms = constant_terms.drop(QUAD_VAR_KEY)
        if dims is not None:
            dims_df = self.data.select(dims).unique(
                maintain_order=Config.maintain_order
            )
            df = constant_terms.join(
                dims_df,
                on=dims,
                how="full",
                coalesce=True,
                maintain_order="left_right" if Config.maintain_order else None,
            )
            return df.with_columns(pl.col(COEF_KEY).fill_null(0.0))
        else:
            if len(constant_terms) == 0:
                return pl.DataFrame(
                    {COEF_KEY: [0.0], VAR_KEY: [CONST_TERM]},
                    schema={COEF_KEY: pl.Float64, VAR_KEY: Config.id_dtype},
                )
            return constant_terms

    @property
    def variable_terms(self) -> pl.DataFrame:
        """Returns all the non-constant terms in the expression."""
        return self.data.filter(pl.col(VAR_KEY) != CONST_TERM)

    @unwrap_single_values
    def evaluate(self) -> pl.DataFrame:
        """Computes the value of the expression using the variables' solutions.

        Returns:
            A Polars `DataFrame` for dimensioned expressions a `float` for dimensionless expressions.

        Examples:
            >>> m = pf.Model()
            >>> m.X = pf.Variable({"dim1": [1, 2, 3]}, lb=10, ub=10)
            >>> m.expr = 2 * m.X * m.X + 1

            >>> m.expr.evaluate()
            Traceback (most recent call last):
            ...
            ValueError: Cannot evaluate the expression 'expr' before calling model.optimize().

            >>> m.constant_expression = m.expr - 2 * m.X * m.X
            >>> m.constant_expression.evaluate()
            shape: (3, 2)
            ┌──────┬──────────┐
            │ dim1 ┆ solution │
            │ ---  ┆ ---      │
            │ i64  ┆ f64      │
            ╞══════╪══════════╡
            │ 1    ┆ 1.0      │
            │ 2    ┆ 1.0      │
            │ 3    ┆ 1.0      │
            └──────┴──────────┘


            >>> m.optimize()
            >>> m.expr.evaluate()
            shape: (3, 2)
            ┌──────┬──────────┐
            │ dim1 ┆ solution │
            │ ---  ┆ ---      │
            │ i64  ┆ f64      │
            ╞══════╪══════════╡
            │ 1    ┆ 201.0    │
            │ 2    ┆ 201.0    │
            │ 3    ┆ 201.0    │
            └──────┴──────────┘

            >>> m.expr.sum().evaluate()
            603.0

        """
        assert self._model is not None, (
            "Expression must be added to the model to use .value"
        )

        df = self.data.rename({COEF_KEY: SOLUTION_KEY})
        sm = self._model.poi
        attr = poi.VariableAttribute.Value

        if self.degree() == 0:
            df = df.drop(self._variable_columns)
        elif (
            self._model.attr.TerminationStatus
            == poi.TerminationStatusCode.OPTIMIZE_NOT_CALLED
        ):
            raise ValueError(
                f"Cannot evaluate the expression '{self.name}' before calling model.optimize()."
            )
        else:
            for var_col in self._variable_columns:
                values = [
                    sm.get_variable_attribute(poi.VariableIndex(v_id), attr)
                    for v_id in df.get_column(var_col).to_list()
                ]

                df = df.drop(var_col).with_columns(
                    pl.col(SOLUTION_KEY) * pl.Series(values, dtype=pl.Float64)
                )

        dims = self.dimensions
        if dims is not None:
            df = df.group_by(dims, maintain_order=Config.maintain_order)
        return df.sum()

    def _to_poi(self) -> poi.ScalarAffineFunction | poi.ScalarQuadraticFunction:
        assert self.dimensions is None, (
            "._to_poi() only works for non-dimensioned expressions."
        )

        data = self.data

        if self.is_quadratic:
            # Workaround for bug https://github.com/metab0t/PyOptInterface/issues/59
            if self._model is None or self._model.solver.name == "highs":
                data = data.sort(VAR_KEY, QUAD_VAR_KEY, descending=False)

            return poi.ScalarQuadraticFunction(
                coefficients=data.get_column(COEF_KEY).to_numpy(),
                var1s=data.get_column(VAR_KEY).to_numpy(),
                var2s=data.get_column(QUAD_VAR_KEY).to_numpy(),
            )
        else:
            return poi.ScalarAffineFunction(
                coefficients=data.get_column(COEF_KEY).to_numpy(),
                variables=data.get_column(VAR_KEY).to_numpy(),
            )

    @overload
    def to_str(
        self,
        str_col_name: str = "expression",
        include_const_term: bool = True,
        return_df: Literal[False] = False,
    ) -> str: ...

    @overload
    def to_str(
        self,
        str_col_name: str = "expression",
        include_const_term: bool = True,
        return_df: Literal[True] = True,
    ) -> pl.DataFrame: ...

    def to_str(
        self,
        str_col_name: str = "expression",
        include_const_term: bool = True,
        return_df: bool = False,
    ) -> str | pl.DataFrame:
        """Converts the expression to a human-readable string, or several arranged in a table.

        Long expressions are truncated according to [`Config.print_max_terms`][pyoframe._Config.print_max_terms] and [`Config.print_polars_config`][pyoframe._Config.print_polars_config].

        `str(pyoframe.Expression)` is equivalent to `pyoframe.Expression.to_str()`.

        Parameters:
            str_col_name:
                The name of the column containing the string representation of the expression (dimensioned expressions only).
            include_const_term:
                If `False`, constant terms are omitted from the string representation.
            return_df:
                If `True`, returns a DataFrame containing the human-readable strings instead of the DataFrame's string representation.

        Examples:
            >>> import polars as pl
            >>> m = pf.Model()
            >>> x = pf.Set(x=range(1000))
            >>> y = pf.Set(y=range(1000))
            >>> m.V = pf.Variable(x, y)
            >>> expr = 2 * m.V * m.V + 3
            >>> print(expr.to_str())
            ┌────────┬────────┬──────────────────────────────┐
            │ x      ┆ y      ┆ expression                   │
            │ (1000) ┆ (1000) ┆                              │
            ╞════════╪════════╪══════════════════════════════╡
            │ 0      ┆ 0      ┆ 3 +2 V[0,0] * V[0,0]         │
            │ 0      ┆ 1      ┆ 3 +2 V[0,1] * V[0,1]         │
            │ 0      ┆ 2      ┆ 3 +2 V[0,2] * V[0,2]         │
            │ 0      ┆ 3      ┆ 3 +2 V[0,3] * V[0,3]         │
            │ 0      ┆ 4      ┆ 3 +2 V[0,4] * V[0,4]         │
            │ …      ┆ …      ┆ …                            │
            │ 999    ┆ 995    ┆ 3 +2 V[999,995] * V[999,995] │
            │ 999    ┆ 996    ┆ 3 +2 V[999,996] * V[999,996] │
            │ 999    ┆ 997    ┆ 3 +2 V[999,997] * V[999,997] │
            │ 999    ┆ 998    ┆ 3 +2 V[999,998] * V[999,998] │
            │ 999    ┆ 999    ┆ 3 +2 V[999,999] * V[999,999] │
            └────────┴────────┴──────────────────────────────┘
            >>> expr = expr.sum("y")
            >>> print(expr.to_str())
            ┌────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
            │ x      ┆ expression                                                                              │
            │ (1000) ┆                                                                                         │
            ╞════════╪═════════════════════════════════════════════════════════════════════════════════════════╡
            │ 0      ┆ 3000 +2 V[0,0] * V[0,0] +2 V[0,1] * V[0,1] +2 V[0,2] * V[0,2] +2 V[0,3] * V[0,3] …      │
            │ 1      ┆ 3000 +2 V[1,0] * V[1,0] +2 V[1,1] * V[1,1] +2 V[1,2] * V[1,2] +2 V[1,3] * V[1,3] …      │
            │ 2      ┆ 3000 +2 V[2,0] * V[2,0] +2 V[2,1] * V[2,1] +2 V[2,2] * V[2,2] +2 V[2,3] * V[2,3] …      │
            │ 3      ┆ 3000 +2 V[3,0] * V[3,0] +2 V[3,1] * V[3,1] +2 V[3,2] * V[3,2] +2 V[3,3] * V[3,3] …      │
            │ 4      ┆ 3000 +2 V[4,0] * V[4,0] +2 V[4,1] * V[4,1] +2 V[4,2] * V[4,2] +2 V[4,3] * V[4,3] …      │
            │ …      ┆ …                                                                                       │
            │ 995    ┆ 3000 +2 V[995,0] * V[995,0] +2 V[995,1] * V[995,1] +2 V[995,2] * V[995,2] +2 V[995,3] * │
            │        ┆ V[995,3] …                                                                              │
            │ 996    ┆ 3000 +2 V[996,0] * V[996,0] +2 V[996,1] * V[996,1] +2 V[996,2] * V[996,2] +2 V[996,3] * │
            │        ┆ V[996,3] …                                                                              │
            │ 997    ┆ 3000 +2 V[997,0] * V[997,0] +2 V[997,1] * V[997,1] +2 V[997,2] * V[997,2] +2 V[997,3] * │
            │        ┆ V[997,3] …                                                                              │
            │ 998    ┆ 3000 +2 V[998,0] * V[998,0] +2 V[998,1] * V[998,1] +2 V[998,2] * V[998,2] +2 V[998,3] * │
            │        ┆ V[998,3] …                                                                              │
            │ 999    ┆ 3000 +2 V[999,0] * V[999,0] +2 V[999,1] * V[999,1] +2 V[999,2] * V[999,2] +2 V[999,3] * │
            │        ┆ V[999,3] …                                                                              │
            └────────┴─────────────────────────────────────────────────────────────────────────────────────────┘
            >>> expr = expr.sum("x")
            >>> print(expr.to_str())
            3000000 +2 V[0,0] * V[0,0] +2 V[0,1] * V[0,1] +2 V[0,2] * V[0,2] +2 V[0,3] * V[0,3] …

        """
        # TODO consider optimizing using LazyFrames since .head() could maybe be automatically pushed up the chain of operations.
        data = self.data if include_const_term else self.variable_terms
        data = cast_coef_to_string(data)

        for var_col in self._variable_columns:
            temp_var_column = f"{var_col}_temp"
            if self._model is not None and self._model._var_map is not None:
                data = self._model._var_map.apply(
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
            pl.concat_str(
                COEF_KEY,
                pl.lit(" "),
                VAR_KEY,
            )
            .str.strip_chars(characters="  ")
            .alias(str_col_name)
        ).drop(COEF_KEY, VAR_KEY)

        if dimensions is not None:
            data = data.group_by(dimensions, maintain_order=Config.maintain_order).agg(
                pl.concat_str(
                    pl.col(str_col_name)
                    .head(Config.print_max_terms)
                    .str.join(delimiter=" "),
                    pl.when(pl.len() > Config.print_max_terms)
                    .then(pl.lit(" …"))
                    .otherwise(pl.lit("")),
                )
            )
        else:
            truncate = data.height > Config.print_max_terms
            if truncate:
                data = data.head(Config.print_max_terms)

            data = data.select(pl.col(str_col_name).str.join(delimiter=" "))

            if truncate:
                data = data.with_columns(
                    pl.concat_str(pl.col(str_col_name), pl.lit(" …"))
                )

        # Remove leading +
        data = data.with_columns(pl.col(str_col_name).str.strip_chars(characters="  +"))

        if not return_df:
            if dimensions is None and not self._allowed_new_dims:
                data = data.item()
            else:
                data = self._add_shape_to_columns(data)
                data = self._add_allowed_new_dims_to_df(data)
                with Config.print_polars_config:
                    data = repr(data)

        return data

    def _str_header(self) -> str:
        """Returns a string representation of the expression's header."""
        return get_obj_repr(
            self,
            f"({self.degree(return_str=True)})",
            height=len(self) if self.dimensions else None,
            terms=self.terms,
        )

    def __repr__(self) -> str:
        return self._str_header() + "\n" + self.to_str()

    def __str__(self) -> str:
        return self.to_str()

    @property
    def terms(self) -> int:
        """The number of terms across all subexpressions.

        Expressions equal to zero count as one term.

        Examples:
            >>> import polars as pl
            >>> m = pf.Model()
            >>> m.v = pf.Variable({"t": [1, 2]})
            >>> coef = pl.DataFrame({"t": [1, 2], "coef": [0, 1]})
            >>> coef * (m.v + 4)
            <Expression (linear) height=2 terms=3>
            ┌─────┬────────────┐
            │ t   ┆ expression │
            │ (2) ┆            │
            ╞═════╪════════════╡
            │ 1   ┆ 0          │
            │ 2   ┆ 4 + v[2]   │
            └─────┴────────────┘
            >>> (coef * (m.v + 4)).terms
            3
        """
        return len(self.data)


@overload
def sum(over: str | Sequence[str], expr: Operable) -> Expression: ...


@overload
def sum(over: Operable) -> Expression: ...


def sum(
    over: str | Sequence[str] | Operable,
    expr: Operable | None = None,
) -> Expression:  # pragma: no cover
    """Deprecated: Use Expression.sum() or Variable.sum() instead.

    Examples:
        >>> x = pf.Set(x=range(100))
        >>> pf.sum(x)
        Traceback (most recent call last):
          ...
        DeprecationWarning: pf.sum() is deprecated. Use Expression.sum() or Variable.sum() instead.
    """
    warnings.warn(
        "pf.sum() is deprecated. Use Expression.sum() or Variable.sum() instead.",
        DeprecationWarning,
    )

    if expr is None:
        assert isinstance(over, BaseOperableBlock)
        return over.to_expr().sum()
    else:
        assert isinstance(over, (str, Sequence))
        if isinstance(over, str):
            over = (over,)
        return expr.to_expr().sum(*over)


def sum_by(by: str | Sequence[str], expr: Operable) -> Expression:  # pragma: no cover
    """Deprecated: Use Expression.sum() or Variable.sum() instead."""
    warnings.warn(
        "pf.sum_by() is deprecated. Use Expression.sum_by() or Variable.sum_by() instead.",
        DeprecationWarning,
    )

    if isinstance(by, str):
        by = [by]
    return expr.to_expr().sum_by(*by)


class Constraint(BaseBlock):
    """An optimization constraint that can be added to a [Model][pyoframe.Model].

    Tip: Implementation Note
        Pyoframe simplifies constraints by moving all the constraint's mathematical terms to the left-hand side.
        This way, the right-hand side is always zero, and constraints only need to manage one expression.

    Warning: Use `<=`, `>=`, or `==` operators to create constraints
        Constraints should be created using the `<=`, `>=`, or `==` operators, not by directly calling the `Constraint` constructor.

    Parameters:
        lhs:
            The constraint's left-hand side expression.
        sense:
            The sense of the constraint.
    """

    def __init__(self, lhs: Expression, sense: ConstraintSense):
        self.lhs: Expression = lhs
        self._model = lhs._model
        self.sense = sense
        self._to_relax: FuncArgs | None = None
        self._attr = Container(self._set_attribute, self._get_attribute)

        dims = self.lhs.dimensions
        data = (
            pl.DataFrame()
            if dims is None
            else self.lhs.data.select(dims).unique(maintain_order=Config.maintain_order)
        )

        super().__init__(data)

    @property
    def attr(self) -> Container:
        """Allows reading and writing constraint attributes similarly to [Model.attr][pyoframe.Model.attr]."""
        return self._attr

    def _set_attribute(self, name, value):
        self._assert_has_ids()
        col_name = name
        try:
            name = poi.ConstraintAttribute[name]
            setter = self._model.poi.set_constraint_attribute
        except KeyError:
            setter = self._model.poi.set_constraint_raw_attribute

        constr_type = (
            poi.ConstraintType.Quadratic
            if self.lhs.is_quadratic
            else poi.ConstraintType.Linear
        )

        if self.dimensions is None:
            for key in self.data.get_column(CONSTRAINT_KEY):
                setter(poi.ConstraintIndex(constr_type, key), name, value)
        else:
            for key, value in (
                self.data.join(
                    value,
                    on=self.dimensions,
                    maintain_order="left" if Config.maintain_order else None,
                )
                .select(pl.col(CONSTRAINT_KEY), pl.col(col_name))
                .iter_rows()
            ):
                setter(poi.ConstraintIndex(constr_type, key), name, value)

    @unwrap_single_values
    def _get_attribute(self, name):
        self._assert_has_ids()
        col_name = name
        try:
            name = poi.ConstraintAttribute[name]
            getter = self._model.poi.get_constraint_attribute
        except KeyError:
            getter = self._model.poi.get_constraint_raw_attribute

        constr_type = (
            poi.ConstraintType.Quadratic
            if self.lhs.is_quadratic
            else poi.ConstraintType.Linear
        )

        ids = self.data.get_column(CONSTRAINT_KEY).to_list()
        attr = [getter(poi.ConstraintIndex(constr_type, v_id), name) for v_id in ids]
        data = self.data.with_columns(pl.Series(attr).alias(col_name))
        return data.select(self._dimensions_unsafe + [col_name])

    def _on_add_to_model(self, model: Model, name: str):
        super()._on_add_to_model(model, name)
        if self._to_relax is not None:
            self.relax(*self._to_relax.args, **self._to_relax.kwargs)
        self._assign_ids()

    def _assign_ids(self):
        """This function is the main bottleneck for pyoframe.

        I've spent a lot of time optimizing it.
        """
        assert self._model is not None

        is_quadratic = self.lhs.is_quadratic
        use_var_names = self._model.solver_uses_variable_names
        sense = self.sense._to_poi()
        dims = self.dimensions
        df = self.lhs.data
        add_constraint = (
            self._model.poi._add_quadratic_constraint
            if is_quadratic
            else self._model.poi._add_linear_constraint
        )

        # GRBaddconstr uses sprintf when no name or "" is given. sprintf is slow. As such, we specify "C" as the name.
        # Specifying "" is the same as not specifying anything, see pyoptinterface:
        # https://github.com/metab0t/PyOptInterface/blob/6d61f3738ad86379cff71fee77077d4ea919f2d5/lib/gurobi_model.cpp#L338
        name = "C" if self._model.solver.accelerate_with_repeat_names else ""

        if dims is None:
            if self._model.solver_uses_variable_names:
                name = self.name
            create_expression = (
                poi.ScalarQuadraticFunction
                if is_quadratic
                else poi.ScalarAffineFunction.from_numpy  # when called only once from_numpy is faster
            )
            constr_id = add_constraint(
                create_expression(
                    *(
                        df.get_column(c).to_numpy()
                        for c in ([COEF_KEY] + self.lhs._variable_columns)
                    )
                ),
                sense,
                0,
                name,
            ).index
            try:
                df = self.data.with_columns(
                    pl.lit(constr_id).alias(CONSTRAINT_KEY).cast(Config.id_dtype)
                )
            except TypeError as e:
                raise TypeError(
                    f"Number of constraints exceeds the current data type ({Config.id_dtype}). Consider increasing the data type by changing Config.id_dtype."
                ) from e
        else:
            create_expression = (
                poi.ScalarQuadraticFunction
                if is_quadratic
                else poi.ScalarAffineFunction  # when called multiple times the default constructor is fastest
            )
            if Config.maintain_order:
                # This adds a 5-10% overhead on _assign_ids but ensures the order
                # is the same as the input data
                df_unique = df.select(dims).unique(maintain_order=True)
                df = (
                    df.join(
                        df_unique.with_row_index(),
                        on=dims,
                        maintain_order="left",
                    )
                    .sort("index", maintain_order=True)
                    .drop("index")
                )
            else:
                df = df.sort(dims, maintain_order=False)
                # must maintain order otherwise results are wrong!
                df_unique = df.select(dims).unique(maintain_order=True)
            coefs = df.get_column(COEF_KEY).to_list()
            vars = df.get_column(VAR_KEY).to_list()
            if is_quadratic:
                vars2 = df.get_column(QUAD_VAR_KEY).to_list()

            split = (
                df.lazy()
                .with_row_index()
                .filter(pl.struct(dims).is_first_distinct())
                .select("index")
                .collect()
                .to_series()
                .to_list()
            ) + [df.height]
            del df

            # Note: list comprehension was slightly faster than using polars map_elements
            # Note 2: not specifying the argument name (`expr=`) was also slightly faster.
            # Note 3: we could have merged the if-else using an expansion operator (*) but that is slow.
            # Note 4: using kwargs is slow and including the constant term for linear expressions is faster.
            if use_var_names:
                names = concat_dimensions(df_unique, prefix=self.name)[
                    "concated_dim"
                ].to_list()
                if is_quadratic:
                    ids = [
                        add_constraint(
                            create_expression(coefs[s0:s1], vars[s0:s1], vars2[s0:s1]),
                            sense,
                            0,
                            names[i],
                        ).index
                        for i, (s0, s1) in enumerate(pairwise(split))
                    ]
                else:
                    ids = [
                        add_constraint(
                            create_expression(coefs[s0:s1], vars[s0:s1], 0),
                            sense,
                            0,
                            names[i],
                        ).index
                        for i, (s0, s1) in enumerate(pairwise(split))
                    ]
            else:
                if is_quadratic:
                    ids = [
                        add_constraint(
                            create_expression(coefs[s0:s1], vars[s0:s1], vars2[s0:s1]),
                            sense,
                            0,
                            name,
                        ).index
                        for s0, s1 in pairwise(split)
                    ]
                else:
                    ids = [
                        add_constraint(
                            create_expression(coefs[s0:s1], vars[s0:s1], 0),
                            sense,
                            0,
                            name,
                        ).index
                        for s0, s1 in pairwise(split)
                    ]
            try:
                df = df_unique.with_columns(
                    pl.Series(ids, dtype=Config.id_dtype).alias(CONSTRAINT_KEY)
                )
            except TypeError as e:
                raise TypeError(
                    f"Number of constraints exceeds the current data type ({Config.id_dtype}). Consider increasing the data type by changing Config.id_dtype."
                ) from e

        self._data = df

    @property
    def dual(self) -> pl.DataFrame | float:
        """Returns the constraint's dual values.

        Examples:
            >>> m = pf.Model()
            >>> m.x = pf.Variable()
            >>> m.y = pf.Variable()
            >>> m.maximize = m.x - m.y

            Notice that for every unit increase in the right-hand side, the objective only improves by 0.5.
            >>> m.constraint_x = 2 * m.x <= 10
            >>> m.constraint_y = 2 * m.y >= 5
            >>> m.optimize()

            For every unit increase in the right-hand side of `constraint_x`, the objective improves by 0.5.
            >>> m.constraint_x.dual
            0.5

            For every unit increase in the right-hand side of `constraint_y`, the objective worsens by 0.5.
            >>> m.constraint_y.dual
            -0.5
        """
        dual = self.attr.Dual
        if isinstance(dual, pl.DataFrame):
            dual = dual.rename({"Dual": DUAL_KEY})

        # Weirdly, IPOPT returns dual values with the opposite sign, so we correct this bug.
        # It also does this for maximization problems
        # but since we flip the objective (because Ipopt doesn't support maximization), the double negatives cancel out.
        assert self._model is not None
        if self._model.solver.name == "ipopt" and self._model.sense == ObjSense.MIN:
            if isinstance(dual, pl.DataFrame):
                dual = dual.with_columns(-pl.col(DUAL_KEY))
            else:
                dual = -dual
        return dual

    @classmethod
    def _get_id_column_name(cls):
        return CONSTRAINT_KEY

    def filter(self, *args, **kwargs) -> pl.DataFrame:
        """Syntactic sugar on `Constraint.lhs.data.filter()`, to help debugging."""
        return self.lhs.data.filter(*args, **kwargs)

    def relax(self, cost: Operable, max: Operable | None = None) -> Constraint:
        """Allows the constraint to be violated at a `cost` and, optionally, up to a maximum.

        Warning:
            `.relax()` must be called before the constraint is assigned to the [Model][pyoframe.Model] (see examples below).

        Parameters:
            cost:
                The cost of violating the constraint. Costs should be positive because Pyoframe will automatically
                make them negative for maximization problems.
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

            `relax` can only be called after the sense of the model has been defined.

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

            >>> homework_due_tomorrow = pl.DataFrame(
            ...     {
            ...         "project": ["A", "B", "C"],
            ...         "cost_per_hour_underdelivered": [10, 20, 30],
            ...         "hours_to_finish": [9, 9, 9],
            ...         "max_underdelivered": [1, 9, 9],
            ...     }
            ... )
            >>> m.hours_spent = pf.Variable(homework_due_tomorrow["project"], lb=0)
            >>> m.must_finish_project = (
            ...     m.hours_spent
            ...     >= homework_due_tomorrow[["project", "hours_to_finish"]]
            ... ).relax(
            ...     homework_due_tomorrow[["project", "cost_per_hour_underdelivered"]],
            ...     max=homework_due_tomorrow[["project", "max_underdelivered"]],
            ... )
            >>> m.only_one_day = m.hours_spent.sum("project") <= 24
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
        if m is None:
            self._to_relax = FuncArgs(args=[cost, max])
            return self

        var_name = f"{self.name}_relaxation"
        assert not hasattr(m, var_name), (
            "Conflicting names, relaxation variable already exists on the model."
        )
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
            penalty = penalty.sum()
        if m.sense is None:
            raise ValueError(
                "Cannot relax a constraint before the objective sense has been set. Try setting the objective first or using Model(sense=...)."
            )
        elif m.sense == ObjSense.MAX:
            penalty *= -1
        if m.has_objective:
            m.objective += penalty
        else:
            m.objective = penalty

        return self

    def estimated_size(self, *args, **kwargs):
        """Returns the estimated size of the constraint.

        Includes the size of the underlying expression (`Constraint.lhs`).

        See [`Expression.estimated_size`][pyoframe.Expression.estimated_size] for details on signature and behavior.

        Examples:
            An dimensionless constraint has contains a 32 bit constraint id and, for each term, a 64 bit coefficient with a 32 bit variable id.
            For a two-term expression that is: (32 + 2 * (64 + 32)) = 224 bits = 28 bytes.

            >>> m = pf.Model()
            >>> m.x = pf.Variable()
            >>> m.con = m.x <= 4
            >>> m.con.estimated_size()
            28
        """
        return super().estimated_size(*args, **kwargs) + self.lhs.estimated_size(
            *args, **kwargs
        )

    @overload
    def to_str(self, return_df: Literal[False] = False) -> str: ...

    @overload
    def to_str(self, return_df: Literal[True] = True) -> pl.DataFrame: ...

    def to_str(self, return_df: bool = False) -> str | pl.DataFrame:
        """Converts the constraint to a human-readable string, or several arranged in a table.

        Long expressions are truncated according to [`Config.print_max_terms`][pyoframe._Config.print_max_terms] and [`Config.print_polars_config`][pyoframe._Config.print_polars_config].

        Parameters:
            return_df:
                If `True`, returns a DataFrame containing strings instead of the string representation of the DataFrame.

        Examples:
            >>> import polars as pl
            >>> m = pf.Model()
            >>> x = pf.Set(x=range(1000))
            >>> y = pf.Set(y=range(1000))
            >>> m.V = pf.Variable(x, y)
            >>> expr = 2 * m.V * m.V
            >>> print((expr <= 3).to_str())
            ┌────────┬────────┬────────────────────────────────┐
            │ x      ┆ y      ┆ constraint                     │
            │ (1000) ┆ (1000) ┆                                │
            ╞════════╪════════╪════════════════════════════════╡
            │ 0      ┆ 0      ┆ 2 V[0,0] * V[0,0] <= 3         │
            │ 0      ┆ 1      ┆ 2 V[0,1] * V[0,1] <= 3         │
            │ 0      ┆ 2      ┆ 2 V[0,2] * V[0,2] <= 3         │
            │ 0      ┆ 3      ┆ 2 V[0,3] * V[0,3] <= 3         │
            │ 0      ┆ 4      ┆ 2 V[0,4] * V[0,4] <= 3         │
            │ …      ┆ …      ┆ …                              │
            │ 999    ┆ 995    ┆ 2 V[999,995] * V[999,995] <= 3 │
            │ 999    ┆ 996    ┆ 2 V[999,996] * V[999,996] <= 3 │
            │ 999    ┆ 997    ┆ 2 V[999,997] * V[999,997] <= 3 │
            │ 999    ┆ 998    ┆ 2 V[999,998] * V[999,998] <= 3 │
            │ 999    ┆ 999    ┆ 2 V[999,999] * V[999,999] <= 3 │
            └────────┴────────┴────────────────────────────────┘
            >>> expr = expr.sum("x")
            >>> print((expr >= 3).to_str())
            ┌────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
            │ y      ┆ constraint                                                                              │
            │ (1000) ┆                                                                                         │
            ╞════════╪═════════════════════════════════════════════════════════════════════════════════════════╡
            │ 0      ┆ 2 V[0,0] * V[0,0] +2 V[1,0] * V[1,0] +2 V[2,0] * V[2,0] +2 V[3,0] * V[3,0] +2 V[4,0] *  │
            │        ┆ V[4,0] … >= 3                                                                           │
            │ 1      ┆ 2 V[0,1] * V[0,1] +2 V[1,1] * V[1,1] +2 V[2,1] * V[2,1] +2 V[3,1] * V[3,1] +2 V[4,1] *  │
            │        ┆ V[4,1] … >= 3                                                                           │
            │ 2      ┆ 2 V[0,2] * V[0,2] +2 V[1,2] * V[1,2] +2 V[2,2] * V[2,2] +2 V[3,2] * V[3,2] +2 V[4,2] *  │
            │        ┆ V[4,2] … >= 3                                                                           │
            │ 3      ┆ 2 V[0,3] * V[0,3] +2 V[1,3] * V[1,3] +2 V[2,3] * V[2,3] +2 V[3,3] * V[3,3] +2 V[4,3] *  │
            │        ┆ V[4,3] … >= 3                                                                           │
            │ 4      ┆ 2 V[0,4] * V[0,4] +2 V[1,4] * V[1,4] +2 V[2,4] * V[2,4] +2 V[3,4] * V[3,4] +2 V[4,4] *  │
            │        ┆ V[4,4] … >= 3                                                                           │
            │ …      ┆ …                                                                                       │
            │ 995    ┆ 2 V[0,995] * V[0,995] +2 V[1,995] * V[1,995] +2 V[2,995] * V[2,995] +2 V[3,995] *       │
            │        ┆ V[3,995] +2 V[4,99…                                                                     │
            │ 996    ┆ 2 V[0,996] * V[0,996] +2 V[1,996] * V[1,996] +2 V[2,996] * V[2,996] +2 V[3,996] *       │
            │        ┆ V[3,996] +2 V[4,99…                                                                     │
            │ 997    ┆ 2 V[0,997] * V[0,997] +2 V[1,997] * V[1,997] +2 V[2,997] * V[2,997] +2 V[3,997] *       │
            │        ┆ V[3,997] +2 V[4,99…                                                                     │
            │ 998    ┆ 2 V[0,998] * V[0,998] +2 V[1,998] * V[1,998] +2 V[2,998] * V[2,998] +2 V[3,998] *       │
            │        ┆ V[3,998] +2 V[4,99…                                                                     │
            │ 999    ┆ 2 V[0,999] * V[0,999] +2 V[1,999] * V[1,999] +2 V[2,999] * V[2,999] +2 V[3,999] *       │
            │        ┆ V[3,999] +2 V[4,99…                                                                     │
            └────────┴─────────────────────────────────────────────────────────────────────────────────────────┘
            >>> expr = expr.sum("y")
            >>> print((expr == 3).to_str())
            2 V[0,0] * V[0,0] +2 V[0,1] * V[0,1] +2 V[0,2] * V[0,2] +2 V[0,3] * V[0,3] +2 V[0,4] * V[0,4] … = 3
        """
        dims = self.dimensions
        str_table = self.lhs.to_str(
            include_const_term=False, return_df=True, str_col_name="constraint"
        )
        rhs = self.lhs.constant_terms.with_columns(pl.col(COEF_KEY) * -1)
        rhs = cast_coef_to_string(rhs, drop_ones=False, always_show_sign=False)
        rhs = rhs.rename({COEF_KEY: "rhs"})
        if dims:
            constr_str = str_table.join(
                rhs, on=dims, how="left", maintain_order="left", coalesce=True
            )
        else:
            constr_str = pl.concat([str_table, rhs], how="horizontal")
        constr_str = constr_str.with_columns(
            pl.concat_str("constraint", pl.lit(f" {self.sense.value} "), "rhs")
        ).drop("rhs")

        if not return_df:
            if self.dimensions is None:
                constr_str = constr_str.item()
            else:
                constr_str = self._add_shape_to_columns(constr_str)
                with Config.print_polars_config:
                    constr_str = repr(constr_str)

        return constr_str

    def __repr__(self) -> str:
        return (
            get_obj_repr(
                self,
                f"'{self.name}'",
                f"({self.lhs.degree(return_str=True)})",
                height=len(self) if self.dimensions else None,
                terms=len(self.lhs.data),
            )
            + "\n"
            + self.to_str()
        )


class Variable(BaseOperableBlock):
    """A decision variable for an optimization model.

    !!! tip
        If `lb` or `ub` are a dimensioned object (e.g. an [Expression][pyoframe.Expression]), they will automatically be [broadcasted](../../learn/concepts/addition.md#adding-expressions-with-differing-dimensions-using-over) to match the variable's dimensions.

    Parameters:
        *indexing_sets:
            If no indexing_sets are provided, a single variable with no dimensions is created.
            Otherwise, a variable is created for each element in the Cartesian product of the indexing_sets (see Set for details on behaviour).
        vtype:
            The type of the variable. Can be either a VType enum or a string. Default is VType.CONTINUOUS.
        lb:
            The lower bound for the variables.
        ub:
            The upper bound for the variables.
        equals:
            When specified, a variable is created for every label in `equals` and a constraint is added to make the variable equal to the provided expression.
            `indexing_sets` cannot be provided when using `equals`.

    Examples:
        >>> import pandas as pd
        >>> m = pf.Model()
        >>> df = pd.DataFrame(
        ...     {"dim1": [1, 1, 2, 2, 3, 3], "dim2": ["a", "b", "a", "b", "a", "b"]}
        ... )
        >>> Variable(df)
        <Variable 'unnamed' height=6>
        ┌──────┬──────┐
        │ dim1 ┆ dim2 │
        │ (3)  ┆ (2)  │
        ╞══════╪══════╡
        │ 1    ┆ a    │
        │ 1    ┆ b    │
        │ 2    ┆ a    │
        │ 2    ┆ b    │
        │ 3    ┆ a    │
        │ 3    ┆ b    │
        └──────┴──────┘

        Variables cannot be used until they're added to the model.

        >>> m.constraint = Variable(df) <= 3
        Traceback (most recent call last):
        ...
        ValueError: Cannot use 'Variable' before it has been added to a model.

        Instead, assign the variable to the model first:
        >>> m.v = Variable(df)
        >>> m.constraint = m.v <= 3

        >>> m.v
        <Variable 'v' height=6>
        ┌──────┬──────┬──────────┐
        │ dim1 ┆ dim2 ┆ variable │
        │ (3)  ┆ (2)  ┆          │
        ╞══════╪══════╪══════════╡
        │ 1    ┆ a    ┆ v[1,a]   │
        │ 1    ┆ b    ┆ v[1,b]   │
        │ 2    ┆ a    ┆ v[2,a]   │
        │ 2    ┆ b    ┆ v[2,b]   │
        │ 3    ┆ a    ┆ v[3,a]   │
        │ 3    ┆ b    ┆ v[3,b]   │
        └──────┴──────┴──────────┘

        >>> m.v2 = Variable(df[["dim1"]])
        Traceback (most recent call last):
        ...
        ValueError: Duplicate rows found in input data.
        >>> m.v3 = Variable(df[["dim1"]].drop_duplicates())
        >>> m.v3
        <Variable 'v3' height=3>
        ┌──────┬──────────┐
        │ dim1 ┆ variable │
        │ (3)  ┆          │
        ╞══════╪══════════╡
        │ 1    ┆ v3[1]    │
        │ 2    ┆ v3[2]    │
        │ 3    ┆ v3[3]    │
        └──────┴──────────┘
    """

    # TODO: Breaking change, remove support for Iterable[AcceptableSets]
    def __init__(
        self,
        *indexing_sets: SetTypes | Iterable[SetTypes],
        lb: Operable | None = None,
        ub: Operable | None = None,
        vtype: VType | VTypeValue = VType.CONTINUOUS,
        equals: Operable | None = None,
    ):
        if equals is not None:
            if isinstance(equals, (float, int)):
                if lb is not None:
                    raise ValueError("Cannot specify 'lb' when 'equals' is a constant.")
                if ub is not None:
                    raise ValueError("Cannot specify 'ub' when 'equals' is a constant.")
                lb = ub = equals
                equals = None
            else:
                assert len(indexing_sets) == 0, (
                    "Cannot specify both 'equals' and 'indexing_sets'"
                )
                equals = equals.to_expr()  # TODO don't rely on monkey patch
                indexing_sets = (equals,)

        data = Set(*indexing_sets).data if len(indexing_sets) > 0 else pl.DataFrame()
        super().__init__(data)

        self.vtype: VType = VType(vtype)
        self._attr = Container(self._set_attribute, self._get_attribute)
        self._equals: Expression | None = equals

        if lb is not None and not isinstance(lb, (float, int)):
            lb: Expression = lb.to_expr()  # TODO don't rely on monkey patch
            if not self.dimensionless:
                lb = lb.over(*self.dimensions)
            self._lb_expr, self.lb = lb, None
        else:
            self._lb_expr, self.lb = None, lb
        if ub is not None and not isinstance(ub, (float, int)):
            ub = ub.to_expr()  # TODO don't rely on monkey patch
            if not self.dimensionless:
                ub = ub.over(*self.dimensions)  # pyright: ignore[reportOptionalIterable]
            self._ub_expr, self.ub = ub, None
        else:
            self._ub_expr, self.ub = None, ub

    @property
    def attr(self) -> Container:
        """Allows reading and writing variable attributes similarly to [Model.attr][pyoframe.Model.attr]."""
        return self._attr

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
                self.data.join(
                    value,
                    on=self.dimensions,
                    maintain_order="left" if Config.maintain_order else None,
                )
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

        ids = self.data.get_column(VAR_KEY).to_list()
        attr = [getter(poi.VariableIndex(v_id), name) for v_id in ids]
        data = self.data.with_columns(pl.Series(attr).alias(col_name))
        return data.select(self._dimensions_unsafe + [col_name])

    def _assign_ids(self):
        assert self._model is not None
        assert self.name is not None

        solver = self._model.solver
        if solver.supports_integer_variables:
            domain = self.vtype._to_poi()
        else:
            if self.vtype != VType.CONTINUOUS:
                raise ValueError(
                    f"Solver {solver.name} does not support integer or binary variables."
                )

        lb = -1e100 if self.lb is None else float(self.lb)
        ub = 1e100 if self.ub is None else float(self.ub)

        poi_add_var = self._model.poi.add_variable

        dims = self.dimensions

        dynamic_names = dims is not None and self._model.solver_uses_variable_names
        if dynamic_names:
            names = concat_dimensions(self.data, prefix=self.name)[
                "concated_dim"
            ].to_list()
            if solver.supports_integer_variables:
                ids = [poi_add_var(domain, lb, ub, name).index for name in names]
            else:
                ids = [poi_add_var(lb, ub, name=name).index for name in names]
        else:
            if self._model.solver_uses_variable_names:
                name = self.name
            elif solver.accelerate_with_repeat_names:
                name = "V"
            else:
                name = ""

            n = 1 if dims is None else len(self.data)

            if solver.supports_integer_variables:
                ids = [poi_add_var(domain, lb, ub, name).index for _ in range(n)]
            else:
                ids = [poi_add_var(lb, ub, name=name).index for _ in range(n)]

        try:
            df = self.data.with_columns(
                pl.Series(ids, dtype=Config.id_dtype).alias(VAR_KEY)
            )
        except TypeError as e:
            raise TypeError(
                f"Number of variables exceeds the current data type ({Config.id_dtype}). Consider increasing the data type by changing Config.id_dtype."
            ) from e

        self._data = df

    def _on_add_to_model(self, model, name):
        super()._on_add_to_model(model, name)
        self._assign_ids()
        if self._lb_expr is not None:
            setattr(model, f"{name}_lb", self._lb_expr <= self)

        if self._ub_expr is not None:
            setattr(model, f"{name}_ub", self <= self._ub_expr)

        if self._equals is not None:
            setattr(model, f"{name}_equals", self == self._equals)

    @classmethod
    def _get_id_column_name(cls):
        return VAR_KEY

    @property
    @unwrap_single_values
    def solution(self):
        """Retrieves a variable's optimal value after the model has been solved.

        Return type is a DataFrame if the variable has dimensions, otherwise it is a single value.
        Binary and integer variables are returned as integers.

        Examples:
            >>> m = pf.Model()
            >>> m.var_continuous = pf.Variable({"dim1": [1, 2, 3]}, lb=5, ub=5)
            >>> m.var_integer = pf.Variable(
            ...     {"dim1": [1, 2, 3]}, lb=4.5, ub=5.5, vtype=pf.VType.INTEGER
            ... )
            >>> m.var_dimensionless = pf.Variable(
            ...     lb=4.5, ub=5.5, vtype=pf.VType.INTEGER
            ... )
            >>> m.var_continuous.solution
            Traceback (most recent call last):
            ...
            RuntimeError: Failed to retrieve solution for variable. Are you sure the model has been solved?
            >>> m.optimize()
            >>> m.var_continuous.solution
            shape: (3, 2)
            ┌──────┬──────────┐
            │ dim1 ┆ solution │
            │ ---  ┆ ---      │
            │ i64  ┆ f64      │
            ╞══════╪══════════╡
            │ 1    ┆ 5.0      │
            │ 2    ┆ 5.0      │
            │ 3    ┆ 5.0      │
            └──────┴──────────┘
            >>> m.var_integer.solution
            shape: (3, 2)
            ┌──────┬──────────┐
            │ dim1 ┆ solution │
            │ ---  ┆ ---      │
            │ i64  ┆ i64      │
            ╞══════╪══════════╡
            │ 1    ┆ 5        │
            │ 2    ┆ 5        │
            │ 3    ┆ 5        │
            └──────┴──────────┘
            >>> m.var_dimensionless.solution
            5
        """
        try:
            solution = self.attr.Value
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to retrieve solution for variable. Are you sure the model has been solved?"
            ) from e
        if isinstance(solution, pl.DataFrame):
            solution = solution.rename({"Value": SOLUTION_KEY})

        if self.vtype in [VType.BINARY, VType.INTEGER]:
            if isinstance(solution, pl.DataFrame):
                # TODO handle values that are out of bounds of Int64 (i.e. when problem is unbounded)
                solution = solution.with_columns(
                    pl.col("solution").alias("solution_float"),
                    pl.col("solution").round().cast(pl.Int64),
                )
                if Config.integer_tolerance != 0:
                    df = solution.filter(
                        (pl.col("solution_float") - pl.col("solution")).abs()
                        > Config.integer_tolerance
                    )
                    assert df.is_empty(), (
                        f"Variable {self.name} has a non-integer value: {df}\nThis should not happen."
                    )
                solution = solution.drop("solution_float")
            else:
                solution_float = solution
                solution = int(round(solution))
                if Config.integer_tolerance != 0:
                    assert abs(solution - solution_float) < Config.integer_tolerance, (
                        f"Value of variable {self.name} is not an integer: {solution}. This should not happen."
                    )

        return solution

    def __repr__(self):
        result = (
            get_obj_repr(
                self,
                f"'{self.name}'",
                lb=self.lb,
                ub=self.ub,
                height=self.data.height if self.dimensions else None,
            )
            + "\n"
        )
        if self._has_ids:
            result += self.to_expr().to_str(str_col_name="variable")
        else:
            with Config.print_polars_config:
                data = self._add_shape_to_columns(self.data)
                # we don't try to include the allowed_new_dims because there are none for Variables (only exist on Expression or Sets)
                result += repr(data)

        return result

    def to_expr(self) -> Expression:
        """Converts the Variable to an Expression."""
        self._assert_has_ids()
        return self._new(self.data.drop(SOLUTION_KEY, strict=False), self.name)  # pyright: ignore[reportArgumentType], we know it's safe after _assert_has_ids()

    def _new(self, data: pl.DataFrame, name: str) -> Expression:
        self._assert_has_ids()
        e = Expression(data.with_columns(pl.lit(1.0).alias(COEF_KEY)), name)
        e._model = self._model
        return e

    @return_new
    def next(self, dim: str, wrap_around: bool = False):
        """Creates an expression where the variable at each label is the next variable in the specified dimension.

        Parameters:
            dim:
                The dimension over which to shift the variable.
            wrap_around:
                If `True`, the last label in the dimension is connected to the first label.

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
            pyoframe._constants.PyoframeError: Cannot subtract the two expressions below because expression 1 has extra labels.
            Expression 1:	(bat_charge + bat_flow)
            Expression 2:	bat_charge.next(…)
            Extra labels in expression 1:
            ┌───────┬─────────┐
            │ time  ┆ city    │
            ╞═══════╪═════════╡
            │ 18:00 ┆ Toronto │
            │ 18:00 ┆ Berlin  │
            └───────┴─────────┘
            Use .drop_extras() or .keep_extras() to indicate how the extra labels should be handled. Learn more at
                https://bravos-power.github.io/pyoframe/latest/learn/concepts/addition

            >>> (m.bat_charge + m.bat_flow).drop_extras() == m.bat_charge.next("time")
            <Constraint 'unnamed' (linear) height=6 terms=18>
            ┌───────┬─────────┬────────────────────────────────────────────────────────────────────────────────┐
            │ time  ┆ city    ┆ constraint                                                                     │
            │ (3)   ┆ (2)     ┆                                                                                │
            ╞═══════╪═════════╪════════════════════════════════════════════════════════════════════════════════╡
            │ 00:00 ┆ Toronto ┆ bat_charge[00:00,Toronto] + bat_flow[00:00,Toronto]                            │
            │       ┆         ┆ - bat_charge[06:00,Toronto] = 0                                                │
            │ 00:00 ┆ Berlin  ┆ bat_charge[00:00,Berlin] + bat_flow[00:00,Berlin] - bat_charge[06:00,Berlin]   │
            │       ┆         ┆ = 0                                                                            │
            │ 06:00 ┆ Toronto ┆ bat_charge[06:00,Toronto] + bat_flow[06:00,Toronto]                            │
            │       ┆         ┆ - bat_charge[12:00,Toronto] = 0                                                │
            │ 06:00 ┆ Berlin  ┆ bat_charge[06:00,Berlin] + bat_flow[06:00,Berlin] - bat_charge[12:00,Berlin]   │
            │       ┆         ┆ = 0                                                                            │
            │ 12:00 ┆ Toronto ┆ bat_charge[12:00,Toronto] + bat_flow[12:00,Toronto]                            │
            │       ┆         ┆ - bat_charge[18:00,Toronto] = 0                                                │
            │ 12:00 ┆ Berlin  ┆ bat_charge[12:00,Berlin] + bat_flow[12:00,Berlin] - bat_charge[18:00,Berlin]   │
            │       ┆         ┆ = 0                                                                            │
            └───────┴─────────┴────────────────────────────────────────────────────────────────────────────────┘

            >>> (m.bat_charge + m.bat_flow) == m.bat_charge.next(
            ...     "time", wrap_around=True
            ... )
            <Constraint 'unnamed' (linear) height=8 terms=24>
            ┌───────┬─────────┬────────────────────────────────────────────────────────────────────────────────┐
            │ time  ┆ city    ┆ constraint                                                                     │
            │ (4)   ┆ (2)     ┆                                                                                │
            ╞═══════╪═════════╪════════════════════════════════════════════════════════════════════════════════╡
            │ 00:00 ┆ Toronto ┆ bat_charge[00:00,Toronto] + bat_flow[00:00,Toronto]                            │
            │       ┆         ┆ - bat_charge[06:00,Toronto] = 0                                                │
            │ 00:00 ┆ Berlin  ┆ bat_charge[00:00,Berlin] + bat_flow[00:00,Berlin] - bat_charge[06:00,Berlin]   │
            │       ┆         ┆ = 0                                                                            │
            │ 06:00 ┆ Toronto ┆ bat_charge[06:00,Toronto] + bat_flow[06:00,Toronto]                            │
            │       ┆         ┆ - bat_charge[12:00,Toronto] = 0                                                │
            │ 06:00 ┆ Berlin  ┆ bat_charge[06:00,Berlin] + bat_flow[06:00,Berlin] - bat_charge[12:00,Berlin]   │
            │       ┆         ┆ = 0                                                                            │
            │ 12:00 ┆ Toronto ┆ bat_charge[12:00,Toronto] + bat_flow[12:00,Toronto]                            │
            │       ┆         ┆ - bat_charge[18:00,Toronto] = 0                                                │
            │ 12:00 ┆ Berlin  ┆ bat_charge[12:00,Berlin] + bat_flow[12:00,Berlin] - bat_charge[18:00,Berlin]   │
            │       ┆         ┆ = 0                                                                            │
            │ 18:00 ┆ Toronto ┆ bat_charge[18:00,Toronto] + bat_flow[18:00,Toronto]                            │
            │       ┆         ┆ - bat_charge[00:00,Toronto] = 0                                                │
            │ 18:00 ┆ Berlin  ┆ bat_charge[18:00,Berlin] + bat_flow[18:00,Berlin] - bat_charge[00:00,Berlin]   │
            │       ┆         ┆ = 0                                                                            │
            └───────┴─────────┴────────────────────────────────────────────────────────────────────────────────┘

        """
        wrapped = (
            self.data.select(dim)
            .unique(maintain_order=Config.maintain_order)
            .sort(by=dim)
        )
        wrapped = wrapped.with_columns(pl.col(dim).shift(-1).alias("__next"))
        if wrap_around:
            wrapped = wrapped.with_columns(pl.col("__next").fill_null(pl.first(dim)))
        else:
            wrapped = wrapped.drop_nulls(dim)

        expr = self.to_expr()
        data = expr.data.rename({dim: "__prev"})

        data = data.join(
            wrapped,
            left_on="__prev",
            right_on="__next",
            # We use "right" instead of "left" to maintain consistency with the behavior without maintain_order
            maintain_order="right" if Config.maintain_order else None,
        ).drop(["__prev", "__next"], strict=False)

        return data
