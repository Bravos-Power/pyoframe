from __future__ import annotations
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    overload,
    Union,
)

import pandas as pd
import polars as pl

from pyoframe.dataframe import (
    COEF_KEY,
    CONST_TERM,
    RESERVED_COL_KEYS,
    VAR_KEY,
    cast_coef_to_string,
    concat_dimensions,
    get_dimensions,
)
from pyoframe.util import _parse_inputs_as_iterable
from pyoframe.var_mapping import DEFAULT_MAP
from pyoframe.model_element import ModelElement

if TYPE_CHECKING:
    from pyoframe.model import Model

VAR_TYPE = pl.UInt32


class ConstraintSense(Enum):
    LE = "<="
    GE = ">="
    EQ = "="


class Expressionable:
    """Any object that can be converted into an expression."""

    def to_expr(self) -> "Expression":
        """Converts the object into an Expression."""
        raise NotImplementedError(
            "to_expr must be implemented in subclass " + self.__class__.__name__
        )

    def __add__(self, other):
        return self.to_expr() + other

    def __neg__(self):
        return self.to_expr() * -1

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
        return self.to_expr() + (other * -1)

    def __mul__(self, other):
        return self.to_expr() * other

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
        return build_constraint(self, other, ConstraintSense.LE)

    def __ge__(self, other):
        """Equality constraint.
        Examples
        >>> from pyoframe import Variable
        >>> Variable() >= 1
        <Constraint sense='>=' size=1 dimensions={} terms=2>
        x1 >= 1
        """
        return build_constraint(self, other, ConstraintSense.GE)

    def __eq__(self, __value: object):
        """Equality constraint.
        Examples
        >>> from pyoframe import Variable
        >>> Variable() == 1
        <Constraint sense='=' size=1 dimensions={} terms=2>
        x1 = 1
        """
        return build_constraint(self, __value, ConstraintSense.EQ)


AcceptableSets = Union[
    pl.DataFrame,
    pd.Index,
    pd.DataFrame,
    Expressionable,
    Mapping[str, Sequence[object]],
    "Set",
]

class Set(ModelElement, Expressionable):
    def __init__(self, *data: AcceptableSets | Iterable[AcceptableSets], **named_data):
        data_list = list(data)
        for name, set in named_data.items():
            data_list.append({name: set})
        df = self._parse_acceptable_sets(*data_list)
        if df.is_duplicated().any():
            raise ValueError("Duplicate rows found in data.")
        super().__init__(df)

    def _new(self, data: pl.DataFrame):
        s = Set(data)
        s._model = self._model
        return s

    @staticmethod
    def _parse_acceptable_sets(
        *over: AcceptableSets | Iterable[AcceptableSets],
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
        over_iter: Iterable[AcceptableSets] = _parse_inputs_as_iterable(over)

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
            data=self.data.with_columns(
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
            raise ValueError("Cannot add two sets.")
        return super().__add__(other)

    def __repr__(self):
        return f"""<Set{' name='+self.name if self.name is not None else ''} size={self.data.height} dimensions={self.shape}>\n{self.to_expr().to_str(max_line_len=80, max_rows=10)}"""


    @staticmethod
    def _set_to_polars(set: "AcceptableSets") -> pl.DataFrame:
        if isinstance(set, dict):
            df = pl.DataFrame(set)
        elif isinstance(set, Expressionable):
            df = set.to_expr().data.drop(RESERVED_COL_KEYS).unique(maintain_order=True)
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


class Expression(Expressionable, ModelElement):
    """A linear expression."""

    def __init__(self, data: pl.DataFrame, model: Optional["Model"] = None):
        """
        >>> import pandas as pd
        >>> from pyoframe import Variable, Model
        >>> df = pd.DataFrame({"item" : [1, 1, 1, 2, 2], "time": ["mon", "tue", "wed", "mon", "tue"], "cost": [1, 2, 3, 4, 5]}).set_index(["item", "time"])
        >>> m = Model()
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
        if data.drop(COEF_KEY).is_duplicated().any():
            duplicated_data = data.filter(data.drop(COEF_KEY).is_duplicated())
            raise ValueError(f"Duplicate indices found:\n{duplicated_data}.")

        super().__init__(data, model=model)

    def indices_match(self, other: Expression):
        # Check that the indices match
        dims = self.dimensions
        assert set(dims) == set(
            other.dimensions
        ), f"Dimensions do not match: {dims} != {other.dimensions}"
        if len(dims) == 0:
            return  # No indices

        unique_dims_left = self.data.select(dims).unique()
        unique_dims_right = other.data.select(dims).unique()
        return len(unique_dims_left) == len(
            unique_dims_left.join(unique_dims_right, on=dims)
        )

    

    def sum(self, over: str | Iterable[str]):
        """
        Examples
        --------
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
        assert set(over) <= set(dims), f"Cannot sum over {over} as it is not in {dims}"
        remaining_dims = [dim for dim in dims if dim not in over]

        return self._new(
            self.data.drop(over)
            .group_by(remaining_dims + [VAR_KEY], maintain_order=True)
            .sum()
        )

    def rolling_sum(self, over: str, window_size: int):
        """
        Calculates the rolling sum of the Expression over a specified window size for a given dimension.

        This method applies a rolling sum operation over the dimension specified by `over`,
        using a window defined by `window_size`.


        Parameters
        ----------
        over : str
               The name of the dimension (column) over which the rolling sum is calculated.
               This dimension must exist within the Expression's dimensions.
        window_size : int
               The size of the moving window in terms of number of records.
               The rolling sum is calculated over this many consecutive elements.

        Returns
        -------
        Expression
               A new Expression instance containing the result of the rolling sum operation.
               This new Expression retains all dimensions (columns) of the original data,
               with the rolling sum applied over the specified dimension.

        Examples
        --------
        >>> import polars as pl
        >>> from pyoframe import Variable, Model
        >>> cost = pl.DataFrame({"item" : [1, 1, 1, 2, 2], "time": [1, 2, 3, 1, 2], "cost": [1, 2, 3, 4, 5]})
        >>> m = Model()
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

    def within(self, set: "AcceptableSets") -> Expression:
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
        set_dims = get_dimensions(df)
        dims = self.dimensions
        dims_in_common = [dim for dim in dims if dim in set_dims]
        by_dims = df.select(dims_in_common).unique()
        return self._new(self.data.join(by_dims, on=dims_in_common))



    def __add__(self, other):
        """
        Examples
        --------
        >>> import pandas as pd
        >>> from pyoframe import Variable
        >>> add = pd.DataFrame({"dim1": [1,2,3], "add": [10, 20, 30]}).set_index("dim1")["add"]
        >>> var = Variable(add.index)
        >>> expr = var + add
        >>> expr.data
        shape: (6, 3)
        ┌──────┬─────────┬───────────────┐
        │ dim1 ┆ __coeff ┆ __variable_id │
        │ ---  ┆ ---     ┆ ---           │
        │ i64  ┆ f64     ┆ u32           │
        ╞══════╪═════════╪═══════════════╡
        │ 1    ┆ 1.0     ┆ 1             │
        │ 2    ┆ 1.0     ┆ 2             │
        │ 3    ┆ 1.0     ┆ 3             │
        │ 1    ┆ 10.0    ┆ 0             │
        │ 2    ┆ 20.0    ┆ 0             │
        │ 3    ┆ 30.0    ┆ 0             │
        └──────┴─────────┴───────────────┘
        >>> expr += 2
        >>> expr.data
        shape: (6, 3)
        ┌──────┬─────────┬───────────────┐
        │ dim1 ┆ __coeff ┆ __variable_id │
        │ ---  ┆ ---     ┆ ---           │
        │ i64  ┆ f64     ┆ u32           │
        ╞══════╪═════════╪═══════════════╡
        │ 1    ┆ 1.0     ┆ 1             │
        │ 2    ┆ 1.0     ┆ 2             │
        │ 3    ┆ 1.0     ┆ 3             │
        │ 1    ┆ 12.0    ┆ 0             │
        │ 2    ┆ 22.0    ┆ 0             │
        │ 3    ┆ 32.0    ┆ 0             │
        └──────┴─────────┴───────────────┘
        >>> expr += pd.DataFrame({"dim1": [1,2], "add": [10, 20]}).set_index("dim1")["add"]
        >>> expr.data
        shape: (6, 3)
        ┌──────┬─────────┬───────────────┐
        │ dim1 ┆ __coeff ┆ __variable_id │
        │ ---  ┆ ---     ┆ ---           │
        │ i64  ┆ f64     ┆ u32           │
        ╞══════╪═════════╪═══════════════╡
        │ 1    ┆ 1.0     ┆ 1             │
        │ 2    ┆ 1.0     ┆ 2             │
        │ 3    ┆ 1.0     ┆ 3             │
        │ 1    ┆ 22.0    ┆ 0             │
        │ 2    ┆ 42.0    ┆ 0             │
        │ 3    ┆ 32.0    ┆ 0             │
        └──────┴─────────┴───────────────┘
        >>> expr = 5 + 2 * Variable()
        >>> expr
        <Expression size=1 dimensions={} terms=2>
        2 x4 +5
        """
        if isinstance(other, (int, float)):
            return self._add_const(other)

        other = other.to_expr()
        dims = self.dimensions
        assert set(dims) == set(
            other.dimensions
        ), f"Adding expressions with different dimensions, {dims} != {other.dimensions}"

        data, other_data = self.data, other.data

        assert sorted(data.columns) == sorted(other_data.columns)
        other_data = other_data.select(data.columns)
        data = pl.concat([data, other_data], how="vertical_relaxed")
        data = data.group_by(dims + [VAR_KEY], maintain_order=True).sum()

        return self._new(data)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.with_columns(pl.col(COEF_KEY) * other)

        other = other.to_expr()

        if (other.data.get_column(VAR_KEY) != CONST_TERM).any():
            self, other = other, self

        if (other.data.get_column(VAR_KEY) != CONST_TERM).any():
            raise ValueError(
                "Multiplication of two expressions with variables is non-liner and not supported."
            )
        multiplier = other.data.drop(VAR_KEY)

        dims_in_common = [dim for dim in self.dimensions if dim in other.dimensions]

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

    def _new(self, data: pl.DataFrame) -> Expression:
        return Expression(data, model=self._model)

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
                .unique()
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
        if dims:
            return constant_terms.join(
                self.data.select(dims).unique(), on=dims, how="outer_coalesce"
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

    def align(self, other: Expression):
        """Returns a new Expression that is aligned with other, meaning it has the same dimensions,
        and the same coordinates in all dimensions, corresponding to the intersection of the
        coordinates in the common dimensions (inner join over these dimensions).
        If other has extra dimensions, broadcasting is performed.

        Examples
        --------
        >>> import polars as pl
        >>> import pyoframe as pf
        >>> m = pf.Model
        >>> m.x = pf.Variable(pl.DataFrame({"p": [1,2,3]}), pl.DataFrame({"t": [0,1]}))
        >>> y = pf.sum("t", m.x)
        >>> z = y.align(m.x.to_expr())
        >>> set(z.dimensions) == set(m.x.dimensions)
        True
        >>> z.filter(p=1, t=0)
        <Expression size=1 dimensions={'p': 1, 't': 1} terms=2>
        [1,0]: x1 + x2
        >>> z.filter(p=3, t=1)
        <Expression size=1 dimensions={'p': 1, 't': 1} terms=2>
        [3,1]: x5 + x6
        """

        common_dimensions = list(set(self.dimensions).intersection(other.dimensions))

        return self._new(
            self.data
            .lazy()
            .join(
                other.data.lazy().select(other.dimensions).unique(), on=common_dimensions
            )
            .collect(),
        )

    def to_str_table(
        self,
        max_line_len=None,
        max_rows=None,
        include_const_term=True,
        var_map=None,
        include_name=True,
    ):
        data = self.data if include_const_term else self.variable_terms
        if var_map is None:
            var_map = self._model.var_map if self._model is not None else DEFAULT_MAP
        data = cast_coef_to_string(data)
        data = var_map.map_vars(data)
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
        if dimensions:
            data = data.group_by(dimensions, maintain_order=True).agg(
                pl.col("expr").str.concat(delimiter=" ")
            )
        else:
            data = data.select(pl.col("expr").str.concat(delimiter=" "))

        # Remove leading +
        data = data.with_columns(pl.col("expr").str.strip_chars(characters=" +"))

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

        # Prefix with the dimensions
        prefix = (
            getattr(self, "name") if hasattr(self, "name") and include_name else None
        )
        if prefix or dimensions:
            data = concat_dimensions(data, prefix=prefix, ignore_columns=["expr"])
            data = data.with_columns(
                pl.concat_str(
                    pl.col("concated_dim"), pl.lit(": "), pl.col("expr")
                ).alias("expr")
            ).drop("concated_dim")

        return data

    def to_str(
        self,
        max_line_len=None,
        max_rows=None,
        include_const_term=True,
        var_map=None,
        include_name=True,
    ):
        str_table = self.to_str_table(
            max_line_len=max_line_len,
            max_rows=max_rows,
            include_const_term=include_const_term,
            var_map=var_map,
            include_name=include_name,
        )
        result = str_table.select(pl.col("expr").str.concat(delimiter="\n")).item()

        return result

    def __repr__(self) -> str:
        return f"<Expression size={len(self)} dimensions={self.shape} terms={len(self.data)}>\n{self.to_str(max_line_len=80, max_rows=15)}"


@overload
def sum(over: str | Sequence[str], expr: Expressionable): ...


@overload
def sum(over: Expressionable): ...


def sum(
    over: str | Sequence[str] | Expressionable, expr: Expressionable | None = None
) -> "Expression":
    if expr is None:
        assert isinstance(over, Expressionable)
        over = over.to_expr()
        return over.sum(over.dimensions)
    else:
        assert isinstance(over, (str, Sequence))
        return expr.to_expr().sum(over)


def sum_by(by: str | Sequence[str], expr: Expressionable) -> "Expression":
    if isinstance(by, str):
        by = [by]
    expr = expr.to_expr()
    dimensions = expr.dimensions
    remaining_dims = [dim for dim in dimensions if dim not in by]
    return sum(over=remaining_dims, expr=expr)


def build_constraint(lhs: Expressionable, rhs, sense):
    lhs = lhs.to_expr()
    if not isinstance(rhs, (int, float)):
        rhs = rhs.to_expr()
        if not lhs.indices_match(rhs):
            raise ValueError(
                "LHS and RHS values have different indices"
                + str(lhs)
                + "\nvs\n"
                + str(rhs)
            )
    return Constraint(lhs - rhs, sense, model=lhs._model)


class Constraint(Expression):
    def __init__(
        self,
        lhs: Expression | pl.DataFrame,
        sense: ConstraintSense,
        model: Optional["Model"] = None,
    ):
        """Adds a constraint to the model.

        Parameters
        ----------
        data: Expression
            The left hand side of the constraint.
        sense: Sense
            The sense of the constraint.
        rhs: Expression
            The right hand side of the constraint.
        """
        if isinstance(lhs, Expression):
            lhs = lhs.data
        super().__init__(lhs)
        self.sense = sense
        self._model = model

    def to_str(self, max_line_len=None, max_rows=None, var_map=None):
        dims = self.dimensions
        str_table = self.to_str_table(
            max_line_len=max_line_len,
            max_rows=max_rows,
            include_const_term=False,
            var_map=var_map,
        )
        rhs = self.constant_terms.with_columns(pl.col(COEF_KEY) * -1)
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
        return f"<Constraint{' name='+self.name if self.name is not None else ''} sense='{self.sense.value}' size={len(self)} dimensions={self.shape} terms={len(self.data)}>\n{self.to_str(max_line_len=80, max_rows=15)}"

    def _new(self, data: pl.DataFrame):
        return Constraint(data, self.sense, model=self._model)

