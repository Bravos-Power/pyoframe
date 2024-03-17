from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, overload

import polars as pl

from pyoframe.dataframe import COEF_KEY, VAR_KEY, concat_dimensions
from pyoframe.var_mapping import DEFAULT_MAP, NamedVariables
from pyoframe.model_element import ModelElement, FrameWrapper

if TYPE_CHECKING:
    from pyoframe.model import Model

VAR_CONST = 0
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
        <Constraint name=unnamed sense='<=' len=1 dimensions={}>
        """
        return build_constraint(self, other, ConstraintSense.LE)

    def __ge__(self, other):
        """Equality constraint.
        Examples
        >>> from pyoframe import Variable
        >>> Variable() >= 1
        <Constraint name=unnamed sense='>=' len=1 dimensions={}>
        """
        return build_constraint(self, other, ConstraintSense.GE)

    def __eq__(self, __value: object):
        """Equality constraint.
        Examples
        >>> from pyoframe import Variable
        >>> Variable() == 1
        <Constraint name=unnamed sense='=' len=1 dimensions={}>
        """
        return build_constraint(self, __value, ConstraintSense.EQ)


class Expression(Expressionable, FrameWrapper):
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
        [1,mon]: 1.0 Time[1,mon] +1.0 Size[1,mon]
        [1,tue]: 2.0 Time[1,tue] +2.0 Size[1,tue]
        [1,wed]: 3.0 Time[1,wed] +3.0 Size[1,wed]
        [2,mon]: 4.0 Time[2,mon] +4.0 Size[2,mon]
        [2,tue]: 5.0 Time[2,tue] +5.0 Size[2,tue]
        """
        # Sanity checks, at least VAR_KEY or COEF_KEY must be present
        assert len(data.columns) == len(set(data.columns))
        assert VAR_KEY in data.columns or COEF_KEY in data.columns

        # Add missing columns if needed
        if VAR_KEY not in data.columns:
            data = data.with_columns(pl.lit(VAR_CONST).cast(VAR_TYPE).alias(VAR_KEY))
        if COEF_KEY not in data.columns:
            data = data.with_columns(pl.lit(1.0).alias(COEF_KEY))

        # Sanity checks
        assert (
            not data.drop(COEF_KEY).is_duplicated().any()
        ), "There are duplicata indices"

        # Cast to proper datatypes (TODO check if needed)
        data = data.with_columns(
            pl.col(COEF_KEY).cast(pl.Float64), pl.col(VAR_KEY).cast(VAR_TYPE)
        )

        self._model = model
        super().__init__(data)

    def indices_match(self, other: Expression):
        # Check that the indices match
        dims = self.dimensions
        assert set(dims) == set(other.dimensions)
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

    def within(self, by: Expressionable):
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
        by = by.to_expr()
        dims = self.dimensions
        dims_in_common = [dim for dim in dims if dim in by.dimensions]
        by_dims = by.data.select(dims_in_common).unique()
        return self._new(self.data.join(by_dims, on=dims_in_common))

    def with_columns(self, polars_expr: pl.Expr) -> "Expression":
        """Returns a new Expression after having transfered the inner .data using Polars' with_columns function."""
        return self._new(self.data.with_columns(polars_expr))

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
        """
        if isinstance(other, (int, float)):
            return self.with_columns(
                pl.when(pl.col(VAR_KEY) == VAR_CONST)
                .then(pl.col(COEF_KEY) + other)
                .otherwise(pl.col(COEF_KEY))
                .alias(COEF_KEY)
            )

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

        if (other.data.get_column(VAR_KEY) != VAR_CONST).any():
            self, other = other, self

        if (other.data.get_column(VAR_KEY) != VAR_CONST).any():
            raise ValueError(
                "Multiplication of two expressions with variables is non-linear and not supported."
            )
        multiplier = other.data.drop(VAR_KEY)

        dims_in_common = [dim for dim in self.dimensions if dim in other.dimensions]

        data = (
            self.data.join(multiplier, on=dims_in_common)
            .with_columns(pl.col(COEF_KEY) * pl.col(COEF_KEY + "_right"))
            .drop(COEF_KEY + "_right")
        )
        return self._new(data)

    def to_expr(self) -> Expression:
        return self

    def _new(self, data: pl.DataFrame) -> Expression:
        return Expression(data, model=self._model)

    @property
    def constant_terms(self):
        dims = self.dimensions
        return (
            self.data.filter(pl.col(VAR_KEY) == VAR_CONST)
            .drop(VAR_KEY)
            .join(self.data.select(dims).unique(), on=dims, how="outer_coalesce")
            .fill_null(0.0)
        )

    @property
    def variable_terms(self):
        return self.data.filter(pl.col(VAR_KEY) != VAR_CONST)

    def to_str_table(
        self, max_line_len=None, max_rows=None, include_const_term=True, var_map=None
    ):
        data = self.data if include_const_term else self.variable_terms
        if var_map is None:
            var_map = (
                self._model.var_map if self._model is not None else DEFAULT_MAP
            )
        data = var_map.map_vars(data)
        dimensions = self.dimensions

        # Create a string for each term
        data = data.with_columns(
            expr=pl.concat_str(
                pl.when(pl.col(COEF_KEY) < 0)
                .then(pl.lit(" -"))
                .otherwise(pl.lit(" +")),
                pl.col(COEF_KEY).abs(),
                pl.lit(" "),
                pl.col(VAR_KEY),
            )
        ).drop(COEF_KEY, VAR_KEY)

        # Combine terms into one string
        if dimensions:
            data = data.group_by(dimensions, maintain_order=True).agg(
                pl.col("expr").str.concat(delimiter="")
            )
        else:
            data = data.select(pl.col("expr").str.concat(delimiter=""))

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
        prefix = getattr(self, "name") if hasattr(self, "name") else None
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
        include_header=True,
    ):
        str_table = self.to_str_table(
            max_line_len=max_line_len,
            max_rows=max_rows,
            include_const_term=include_const_term,
            var_map=var_map,
        )
        result = str_table.select(pl.col("expr").str.concat(delimiter="\n")).item()
        if include_header:
            result = (
                f"<Expression size={len(self)} dimensions={self.shape} terms={len(self.data)}>\n"
                + result
            )
        return result

    def __repr__(self) -> str:
        return self.to_str(max_line_len=80, max_rows=5)


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


def build_constraint(lhs: Expressionable, rhs, sense):
    lhs = lhs.to_expr()
    if not isinstance(rhs, (int, float)):
        rhs = rhs.to_expr()
        if not lhs.indices_match(rhs):
            raise ValueError("LHS and RHS values have different indices")
    return Constraint(lhs - rhs, sense)


class Constraint(Expression, ModelElement):
    def __init__(
        self,
        lhs: Expression,
        sense: ConstraintSense,
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
        super().__init__(lhs.data)
        self.sense = sense

    def __repr__(self):
        return f"""<Constraint name={self.name} sense='{self.sense.value}' len={len(self)} dimensions={self.shape}>"""
