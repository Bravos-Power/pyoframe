from enum import Enum
from typing import Iterable, Self, Sequence, overload

import polars as pl

from pyoframe.model_element import COEF_KEY, VAR_KEY, ModelElement, FrameWrapper

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

    def __init__(self, data: pl.DataFrame):
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

        super().__init__(data)

    def indices_match(self, other: Self):
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
        >>> expr.data.sort(["item", "__variable_id"])
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
        assert set(over) <= set(dims)
        remaining_dims = [dim for dim in dims if dim not in over]

        return Expression(
            self.data.drop(over).group_by(remaining_dims + [VAR_KEY]).sum()
        )

    def within(self, by: Expressionable):
        """
        Examples
        >>> import pandas as pd
        >>> general_expr = pd.DataFrame({"dim1": [1, 2, 3], "value": [1, 2, 3]}).to_expr()
        >>> filter_expr = pd.DataFrame({"dim1": [1, 3], "value": [5, 6]}).to_expr()
        >>> general_expr.within(filter_expr).data.sort(["dim1", "__variable_id"])
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
        return Expression(self.data.join(by_dims, on=dims_in_common))

    def __add__(self, other):
        """
        Examples
        --------
        >>> import pandas as pd
        >>> from pyoframe import Variable
        >>> add = pd.DataFrame({"dim1": [1,2,3], "add": [10, 20, 30]}).set_index("dim1")["add"]
        >>> var = Variable(add.index)
        >>> expr = var + add
        >>> expr.data.sort(["dim1", "__variable_id"])
        shape: (6, 3)
        ┌──────┬─────────┬───────────────┐
        │ dim1 ┆ __coeff ┆ __variable_id │
        │ ---  ┆ ---     ┆ ---           │
        │ i64  ┆ f64     ┆ u32           │
        ╞══════╪═════════╪═══════════════╡
        │ 1    ┆ 10.0    ┆ 0             │
        │ 1    ┆ 1.0     ┆ 1             │
        │ 2    ┆ 20.0    ┆ 0             │
        │ 2    ┆ 1.0     ┆ 2             │
        │ 3    ┆ 30.0    ┆ 0             │
        │ 3    ┆ 1.0     ┆ 3             │
        └──────┴─────────┴───────────────┘
        >>> expr += 2
        >>> expr.data.sort(["dim1", "__variable_id"])
        shape: (6, 3)
        ┌──────┬─────────┬───────────────┐
        │ dim1 ┆ __coeff ┆ __variable_id │
        │ ---  ┆ ---     ┆ ---           │
        │ i64  ┆ f64     ┆ u32           │
        ╞══════╪═════════╪═══════════════╡
        │ 1    ┆ 12.0    ┆ 0             │
        │ 1    ┆ 1.0     ┆ 1             │
        │ 2    ┆ 22.0    ┆ 0             │
        │ 2    ┆ 1.0     ┆ 2             │
        │ 3    ┆ 32.0    ┆ 0             │
        │ 3    ┆ 1.0     ┆ 3             │
        └──────┴─────────┴───────────────┘
        >>> expr += pd.DataFrame({"dim1": [1,2], "add": [10, 20]}).set_index("dim1")["add"]
        >>> expr.data.sort(["dim1", "__variable_id"])
        shape: (6, 3)
        ┌──────┬─────────┬───────────────┐
        │ dim1 ┆ __coeff ┆ __variable_id │
        │ ---  ┆ ---     ┆ ---           │
        │ i64  ┆ f64     ┆ u32           │
        ╞══════╪═════════╪═══════════════╡
        │ 1    ┆ 22.0    ┆ 0             │
        │ 1    ┆ 1.0     ┆ 1             │
        │ 2    ┆ 42.0    ┆ 0             │
        │ 2    ┆ 1.0     ┆ 2             │
        │ 3    ┆ 32.0    ┆ 0             │
        │ 3    ┆ 1.0     ┆ 3             │
        └──────┴─────────┴───────────────┘
        """
        if isinstance(other, (int, float)):
            return Expression(
                self.data.with_columns(
                    pl.when(pl.col(VAR_KEY) == VAR_CONST)
                    .then(pl.col(COEF_KEY) + other)
                    .otherwise(pl.col(COEF_KEY))
                    .alias(COEF_KEY)
                ),
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
        data = data.group_by(dims + [VAR_KEY]).sum()

        return Expression(data)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Expression(self.data.with_columns(pl.col(COEF_KEY) * other))

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
        return Expression(data)

    def to_expr(self) -> Self:
        return self

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

    def __repr__(self) -> str:
        return f"<Expression size={len(self)} dimensions={self.shape}>"


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
