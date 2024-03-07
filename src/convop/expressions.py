from typing import Self, Sequence

import polars as pl

CONSTANTS_KEY = "__constants"
VARIABLES_KEY = "__variables"
COEFFICIENTS_KEY = "__coefficients"
RESERVED_KEYS = (CONSTANTS_KEY, VARIABLES_KEY, COEFFICIENTS_KEY)


class Expressionable:
    """Any object that can be converted into an expression."""

    def to_expression(self):
        """Convert the object into an Expression."""
        raise NotImplementedError(
            "to_expression must be implemented in subclass " + self.__class__.__name__
        )

    def sum(self, over: str | Sequence[str]):
        return self.to_expression().sum(over)

    def __add__(self, other):
        return self.to_expression() + other

    def __sub__(self, other):
        return self.to_expression() - other

    def __mul__(self, other):
        return self.to_expression() * other


class Expression(Expressionable):
    """A linear expression."""

    def __init__(
        self,
        constants: pl.DataFrame | None,
        variables: pl.DataFrame | None,
    ):
        self.constants = constants
        self.variables = variables
        if constants is not None and variables is not None:
            constants_columns = set(constants.columns).difference(RESERVED_KEYS)
            variables_columns = set(variables.columns).difference(RESERVED_KEYS)
            assert constants_columns == variables_columns
            assert constants.select(constants_columns).n_unique() == len(constants)
            assert variables.select(variables_columns).n_unique() == len(constants)

    def sum(self, over: str | Sequence[str]):
        if isinstance(over, str):
            over = [over]

        constants = None
        if self.constants is not None:
            remaining_columns = (
                set(self.constants.columns).difference(RESERVED_KEYS).difference(over)
            )
            constants = self.constants.group_by(remaining_columns).sum()
        variables = None
        if self.variables is not None:
            variables = self.variables.drop(over)

        return Expression(constants, variables)

    def __add__(self, other):
        other = other.to_expression()

        if self.variables is not None and other.variables is not None:
            assert self.variables.columns == other.variables.columns
            # Get all columns except COEFFICIENTS_KEY
            variables = (
                pl.concat([self.variables, other.variables])
                .group_by(set(self.variables.columns).difference([COEFFICIENTS_KEY]))
                .sum()
            )
        elif self.variables is not None:
            variables = self.variables
        elif other.variables is not None:
            variables = other.variables
        else:
            variables = None

        if self.constants is not None and other.constants is not None:
            assert self.constants.columns == other.constants.columns
            constants = (
                pl.concat([self.constants, other.constants])
                .group_by(set(self.constants.columns).difference([CONSTANTS_KEY]))
                .sum()
            )
        elif self.constants is not None:
            constants = self.constants
        elif other.constants is not None:
            constants = other.constants
        else:
            constants = None

        return Expression(constants, variables)

    def __sub__(self, other):
        assert isinstance(other, Expression)
        return self + (other * -1)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            constants = None
            if self.constants is not None:
                constants = self.constants.with_columns(
                    pl.col(CONSTANTS_KEY).mul(other)
                )

            variables = None
            if self.variables is not None:
                variables = self.variables.with_columns(
                    pl.col(COEFFICIENTS_KEY).mul(other)
                )

            return Expression(constants, variables)
        else:
            other = other.to_expression()

            if other.variables is not None and self.variables is not None:
                raise ValueError(
                    "Multiplication of two expressions with variables is not supported."
                )

            if other.variables is not None:
                return other * self
            # Now other doesn't have any variables.
            assert other.variables is None
            assert other.constants is not None

            constant_multiplier = other.constants

            constants = None
            if self.constants is not None:
                indexes_in_common = set(constant_multiplier.columns).intersection(
                    self.constants.columns
                )
                constants = self.constants.join(
                    constant_multiplier, on=tuple(indexes_in_common)
                )
                left_coef, right_coef = CONSTANTS_KEY, CONSTANTS_KEY + "_right"
                constants = constants.with_columns(
                    pl.col(left_coef).mul(pl.col(right_coef))
                ).drop(right_coef)

            variables = None
            if self.variables is not None:
                indexes_in_common = set(constant_multiplier.columns).intersection(
                    self.variables.columns
                )
                variables = self.variables.join(
                    constant_multiplier, on=tuple(indexes_in_common)
                )
                variables = variables.with_columns(
                    pl.col(COEFFICIENTS_KEY).mul(pl.col(CONSTANTS_KEY))
                ).drop(CONSTANTS_KEY)

            return Expression(constants, variables)

    def to_expression(self) -> Self:
        return self

    def __repr__(self) -> str:
        return f"Constants: {self.constants}\nVariables: {self.variables}"

    def __len__(self):
        if self.constants is not None:
            return len(self.constants)
        assert self.variables is not None
        return self.variables.select(
            set(self.variables.columns).difference(RESERVED_KEYS)
        ).n_unique()


def sum(over: str | Sequence[str], expr: Expressionable) -> Expression:
    return expr.sum(over)
