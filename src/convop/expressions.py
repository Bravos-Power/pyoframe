from typing import Iterable, Self, List, Set

import polars as pl
from convop.expressionable import Expressionable


CONST_KEY = "__constants"
VAR_KEY = "__variables"
COEF_KEY = "__coefficients"
RESERVED_KEYS = (CONST_KEY, VAR_KEY, COEF_KEY)


class Expression(Expressionable):
    """A linear expression."""

    def __init__(
        self,
        constants: pl.DataFrame | None = None,
        variables: pl.DataFrame | None = None,
    ):
        if constants is None:
            assert variables is not None
            constants = (
                variables.clear()
                .select(_get_dimensions(variables))
                .with_columns(pl.lit(0.0).alias(CONST_KEY))
            )
        if variables is None:
            variables = (
                constants.clear()
                .select(_get_dimensions(constants))
                .with_columns(
                    pl.lit(0).cast(pl.UInt32).alias(VAR_KEY),
                    pl.lit(0.0).alias(COEF_KEY),
                )
            )

        assert CONST_KEY in constants.columns
        assert VAR_KEY in variables.columns
        assert COEF_KEY in variables.columns

        dim = _get_dimensions(variables)
        assert (
            _get_dimensions(constants) == dim
        ), f"Dimensions do not match. {_get_dimensions(constants)} != {dim}"
        assert not variables.drop(COEF_KEY).is_duplicated().any()
        if len(dim) > 0:
            if len(constants) > 0:
                assert not constants.drop(CONST_KEY).is_duplicated().any()
        else:
            assert len(constants) <= 1

        # constants = constants.with_columns(pl.col(CONST_KEY).cast(pl.Float64))
        # variables = variables.with_columns(pl.col(COEF_KEY).cast(pl.Float64))

        self._constants: pl.DataFrame = constants
        self._variables: pl.DataFrame = variables

    @property
    def constants(self) -> pl.DataFrame:
        return self._constants

    @property
    def variables(self) -> pl.DataFrame:
        return self._variables

    @property
    def dimensions(self) -> Set[str]:
        dim_consts = _get_dimensions(self.constants)
        dim_vars = _get_dimensions(self.variables)
        assert dim_consts == dim_vars
        dims = set(dim_consts)
        assert len(dims) == len(dim_consts)
        return dims

    def sum(self, over: str | Iterable[str]):
        if isinstance(over, str):
            over = {over}
        over = set(over)

        dims = self.dimensions
        assert over <= dims
        remaining_dims = dims - over

        constants = self.constants.drop(over)
        variables = self.variables.drop(over)

        constants = (
            constants.group_by(remaining_dims) if len(remaining_dims) > 0 else constants
        ).sum()

        return Expression(
            constants,
            variables.group_by(remaining_dims | {VAR_KEY}).sum(),
        )

    def __add__(self, other):
        other = other.to_expression()
        dims = self.dimensions
        assert dims == other.dimensions

        constants = _align_and_concat(self.constants, other.constants)
        constants = (constants.group_by(dims) if len(dims) > 0 else constants).sum()

        return Expression(
            constants,
            _align_and_concat(self.variables, other.variables)
            .group_by(dims | {VAR_KEY})
            .sum(),
        )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Expression(
                self.constants.with_columns(pl.col(CONST_KEY) * other),
                self.variables.with_columns(pl.col(COEF_KEY) * other),
            )

        other = other.to_expression()

        if len(self.variables) == 0:
            self, other = other, self

        assert (
            len(other.variables) == 0
        ), "Multiplication of two expressions with variables is non-linear and not supported."

        dims_in_common = tuple(self.dimensions & other.dimensions)

        constants = (
            self.constants.join(other.constants, on=dims_in_common)
            .with_columns(pl.col(CONST_KEY) * pl.col(CONST_KEY + "_right"))
            .drop(CONST_KEY + "_right")
        )

        variables = (
            self.variables.join(other.constants, on=dims_in_common)
            .with_columns(pl.col(COEF_KEY) * pl.col(CONST_KEY))
            .drop(CONST_KEY)
        )

        return Expression(constants, variables)

    def to_expression(self) -> Self:
        return self

    def __repr__(self) -> str:
        return f"Constants: {self.constants}\nVariables: {self.variables}"


def _get_dimensions(df: pl.DataFrame) -> List[str]:
    return [x for x in df.columns if x not in RESERVED_KEYS]


def _align_and_concat(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    assert sorted(left.columns) == sorted(right.columns)
    right = right.select(left.columns)
    return pl.concat([left, right], how="vertical_relaxed")
