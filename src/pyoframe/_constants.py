"""Contains shared constants which are used across the package."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import polars as pl
import pyoptinterface as poi

COEF_KEY = "__coeff"
VAR_KEY = "__variable_id"
QUAD_VAR_KEY = "__quadratic_variable_id"
CONSTRAINT_KEY = "__constraint_id"
SOLUTION_KEY = "solution"
DUAL_KEY = "dual"

KEY_TYPE = pl.UInt32


@dataclass
class _Solver:
    name: SUPPORTED_SOLVER_TYPES
    supports_integer_variables: bool = True
    supports_quadratics: bool = True
    supports_duals: bool = True
    supports_objective_sense: bool = True
    supports_write: bool = True

    def check_supports_integer_variables(self):
        if not self.supports_integer_variables:
            raise ValueError(
                f"Solver {self.name} does not support integer or binary variables."
            )

    def check_supports_write(self):
        if not self.supports_write:
            raise ValueError(f"Solver {self.name} does not support .write()")

    def __repr__(self):
        return self.name


SUPPORTED_SOLVERS = [
    _Solver("gurobi"),
    _Solver("highs", supports_quadratics=False, supports_duals=False),
    _Solver(
        "ipopt",
        supports_integer_variables=False,
        supports_objective_sense=False,
        supports_write=False,
    ),
]


# Variable ID for constant terms. This variable ID is reserved.
CONST_TERM = 0

RESERVED_COL_KEYS = (
    COEF_KEY,
    VAR_KEY,
    QUAD_VAR_KEY,
    CONSTRAINT_KEY,
    SOLUTION_KEY,
    DUAL_KEY,
)


class _ConfigMeta(type):
    """Metaclass for Config that stores the default values of all configuration options."""

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        cls._defaults = {
            k: v
            for k, v in dct.items()
            if not k.startswith("_") and type(v) != classmethod  # noqa: E721 (didn't want to mess with it since it works)
        }


class Config(metaclass=_ConfigMeta):
    """General settings for Pyoframe (for advanced users).

    Accessible via `pf.Config` (see examples below).
    """

    default_solver: SUPPORTED_SOLVER_TYPES | _Solver | Literal["raise", "auto"] = "auto"
    """The solver to use when [Model][pyoframe.Model] is instantiated without specifying a solver.

    If `auto`, Pyoframe will try to use whichever solver is installed.
    If `raise`, an exception will be raised when [Model][pyoframe.Model] is instantiated without specifying a solver.

    We recommend that users specify their solver when instantiating [Model][pyoframe.Model] rather than relying on this option.
    """

    disable_unmatched_checks: bool = False
    """When `True`, improves performance by skipping unmatched checks (not recommended).

    When `True`, unmatched checks are disabled which effectively means that all expressions
    are treated as if they contained [`.keep_unmatched()`][pyoframe.Expression.keep_unmatched]
    (unless [`.drop_unmatched()`][pyoframe.Expression.drop_unmatched] was applied).

    !!! warning
        This might improve performance, but it will suppress the "unmatched" errors that alert developers to unexpected
        behaviors (see [here](../learn/get-started/special-functions.md#drop_unmatched-and-keep_unmatched)).
        Only consider enabling after you have thoroughly tested your code.

    Examples:
        >>> import polars as pl
        >>> population = pl.DataFrame({"city": ["Toronto", "Vancouver", "Montreal"], "pop": [2_731_571, 631_486, 1_704_694]}).to_expr()
        >>> population_influx = pl.DataFrame({"city": ["Toronto", "Vancouver", "Montreal"],"influx": [100_000, 50_000, None],}).to_expr()
        
        Normally, an error warns users that the two expressions have conflicting indices:
        >>> population + population_influx
        Traceback (most recent call last):
        ...
        pyoframe._constants.PyoframeError: Failed to add expressions:
        <Expression height=3 terms=3> + <Expression height=2 terms=2>
        Due to error:
        DataFrame has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()
        shape: (1, 2)
        ┌──────────┬────────────┐
        │ city     ┆ city_right │
        │ ---      ┆ ---        │
        │ str      ┆ str        │
        ╞══════════╪════════════╡
        │ Montreal ┆ null       │
        └──────────┴────────────┘
        
        But if `Config.disable_unmatched_checks = True`, the error is suppressed and the sum is considered to be `population.keep_unmatched() + population_influx.keep_unmatched()`:
        >>> pf.Config.disable_unmatched_checks = True
        >>> population + population_influx
        <Expression height=3 terms=3>
        ┌───────────┬────────────┐
        │ city      ┆ expression │
        │ (3)       ┆            │
        ╞═══════════╪════════════╡
        │ Toronto   ┆ 2831571    │
        │ Vancouver ┆ 681486     │
        │ Montreal  ┆ 1704694    │
        └───────────┴────────────┘
    """

    enable_is_duplicated_expression_safety_check: bool = False
    """Setting for internal testing purposes only.
    
    When `True`, pyoframe checks that there are no bugs leading to duplicated terms in expressions.
    """

    integer_tolerance: float = 1e-8
    """Tolerance for checking if a floating point value is an integer.

    !!! info
        For convenience, Pyoframe returns the solution of integer and binary variables as integers not floating point values.
        To do so, Pyoframe must convert the solver-provided floating point values to integers. To avoid unexpected rounding errors,
        Pyoframe uses this tolerance to check that the floating point result is an integer as expected. Overly tight tolerances can trigger
        unexpected errors. Setting the tolerance to zero disables the check.
    """

    float_to_str_precision: int | None = 5
    """Number of decimal places to use when displaying mathematical expressions."""

    print_polars_config = pl.Config(
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        fmt_str_lengths=100,  # Set to a large value to avoid truncation (within reason)
        apply_on_context_enter=True,
    )
    """Polars `Config` used when printing Pyoframe objects."""

    @classmethod
    def reset_defaults(cls):
        """Resets all configuration options to their default values.

        Examples:
            >>> pf.Config.disable_unmatched_checks
            False
            >>> pf.Config.disable_unmatched_checks = True
            >>> pf.Config.disable_unmatched_checks
            True
            >>> pf.Config.reset_defaults()
            >>> pf.Config.disable_unmatched_checks
            False
        """
        for key, value in cls._defaults.items():
            setattr(cls, key, value)


class ConstraintSense(Enum):
    LE = "<="
    GE = ">="
    EQ = "="

    def _to_poi(self):
        """Converts the constraint sense to its pyoptinterface equivalent."""
        if self == ConstraintSense.LE:
            return poi.ConstraintSense.LessEqual
        elif self == ConstraintSense.EQ:
            return poi.ConstraintSense.Equal
        elif self == ConstraintSense.GE:
            return poi.ConstraintSense.GreaterEqual
        else:
            raise ValueError(f"Invalid constraint type: {self}")  # pragma: no cover


class ObjSense(Enum):
    MIN = "min"
    MAX = "max"

    def _to_poi(self):
        """Converts the objective sense to its pyoptinterface equivalent."""
        if self == ObjSense.MIN:
            return poi.ObjectiveSense.Minimize
        elif self == ObjSense.MAX:
            return poi.ObjectiveSense.Maximize
        else:
            raise ValueError(f"Invalid objective sense: {self}")  # pragma: no cover


class VType(Enum):
    """An [Enum](https://realpython.com/python-enum/) that can be used to specify the variable type.

    Examples:
        >>> m = pf.Model()
        >>> m.X = pf.Variable(vtype=VType.BINARY)

        The enum's string values can also be used directly although this is prone to typos:

        >>> m.Y = pf.Variable(vtype="binary")
    """

    CONTINUOUS = "continuous"
    """Variables that can be any real value."""
    BINARY = "binary"
    """Variables that must be either 0 or 1."""
    INTEGER = "integer"
    """Variables that must be integer values."""

    def _to_poi(self):
        """Convert the Variable type to its pyoptinterface equivalent."""
        if self == VType.CONTINUOUS:
            return poi.VariableDomain.Continuous
        elif self == VType.BINARY:
            return poi.VariableDomain.Binary
        elif self == VType.INTEGER:
            return poi.VariableDomain.Integer
        else:
            raise ValueError(f"Invalid variable type: {self}")  # pragma: no cover


class UnmatchedStrategy(Enum):
    """An enum to specify how to handle unmatched values in expressions."""

    UNSET = "not_set"
    DROP = "drop"
    KEEP = "keep"


# This is a hack to get the Literal type for VType
# See: https://stackoverflow.com/questions/67292470/type-hinting-enum-member-value-in-python
ObjSenseValue = Literal["min", "max"]
VTypeValue = Literal["continuous", "binary", "integer"]
for enum, type in [(ObjSense, ObjSenseValue), (VType, VTypeValue)]:
    assert set(typing.get_args(type)) == {vtype.value for vtype in enum}

SUPPORTED_SOLVER_TYPES = Literal["gurobi", "highs", "ipopt"]
assert set(typing.get_args(SUPPORTED_SOLVER_TYPES)) == {
    s.name for s in SUPPORTED_SOLVERS
}


class PyoframeError(Exception):
    """Class for all Pyoframe-specific errors, typically errors arising from improper arithmetic operations."""

    pass
