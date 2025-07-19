"""File containing shared constants used across the package."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

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
class Solver:
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
    Solver("gurobi"),
    Solver("highs", supports_quadratics=False, supports_duals=False),
    Solver(
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

    default_solver: SUPPORTED_SOLVER_TYPES | Solver | Literal["raise", "auto"] = "auto"
    """
    The solver to use when [Model][pyoframe.Model] is instantiated without specifying a solver.
    If `auto`, Pyoframe will choose the first solver in [SUPPORTED_SOLVERS][pyoframe.constants.SUPPORTED_SOLVERS] that doesn't produce an error.
    If `raise`, an exception will be raised when [Model][pyoframe.Model] is instantiated without specifying a solver.

    We recommend that users specify their solver when instantiating [Model][pyoframe.Model] rather than relying on this option.
    """

    disable_unmatched_checks: bool = False
    """
    Improve performance by skipping unmatched checks (not recommended).

    When `True`, unmatched checks are disabled which effectively means that all expressions
    are treated as if they contained [`.keep_unmatched()`][pyoframe.Expression.keep_unmatched]
    (unless [`.drop_unmatched()`][pyoframe.Expression.drop_unmatched] was applied).

    !!! warning
        This might improve performance, but it will suppress the "unmatched" errors that alert developers to unexpected
        behaviors (see [here](/pyoframe/learn/getting-started/special-functions#drop_unmatched-and-keep_unmatched)).
        Only consider enabling after you have thoroughly tested your code.

    Examples:
        >>> import polars as pl
        >>> population = pl.DataFrame({"city": ["Toronto", "Vancouver", "Montreal"], "pop": [2_731_571, 631_486, 1_704_694]}).to_expr()
        >>> population_influx = pl.DataFrame({"city": ["Toronto", "Vancouver", "Montreal"],"influx": [100_000, 50_000, None],}).to_expr()
        
        Normally, an error warns users that the two expressions have conflicting indices:
        >>> population + population_influx
        Traceback (most recent call last):
        ...
        pyoframe.constants.PyoframeError: Failed to add expressions:
        <Expression size=3 dimensions={'city': 3} terms=3> + <Expression size=2 dimensions={'city': 2} terms=2>
        Due to error:
        Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()
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
        <Expression size=3 dimensions={'city': 3} terms=3>
        [Toronto]: 2831571
        [Vancouver]: 681486
        [Montreal]: 1704694
    """

    print_max_line_length: int = 80
    """
    Maximum number of characters to print in a single line.

    Examples:
        >>> pf.Config.print_max_line_length = 20
        >>> m = pf.Model()
        >>> m.vars = pf.Variable({"x": range(1000)})
        >>> pf.sum(m.vars)
        <Expression size=1 dimensions={} terms=1000>
        vars[0] + vars[1] + …

    """

    print_max_lines: int = 15
    """
    Maximum number of lines to print.

    Examples:
        >>> pf.Config.print_max_lines = 3
        >>> import pandas as pd
        >>> expr = pd.DataFrame({"day_of_year": list(range(365)), "value": list(range(365))}).to_expr()
        >>> expr
        <Expression size=365 dimensions={'day_of_year': 365} terms=365>
        [0]: 0
        [1]: 1
        [2]: 2
         ⋮
    """

    print_max_set_elements: int = 50
    """
    Maximum number of elements in a set to print.
    
    Examples:
        >>> pf.Config.print_max_set_elements = 5
        >>> pf.Set(x=range(1000))
        <Set size=1000 dimensions={'x': 1000}>
        [0, 1, 2, 3, 4, …]
    """

    enable_is_duplicated_expression_safety_check: bool = False
    """
    Setting for internal testing purposes only.
    
    When `True`, pyoframe checks that there are no bugs leading to duplicated terms in expressions.
    """

    integer_tolerance: float = 1e-8
    """
    Tolerance for checking if a floating point value is an integer.

    !!! info
        For convenience, Pyoframe returns the solution of integer and binary variables as integers not floating point values.
        To do so, Pyoframe must convert the solver-provided floating point values to integers. To avoid unexpected rounding errors,
        Pyoframe uses this tolerance to check that the floating point result is an integer as expected. Overly tight tolerances can trigger
        unexpected errors. Setting the tolerance to zero disables the check.
    """

    float_to_str_precision: Optional[int] = 5
    """Number of decimal places to use when displaying mathematical expressions."""

    print_uses_variable_names: bool = True
    """
    Improve performance by not tracking the link between variable IDs and variable names.

    If set to `False`, printed expression will use variable IDs instead of variable names
    which might make debugging difficult.

    !!! warning
        This setting must be changed before instantiating a [Model][pyoframe.Model].
    
    Examples:
        >>> pf.Config.print_uses_variable_names = False
        >>> m = pf.Model()
        >>> m.my_var = pf.Variable()
        >>> 2 * m.my_var
        <Expression size=1 dimensions={} terms=1>
        2 x1
    """

    @classmethod
    def reset_defaults(cls):
        """Resets all configuration options to their default values.

        Examples:
            >>> pf.Config.print_uses_variable_names
            True
            >>> pf.Config.print_uses_variable_names = False
            >>> pf.Config.print_uses_variable_names
            False
            >>> pf.Config.reset_defaults()
            >>> pf.Config.print_uses_variable_names
            True
        """
        for key, value in cls._defaults.items():
            setattr(cls, key, value)


class ConstraintSense(Enum):
    LE = "<="
    GE = ">="
    EQ = "="

    def to_poi(self):
        """Convert the constraint sense to its pyoptinterface equivalent."""
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

    def to_poi(self):
        """Convert the objective sense to its pyoptinterface equivalent."""
        if self == ObjSense.MIN:
            return poi.ObjectiveSense.Minimize
        elif self == ObjSense.MAX:
            return poi.ObjectiveSense.Maximize
        else:
            raise ValueError(f"Invalid objective sense: {self}")  # pragma: no cover


class VType(Enum):
    """An enum to specify the variable type (continuous, binary, or integer)."""

    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"

    def to_poi(self):
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
    """Class for all Pyoframe-specific errors."""

    pass
