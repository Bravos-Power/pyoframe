"""
File containing shared constants used across the package.
"""

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
    """
    Configuration options that apply to the entire library.
    """

    default_solver: SUPPORTED_SOLVER_TYPES | Solver | Literal["raise", "auto"] = "auto"
    """
    The solver to use when [Model][pyoframe.Model] is instantiated without specifying a solver.
    If `auto`, Pyoframe will choose the first solver in [SUPPORTED_SOLVERS][pyoframe.constants.SUPPORTED_SOLVERS] that doesn't produce an error.
    If `raise`, an exception will be raised when [Model][pyoframe.Model] is instantiated without specifying a solver.

    We recommend that users specify their solver when instantiating [Model][pyoframe.Model] rather than relying on this option.
    """

    disable_unmatched_checks: bool = False
    float_to_str_precision: Optional[int] = 5
    print_uses_variable_names: bool = True
    print_max_line_length: int = 80
    print_max_lines: int = 15
    print_max_set_elements: int = 50
    "Number of elements to show when printing a set to the console (additional elements are replaced with ...)"

    enable_is_duplicated_expression_safety_check: bool = False

    integer_tolerance: float = 1e-8
    """
    For convenience, Pyoframe returns the solution of integer and binary variables as integers not floating point values.
    To do so, Pyoframe must convert the solver-provided floating point values to integers. To avoid unexpected rounding errors,
    Pyoframe uses this tolerance to check that the floating point result is an integer as expected. Overly tight tolerances can trigger
    unexpected errors. Setting the tolerance to zero disables the check.
    """

    maintain_order: bool = True
    """
    If True, performance and memory usage may worsen, but the order of terms within expressions will not change across runs.
    """

    @classmethod
    def reset_defaults(cls):
        """
        Resets all configuration options to their default values.
        """
        for key, value in cls._defaults.items():
            setattr(cls, key, value)


class ConstraintSense(Enum):
    LE = "<="
    GE = ">="
    EQ = "="

    def to_poi(self):
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
        if self == ObjSense.MIN:
            return poi.ObjectiveSense.Minimize
        elif self == ObjSense.MAX:
            return poi.ObjectiveSense.Maximize
        else:
            raise ValueError(f"Invalid objective sense: {self}")  # pragma: no cover


class VType(Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"

    def to_poi(self):
        if self == VType.CONTINUOUS:
            return poi.VariableDomain.Continuous
        elif self == VType.BINARY:
            return poi.VariableDomain.Binary
        elif self == VType.INTEGER:
            return poi.VariableDomain.Integer
        else:
            raise ValueError(f"Invalid variable type: {self}")  # pragma: no cover


class UnmatchedStrategy(Enum):
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
    pass
