"""
File containing shared constants used across the package.
"""

import importlib.metadata
import typing
from enum import Enum
from typing import Literal, Optional

import polars as pl
import pyoptinterface as poi
from packaging import version

# We want to try and support multiple major versions of polars
POLARS_VERSION = version.parse(importlib.metadata.version("polars"))

COEF_KEY = "__coeff"
VAR_KEY = "__variable_id"
QUAD_VAR_KEY = "__quadratic_variable_id"
CONSTRAINT_KEY = "__constraint_id"
SOLUTION_KEY = "solution"
DUAL_KEY = "dual"
SUPPORTED_SOLVER_TYPES = Literal["gurobi", "highs"]
KEY_TYPE = pl.UInt32

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
            if not k.startswith("_") and type(v) != classmethod
        }


class Config(metaclass=_ConfigMeta):
    """
    Configuration options that apply to the entire library.
    """

    default_solver: Optional[SUPPORTED_SOLVER_TYPES] = None
    disable_unmatched_checks: bool = False
    float_to_str_precision: Optional[int] = 5
    print_uses_variable_names: bool = True
    print_max_line_length: int = 80
    print_max_lines: int = 15
    # Number of elements to show when printing a set to the console (additional elements are replaced with ...)
    print_max_set_elements: int = 50
    enable_is_duplicated_expression_safety_check: bool = False

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


class PyoframeError(Exception):
    pass
