"""
File containing shared constants used across the package.
"""

from enum import Enum
import typing
from typing import Literal, Optional


COEF_KEY = "__coeff"
VAR_KEY = "__variable_id"
CONSTRAINT_KEY = "__constraint_id"
SOLUTION_KEY = "__solution"
DUAL_KEY = "__dual"
CONST_TERM = 0

RESERVED_COL_KEYS = (COEF_KEY, VAR_KEY, CONSTRAINT_KEY, SOLUTION_KEY, DUAL_KEY)


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
    disable_unmatched_checks: bool = False
    printing_float_precision: Optional[int] = 5
    preserve_full_names: bool = False

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


class ObjSense(Enum):
    MIN = "minimize"
    MAX = "maximize"


class VType(Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"


class UnmatchedStrategy(Enum):
    UNSET = "not_set"
    DROP = "drop"
    KEEP = "keep"


# This is a hack to get the Literal type for VType
# See: https://stackoverflow.com/questions/67292470/type-hinting-enum-member-value-in-python
ObjSenseValue = Literal["minimize", "maximize"]
VTypeValue = Literal["continuous", "binary", "integer"]
for enum, type in [(ObjSense, ObjSenseValue), (VType, VTypeValue)]:
    assert set(typing.get_args(type)) == {vtype.value for vtype in enum}
