"""
File containing shared constants used across the package.
"""

from enum import Enum
import typing
from typing import Literal, Optional


COEF_KEY = "__coeff"
VAR_KEY = "__variable_id"
CONST_TERM = 0

RESERVED_COL_KEYS = (COEF_KEY, VAR_KEY)

class Config:
    disable_unmatched_checks = False
    printing_float_precision: Optional[int] = 6


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
