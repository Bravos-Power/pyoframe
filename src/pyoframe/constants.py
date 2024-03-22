"""
File containing shared constants used across the package.
"""

from enum import Enum
import typing
from typing import Literal


COEF_KEY = "__coeff"
VAR_KEY = "__variable_id"
CONST_TERM = 0

RESERVED_COL_KEYS = (COEF_KEY, VAR_KEY)


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


class MissingStrategy(Enum):
    ERROR = "error"
    DROP = "drop"
    FILL = "fill"


# This is a hack to get the Literal type for VType
# See: https://stackoverflow.com/questions/67292470/type-hinting-enum-member-value-in-python
ObjSenseValue = Literal["minimize", "maximize"]
VTypeValue = Literal["continuous", "binary", "integer"]
for enum, type in [(ObjSense, ObjSenseValue), (VType, VTypeValue)]:
    assert set(typing.get_args(type)) == {vtype.value for vtype in enum}
