"""Pyoframe's public API accessible via `import pyoframe as pf`."""

from pyoframe._constants import (
    Config,
    ObjSense,
    PyoframeError,
    VType,
    _Config,  # noqa: F401 Should be kept here to allow cross referencing in the documentation
)
from pyoframe._core import Constraint, Expression, Set, Variable, sum, sum_by
from pyoframe._model import Model
from pyoframe._monkey_patch import patch_dataframe_libraries
from pyoframe._objective import Objective
from pyoframe._param import Param

try:
    from pyoframe._version import __version__, __version_tuple__  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pass

patch_dataframe_libraries()

__all__ = [
    "Model",
    "Variable",
    "Expression",
    "Constraint",
    "Objective",
    "Param",
    "Set",
    "Config",
    "sum",
    "sum_by",
    "VType",
    "ObjSense",
    "PyoframeError",
]
