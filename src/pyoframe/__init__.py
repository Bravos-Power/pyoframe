"""Pyoframe's public API accessible via `import pyoframe`.

Note:
    `import pyoframe` will automatically patch Polars and Pandas
    to make `DataFrame.to_expr()` available.
"""

from pyoframe._constants import Config, ObjSense, PyoframeError, VType
from pyoframe._core import Constraint, Expression, Set, Variable, sum, sum_by
from pyoframe._model import Model
from pyoframe._monkey_patch import patch_dataframe_libraries
from pyoframe._objective import Objective

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
    "Set",
    "Config",
    "sum",
    "sum_by",
    "VType",
    "ObjSense",
    "PyoframeError",
]
