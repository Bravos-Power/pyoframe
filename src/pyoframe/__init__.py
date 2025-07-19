"""Pyoframe's public API accessible via `import pyoframe`.

Tip:
    `import pyoframe` will automatically patch Polars and Pandas
    to make `DataFrame.to_expr()` available.
"""

from pyoframe._constants import Config, VType
from pyoframe.core import Constraint, Expression, Set, Variable, sum, sum_by
from pyoframe.model import Model
from pyoframe._monkey_patch import patch_dataframe_libraries

try:
    from pyoframe._version import __version__, __version_tuple__  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    pass

patch_dataframe_libraries()

__all__ = [
    "sum",
    "sum_by",
    "Variable",
    "Model",
    "Set",
    "VType",
    "Config",
    "Constraint",
    "Expression",
]
