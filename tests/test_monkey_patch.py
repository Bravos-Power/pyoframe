"""Tests for the monkey patching of pandas and polars."""

import importlib
import sys

from pyoframe._monkey_patch import _PandasFinderInterceptor


def test_pandas_patching_works():
    # Test import AFTER pyoframe
    reset_imports(["pandas", "pyoframe", "polars"])

    import pandas as pd  # noqa: I001
    import polars as pl
    import pyoframe  # noqa: F401, F811

    assert hasattr(pl.DataFrame, "__pyoframe_patched__")
    assert hasattr(pd.DataFrame, "__pyoframe_patched__")

    # Test import BEFORE pyoframe
    reset_imports(["pandas", "pyoframe", "polars"])

    import pandas as pd
    import polars as pl

    assert not hasattr(pl.DataFrame, "__pyoframe_patched__")
    assert not hasattr(pd.DataFrame, "__pyoframe_patched__")

    import pyoframe  # noqa: F401, F811

    assert hasattr(pl.DataFrame, "__pyoframe_patched__")
    assert hasattr(pd.DataFrame, "__pyoframe_patched__")


def test_pyoframe_does_not_import_pandas():
    reset_imports(["pandas", "pyoframe", "polars"])

    assert "pandas" not in sys.modules

    import pyoframe  # noqa: F401, F811

    assert "pandas" not in sys.modules

    import pandas  # noqa: F401, F811

    assert "pandas" in sys.modules


def test_no_double_patch():
    reset_imports(["pandas", "pyoframe", "polars"])

    base_length = len(sys.meta_path)

    from pyoframe._monkey_patch import patch_dataframe_libraries

    assert len(sys.meta_path) == base_length + 1

    patch_dataframe_libraries()
    patch_dataframe_libraries()
    patch_dataframe_libraries()

    assert len(sys.meta_path) == base_length + 1


def reset_imports(modules: list[str]):
    for name in list(sys.modules.keys()):
        if any(name == mod or name.startswith(mod + ".") for mod in modules):
            del sys.modules[name]
    importlib.invalidate_caches()
    for finder in sys.meta_path:
        if isinstance(finder, _PandasFinderInterceptor):
            sys.meta_path.remove(finder)
