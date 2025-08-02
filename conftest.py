"""File related to setting up pytest."""

import os
from pathlib import Path

import polars as pl
import pytest

import pyoframe as pf


def _setup_before_each_test(doctest_namespace):
    doctest_namespace["pf"] = pf
    pf.Config.reset_defaults()
    pf.Config.enable_is_duplicated_expression_safety_check = True
    pf.Config.maintain_order = True


@pytest.fixture(autouse=True)
def _setup_fixture(doctest_namespace):
    _setup_before_each_test(doctest_namespace)


def pytest_collection_modifyitems(items):
    """Exclude certain paths from contributing to the test coverage.

    Specifically, the integration tests and documentation code snippets are not
    counted towards the coverage metrics because they're not rigorous enough.

    See:
        - https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_collection_modifyitems
        - https://stackoverflow.com/questions/60608511/pytest-cov-dont-count-coverage-for-the-directory-of-integration-tests
    """
    root_dir = Path(__file__).parent
    no_coverage_paths = (
        root_dir / "docs",
        root_dir / "tests" / "test_examples.py",
    )

    for item in items:
        test_path = Path(item.fspath)
        if any(test_path.is_relative_to(p) for p in no_coverage_paths):
            item.add_marker(pytest.mark.no_cover)


@pytest.fixture(scope="module")
def markdown_setup_fixture():
    cwd = os.getcwd()
    pl.Config.set_tbl_hide_dataframe_shape(True)
    yield
    os.chdir(cwd)
    pl.Config.set_tbl_hide_dataframe_shape(False)


try:
    from sybil import Sybil
    from sybil.parsers.markdown import (
        ClearNamespaceParser,
        PythonCodeBlockParser,
        SkipParser,
    )
    from sybil.parsers.rest import DocTestParser

    pytest_collect_file = Sybil(
        parsers=[
            PythonCodeBlockParser(),
            SkipParser(),
            ClearNamespaceParser(),
            DocTestParser(),
        ],
        patterns=["*.md"],
        setup=_setup_before_each_test,
        fixtures=["markdown_setup_fixture"],
    ).pytest()
except ImportError:
    # Sybil is not installed, so we won't collect markdown files.
    pass
