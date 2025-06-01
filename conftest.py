from pathlib import Path

import pytest

import pyoframe as pf


@pytest.fixture(autouse=True)
def _setup_before_each_test(doctest_namespace):
    doctest_namespace["pf"] = pf
    pf.Config.reset_defaults()
    pf.Config.enable_is_duplicated_expression_safety_check = True


def pytest_collection_modifyitems(items):
    """
    Exclude certain paths from contributing to the test coverage.
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


def pytest_markdown_docs_globals():
    return {"pf": pf}
