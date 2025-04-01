import pytest

import pyoframe as pf


@pytest.fixture(autouse=True)
def _setup_before_each_test(doctest_namespace):
    doctest_namespace["pf"] = pf
    pf.Config.reset_defaults()
    pf.Config.enable_is_duplicated_expression_safety_check = True


def pytest_markdown_docs_globals():
    return {"pf": pf}
