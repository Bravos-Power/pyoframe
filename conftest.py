import pytest

import pyoframe as pf


@pytest.fixture(autouse=True)
def _setup_before_each_test():
    pf.Config.reset_defaults()
    pf.Config.enable_is_duplicated_expression_safety_check = True
