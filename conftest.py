import pytest

import pyoframe as pf


@pytest.fixture(autouse=True)
def setup_before_each_test():
    pf.Variable._reset_counter()
    pf.Constraint._reset_counter()
    pf.Config.reset_defaults()
