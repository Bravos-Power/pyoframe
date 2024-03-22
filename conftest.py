import pytest

from pyoframe.variables import Variable


@pytest.fixture(autouse=True)
def setup_before_each_test():
    Variable.reset_var_count()
