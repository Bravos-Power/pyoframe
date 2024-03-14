import pytest

from convop.variables import Variable


@pytest.fixture(autouse=True)
def setup_before_each_test():
    Variable._reset_count()
