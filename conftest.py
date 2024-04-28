import pytest

import pyoframe as pf
from pyoframe.util import IdCounterMixin


@pytest.fixture(autouse=True)
def setup_before_each_test():
    IdCounterMixin._reset_counters()
    pf.Config.reset_defaults()
