import pytest

import pyoframe as pf
from pyoframe.model_element import CountableModelElement


@pytest.fixture(autouse=True)
def _setup_before_each_test():
    CountableModelElement.reset_counters()
    pf.Config.reset_defaults()
