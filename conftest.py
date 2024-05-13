import pytest

import pyoframe as pf
from pyoframe.model_element import ModelElementWithId


@pytest.fixture(autouse=True)
def _setup_before_each_test():
    ModelElementWithId.reset_counters()
    pf.Config.reset_defaults()
