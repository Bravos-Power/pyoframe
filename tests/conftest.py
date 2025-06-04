import pytest

import pyoframe as pf
from pyoframe.constants import SUPPORTED_SOLVERS


@pytest.fixture(params=SUPPORTED_SOLVERS, ids=lambda s: s.name)
def solver(request):
    return request.param


@pytest.fixture(autouse=True)
def _apply_solver(solver):
    pf.Config.default_solver = solver.name


@pytest.fixture(params=[True, False])
def use_var_names(request):
    return request.param
