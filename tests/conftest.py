import pytest

import pyoframe as pf
from pyoframe.constants import SUPPORTED_SOLVERS

_installed_solvers = []

for solver in SUPPORTED_SOLVERS:
    try:
        pf.Model(solver=solver)
        _installed_solvers.append(solver)
    except RuntimeError:
        pass

if _installed_solvers == []:
    raise ValueError("No solvers detected.")


@pytest.fixture(params=_installed_solvers, ids=lambda s: s.name)
def solver(request):
    return request.param


@pytest.fixture(autouse=True)
def _apply_solver(solver):
    pf.Config.default_solver = solver.name


@pytest.fixture(params=[True, False])
def use_var_names(request):
    return request.param
