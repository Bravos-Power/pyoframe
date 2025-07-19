import pytest

import pyoframe as pf
from pyoframe.constants import SUPPORTED_SOLVERS

_installed_solvers = []
for s in SUPPORTED_SOLVERS:
    try:
        pf.Model(solver=s.name)
        _installed_solvers.append(s)
    except RuntimeError:
        pass
if not _installed_solvers:
    raise ValueError("No solvers detected. Make sure a solver is installed.")


@pytest.fixture(params=SUPPORTED_SOLVERS, ids=lambda s: s.name)
def solver(request):
    if request.param not in _installed_solvers:
        return pytest.skip("Solver not installed.")
    return request.param


@pytest.fixture(autouse=True)
def _force_solver_selection():
    # Force each test to use the solver fixture if appropriateF
    pf.Config.default_solver = "raise"


@pytest.fixture(params=[True, False])
def use_var_names(request):
    return request.param
