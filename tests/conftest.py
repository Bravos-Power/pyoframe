import pytest

import pyoframe as pf
from pyoframe._constants import SUPPORTED_SOLVERS

_available_solvers = []
for s in SUPPORTED_SOLVERS:
    try:
        pf.Model(solver=s.name)
        _available_solvers.append(s)
    except RuntimeError:
        pass
if not _available_solvers:
    raise ValueError("No solvers installed. Cannot run tests.")


@pytest.fixture(autouse=True)
def _force_solver_selection():
    # Force each test to use the solver fixture if appropriateF
    pf.Config.default_solver = "raise"


@pytest.fixture(params=SUPPORTED_SOLVERS, ids=lambda s: s.name)
def solver(request):
    if request.param not in _available_solvers:
        return pytest.skip("Solver not installed.")
    return request.param


@pytest.fixture(params=[True, False])
def use_var_names(request):
    return request.param
