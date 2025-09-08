import pytest

import pyoframe as pf
from pyoframe._constants import SUPPORTED_SOLVERS

_installed_solvers = []
for s in SUPPORTED_SOLVERS:
    try:
        pf.Model(s.name)
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


@pytest.fixture
def default_solver():
    """Use when the test doesn't need to be repeated for every solver.

    TODO increase the use of default_solver over solver when appropriate to reduce test times.
    """
    return _installed_solvers[0]


@pytest.fixture(autouse=True)
def _force_solver_selection():
    # Force each test to use the solver fixture if appropriateF
    pf.Config.default_solver = "raise"


@pytest.fixture(params=[True, False], ids=["var_names", ""])
def use_var_names(request):
    return request.param
