import pytest
import pyoframe as pf


@pytest.fixture(params=["gurobi", "highs"])
def solver(request):
    return request.param


@pytest.fixture(autouse=True)
def _apply_solver(solver):
    pf.Config.default_solver = solver
