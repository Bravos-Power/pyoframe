import pytest

import pyoframe as pf

@pytest.fixture(params=["gurobi", "highs"])
def solver(request):
    return request.param

@pytest.fixture(autouse=True)
def _setup_before_each_test(solver, doctest_namespace):
    doctest_namespace["pf"] = pf
    pf.Config.reset_defaults()
    pf.Config.default_solver = solver
    pf.Config.enable_is_duplicated_expression_safety_check = True
