import pytest

import pyoframe as pf
from pyoframe.constants import SUPPORTED_SOLVERS


@pytest.fixture(params=SUPPORTED_SOLVERS)
def solver(request):
    return request.param


@pytest.fixture(autouse=True)
def _apply_solver(solver):
    pf.Config.default_solver = solver


@pytest.fixture(params=[True, False])
def use_var_names(request, solver):
    if solver == "highs" and request.param:
        pytest.skip(
            "Highs does not support variable names. See https://github.com/Bravos-Power/pyoframe/issues/102#issuecomment-2727521430"
        )
    return request.param
