import pytest

import pyoframe as pf


@pytest.fixture(autouse=True)
def _setup_before_each_test(doctest_namespace):
    doctest_namespace["pf"] = pf
    pf.Config.reset_defaults()
    pf.Config.enable_is_duplicated_expression_safety_check = True


@pytest.fixture(params=[True, False])
def use_var_names(request, solver):
    if solver == "highs" and request.param:
        pytest.skip(
            "Highs does not support variable names. See https://github.com/Bravos-Power/pyoframe/issues/102#issuecomment-2727521430"
        )
    return request.param
