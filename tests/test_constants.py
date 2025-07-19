"""Module that runs the doctests in pyoframe.Config which otherwise would go undetected."""

import doctest
import os
import tempfile

import pytest

import pyoframe as pf
from tests.util import get_attr_docs


@pytest.fixture(params=get_attr_docs(pf.Config).items(), ids=lambda x: x[0])
def param_doctest(request):
    return request.param[1]


def test_Config_doctests(param_doctest, doctest_namespace):
    with tempfile.TemporaryDirectory() as tmp:
        filepath = os.path.join(tmp, "config_doctest.py")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(param_doctest)
        pf.Config.default_solver = "auto"
        failed_count, _ = doctest.testfile(
            filepath,
            module_relative=False,
            globs=doctest_namespace,
            encoding="utf-8",
            optionflags=doctest.ELLIPSIS,
            raise_on_error=False,
        )
    assert failed_count == 0
