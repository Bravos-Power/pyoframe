import pytest

import pyoframe as pf


def test_set_objective():
    # There are many ways to set the objective in a model.
    # A)
    m = pf.Model()
    m.minimize = 2

    # B)
    m = pf.Model()
    m.maximize = 2

    # C)
    m = pf.Model(sense="min")
    m.objective = 3

    # We can also add or subtract from the objective
    m.minimize += 3
    m.minimize -= 2
    assert m.minimize.evaluate() == 4

    # Other ways should throw an error
    with pytest.raises(
        ValueError, match="Did you use .objective instead of .minimize or .maximize ?"
    ):
        m = pf.Model()
        m.objective = 3

    with pytest.raises(
        ValueError, match="Can't set .minimize in a maximization problem."
    ):
        m = pf.Model(sense="max")
        m.minimize = 3

    with pytest.raises(
        ValueError, match="Can't set .maximize in a minimization problem."
    ):
        m = pf.Model(sense="min")
        m.maximize = 3

    with pytest.raises(
        ValueError, match="Can't get .minimize in a maximization problem."
    ):
        m = pf.Model()
        m.maximize = 3
        m.minimize

    with pytest.raises(
        ValueError, match="Can't get .maximize in a minimization problem."
    ):
        m = pf.Model()
        m.minimize = 3
        m.maximize
