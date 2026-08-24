"""Tests relating to constraints."""

import re

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pyoframe as pf
from pyoframe._constants import VAR_KEY


def test_creation(default_solver):
    m = pf.Model(default_solver)
    df = pl.DataFrame({"x": [1, 2, 3], "val": [10, 20, 30]})

    m.X = pf.Variable(df["x"])

    with pytest.raises(
        pf.PyoframeError, match=re.escape("Did you mean to use '<=' or '>=' instead?")
    ):
        _ = m.X > df
    with pytest.raises(
        pf.PyoframeError, match=re.escape("Did you mean to use '<=' or '>=' instead?")
    ):
        _ = df > m.X
    with pytest.raises(
        pf.PyoframeError, match=re.escape("Did you mean to use '<=' or '>=' instead?")
    ):
        _ = m.X < df
    with pytest.raises(
        pf.PyoframeError, match=re.escape("Did you mean to use '<=' or '>=' instead?")
    ):
        _ = df < m.X

    m.c1 = m.X >= df
    m.c2 = m.X <= df
    m.c3 = df >= m.X
    m.c4 = df <= m.X
    m.c5 = m.X == df


@pytest.fixture
def update_base_model(solver):
    # setup (write it out it will help!)
    m = pf.Model(solver)
    df = pl.DataFrame({"i": [1, 2], "val": [12, 15]})
    x_coef = pl.DataFrame({"i": [1, 2], "coef": [0.5, 1.0]})
    y_coef = pl.DataFrame({"i": [1, 2], "coef": [2.0, 3.0]})
    m.X = pf.Variable(df["i"], lb=0, ub=10)
    m.Y = pf.Variable(df["i"], lb=0, ub=10)
    m.Dimensioned = x_coef * m.X + y_coef * m.Y <= df
    m.Dimensionless = (x_coef * m.X).sum() <= 12
    m.maximize = m.X.sum() + m.Y.sum()
    return m


def solve_and_check_base_model_solution(m, expected_X, expected_Y):
    m.optimize()
    for var, expected in zip([m.X, m.Y], [expected_X, expected_Y]):
        solution = var.solution.sort("i").get_column("solution").round(3).to_list()
        expected = [round(val, 3) for val in expected]
        assert solution == expected, (
            f"Expected {expected} but got {solution} for variable {var.name}"
        )


def test_update_base_model(update_base_model):
    m = update_base_model
    solve_and_check_base_model_solution(m, [10, 7], [3.5, 8 / 3])


def test_illegal_updates(solver, update_base_model):
    m = update_base_model
    m.optimize()

    if not solver.supports_updating_coefficients:
        with pytest.raises(pf.PyoframeError, match=re.escape("does not support")):
            m.Dimensioned.update(m.X + m.Y <= 12)
        return

    # test invalid updates
    with pytest.raises(NotImplementedError, match=re.escape("quadratic")):
        m.Dimensioned.update(m.X**2 <= 12)
    with pytest.raises(pf.PyoframeError, match=re.escape("different dimensions")):
        m.Dimensionless.update(m.X <= 12)
    with pytest.raises(pf.PyoframeError, match=re.escape("equality")):
        m.Dimensioned.update(m.X == 12)
    with pytest.raises(pf.PyoframeError, match=re.escape("already added to the model")):
        m.Dimensionless.update(m.Dimensionless)
    with pytest.raises(
        pf.PyoframeError,
        match=re.escape("contains labels that do not exist in the existing constraint"),
    ):
        m.Z = pf.Variable({"i": [1, 2, 3]})
        m.Dimensioned.update(m.Z <= 12)


def test_update_dimensionless_full(solver, update_base_model):
    if not solver.supports_updating_coefficients:
        pytest.skip(f"Solver '{solver.name}' does not support updating coefficients.")
    m = update_base_model
    m.optimize()
    # Now test valid updates
    # Full overwrite
    y_coef = pl.DataFrame({"i": [1, 2], "coef": [2.0, 3.0]})
    new_constr = (y_coef * m.Y).sum() <= 10
    m.Dimensionless.update(new_constr)
    solve_and_check_base_model_solution(m, [10, 10], [3.5, 1])
    assert_frame_equal(
        m.Dimensionless.lhs.data.sort(VAR_KEY), new_constr.lhs.data.sort(VAR_KEY)
    )


def test_update_dimensionless_full_flip(solver, update_base_model):
    if not solver.supports_updating_coefficients:
        pytest.skip(f"Solver '{solver.name}' does not support updating coefficients.")
    m = update_base_model
    m.optimize()
    y_coef = pl.DataFrame({"i": [1, 2], "coef": [2.0, 3.0]})
    new_constr = -(y_coef * m.Y).sum() >= -10
    print(new_constr.sense)
    m.Dimensionless.update(new_constr)
    solve_and_check_base_model_solution(m, [10, 10], [3.5, 1])
    assert_frame_equal(
        m.Dimensionless.lhs.data.sort(VAR_KEY), (-new_constr.lhs).data.sort(VAR_KEY)
    )
    assert m.Dimensionless.sense._flip() == new_constr.sense


def test_update_dimensionless_partial(solver, update_base_model):
    if not solver.supports_updating_coefficients:
        pytest.skip(f"Solver '{solver.name}' does not support updating coefficients.")
    m = update_base_model
    m.optimize()

    x_coef = pl.DataFrame({"i": [1, 2], "coef": [1.0, 0.5]})
    new_constr = (x_coef * m.X).sum() <= 12
    m.Dimensionless.update(new_constr)
    solve_and_check_base_model_solution(m, [7, 10], [(12 - 3.5) / 2, 5 / 3])
    assert_frame_equal(
        m.Dimensionless.lhs.data.sort(VAR_KEY), new_constr.lhs.data.sort(VAR_KEY)
    )

    new_constr = (x_coef * m.X).sum() <= 14
    m.Dimensionless.update(new_constr)
    solve_and_check_base_model_solution(m, [9, 10], [(12 - 4.5) / 2, 5 / 3])
    assert_frame_equal(
        m.Dimensionless.lhs.data.sort(VAR_KEY), new_constr.lhs.data.sort(VAR_KEY)
    )


def test_update_dimensioned(solver, update_base_model):
    if not solver.supports_updating_coefficients:
        pytest.skip(f"Solver '{solver.name}' does not support updating coefficients.")
    m = update_base_model
    m.optimize()

    updated_constr = m.X.filter(i=1) <= 8
    m.Dimensioned.update(updated_constr)
    solve_and_check_base_model_solution(m, [8, 8], [10, 7 / 3])

    # expected constr after updates
    y_coef = pl.DataFrame({"i": [1, 2], "coef": [0, 3]})
    rhs = pl.DataFrame({"i": [1, 2], "val": [8, 15]})
    expected_constr = m.X + y_coef * m.Y <= rhs
    assert_frame_equal(
        m.Dimensioned.lhs.data.sort("i", VAR_KEY),
        expected_constr.lhs.data.sort("i", VAR_KEY),
    )
