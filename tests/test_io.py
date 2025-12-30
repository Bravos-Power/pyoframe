"""Tests related to converting Pyoframe objects to strings or writing them to files."""

import os
import re
from tempfile import TemporaryDirectory

import gurobipy as gp
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from pyoframe import Model, Variable
from pyoframe._constants import _Solver


def test_variables_to_string(solver):
    m = Model(solver)
    m.x1 = Variable()
    m.x2 = Variable()
    m.x3 = Variable()
    m.x4 = Variable()
    expression = 5 * m.x1 + 3.4 * m.x2 - 2.1 * m.x3 + 1.1231237019273 * m.x4
    assert expression.to_str() == "5 x1 +3.4 x2 -2.1 x3 +1.12312 x4"


def test_variables_to_string_with_dimensions(solver):
    df = pl.DataFrame(
        {
            "x": [1, 2, 1, 2],
            "y": [1, 1, 2, 2],
        }
    )
    m = Model(solver)
    m.v1 = Variable(df)
    m.v2 = Variable(df)
    m.v3 = Variable(df)
    m.v4 = Variable(df)

    expression_with_dimensions = (
        5 * m.v1 + 3.4 * m.v2 - 2.1 * m.v3 + 1.1231237019273 * m.v4
    )
    assert_frame_equal(
        expression_with_dimensions.to_str(return_df=True),
        pl.DataFrame(
            [
                [1, 1, "5 v1[1,1] +3.4 v2[1,1] -2.1 v3[1,1] +1.12312 v4[1,1]"],
                [2, 1, "5 v1[2,1] +3.4 v2[2,1] -2.1 v3[2,1] +1.12312 v4[2,1]"],
                [1, 2, "5 v1[1,2] +3.4 v2[1,2] -2.1 v3[1,2] +1.12312 v4[1,2]"],
                [2, 2, "5 v1[2,2] +3.4 v2[2,2] -2.1 v3[2,2] +1.12312 v4[2,2]"],
            ],
            schema=["x", "y", "expression"],
            orient="row",
        ),
    )


def test_expression_with_const_to_str(solver):
    m = Model(solver)
    m.x1 = Variable()
    expr = 5 + 2 * m.x1
    assert str(expr) == "2 x1 +5"


def test_constraint_to_str(solver: _Solver):
    if not solver.supports_quadratic_constraints:
        pytest.skip("Solver does not support quadratic constraints.")
    m = Model(solver)
    m.x1 = Variable()
    m.constraint = m.x1**2 <= 5
    assert (
        repr(m.constraint)
        == """<Constraint 'constraint' (quadratic) terms=2>
x1 * x1 <= 5"""
    )

    # Now with dimensions
    m.x2 = Variable({"x": [1, 2, 3]})
    m.constraint_2 = m.x2 * m.x1 <= 5
    assert_frame_equal(
        m.constraint_2.to_str(return_df=True),
        pl.DataFrame(
            [
                [1, "x2[1] * x1 <= 5"],
                [2, "x2[2] * x1 <= 5"],
                [3, "x2[3] * x1 <= 5"],
            ],
            schema=["x", "constraint"],
            orient="row",
        ),
    )


def test_write_lp(use_var_names, solver: _Solver):
    m = Model(solver=solver, solver_uses_variable_names=use_var_names)

    if not solver.supports_write:
        with pytest.raises(
            NotImplementedError,
            match=re.escape(f"{solver.name} does not support .write()"),
        ):
            m.write("test.lp")
        return

    if not use_var_names and solver.accelerate_with_repeat_names:
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"{solver.name} requires solver_uses_variable_names=True to use .write()"
            ),
        ):
            m.write("test.lp")
        return

    with TemporaryDirectory() as tmpdir:
        cities = pl.DataFrame(
            {
                "city": ["Toronto", "Montreal", "Vancouver"],
                "country": ["CAN", "CAN", "CAN"],
                "rent": [1000, 800, 1200],
                "capacity": [100, 200, 150],
            }
        )
        m.population = Variable(cities[["country", "city"]])
        m.minimize = (cities[["country", "city", "rent"]] * m.population).sum()
        m.total_pop = m.population.sum() >= 310
        m.capacity_constraint = m.population <= cities[["country", "city", "capacity"]]

        file_path = os.path.join(tmpdir, "test.lp")
        m.write(file_path)
        m.optimize()
        obj_value = m.objective.value
        gp_model = gp.read(file_path)
        gp_model.optimize()
        assert gp_model.ObjVal == obj_value

        with open(file_path) as f:
            if use_var_names:
                assert "population[CAN,Toronto]" in f.read()
            else:
                assert "population[CAN,Toronto]" not in f.read()


def test_write_sol(use_var_names, solver):
    if not (
        solver.supports_write
        and (use_var_names or not solver.accelerate_with_repeat_names)
    ):
        pytest.skip(f"{solver.name} does not support writing solution files.")
    with TemporaryDirectory() as tmpdir:
        m = Model(solver, solver_uses_variable_names=use_var_names)
        cities = pl.DataFrame(
            {
                "city": ["Toronto", "Montreal", "Vancouver"],
                "country": ["CAN", "CAN", "CAN"],
                "rent": [1000, 800, 1200],
                "capacity": [100, 200, 150],
            }
        )
        m.population = Variable(cities[["country", "city"]])
        m.minimize = (cities[["country", "city", "rent"]] * m.population).sum()
        m.total_pop = m.population.sum() >= 310
        m.capacity_constraint = m.population <= cities[["country", "city", "capacity"]]

        file_path = os.path.join(tmpdir, "test.sol")
        m.optimize()
        m.write(file_path)

        with open(file_path) as f:
            if use_var_names:
                assert "population[CAN,Toronto]" in f.read()
            else:
                assert "population[CAN,Toronto]" not in f.read()


if __name__ == "__main__":
    pytest.main([__file__])
