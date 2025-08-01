"""Tests related to converting Pyoframe objects to strings or writing them to files."""

import os
from tempfile import TemporaryDirectory

import gurobipy as gp
import polars as pl
import pytest

from pyoframe import Config, Model, Variable, sum
from tests.util import csvs_to_expr


def test_variables_to_string(solver):
    m = Model(solver=solver)
    m.x1 = Variable()
    m.x2 = Variable()
    m.x3 = Variable()
    m.x4 = Variable()
    expression = 5 * m.x1 + 3.4 * m.x2 - 2.1 * m.x3 + 1.1231237019273 * m.x4
    assert str(expression) == "5 x1 +3.4 x2 -2.1 x3 +1.12312 x4"


def test_variables_to_string_with_dimensions(solver):
    df = pl.DataFrame(
        {
            "x": [1, 2, 1, 2],
            "y": [1, 1, 2, 2],
        }
    )
    m = Model(solver=solver)
    m.v1 = Variable(df)
    m.v2 = Variable(df)
    m.v3 = Variable(df)
    m.v4 = Variable(df)

    expression_with_dimensions = (
        5 * m.v1 + 3.4 * m.v2 - 2.1 * m.v3 + 1.1231237019273 * m.v4
    )
    result = expression_with_dimensions._to_str(include_header=False)
    assert (
        result
        == """[1,1]: 5 v1[1,1] +3.4 v2[1,1] -2.1 v3[1,1] +1.12312 v4[1,1]
[2,1]: 5 v1[2,1] +3.4 v2[2,1] -2.1 v3[2,1] +1.12312 v4[2,1]
[1,2]: 5 v1[1,2] +3.4 v2[1,2] -2.1 v3[1,2] +1.12312 v4[1,2]
[2,2]: 5 v1[2,2] +3.4 v2[2,2] -2.1 v3[2,2] +1.12312 v4[2,2]"""
    )


def test_expression_with_const_to_str(solver):
    m = Model(solver=solver)
    m.x1 = Variable()
    expr = 5 + 2 * m.x1
    assert str(expr) == "2 x1 +5"


def test_constraint_to_str(solver):
    if not solver.supports_quadratics:
        pytest.skip("Solver does not support quadratic constraints.")
    m = Model(solver=solver)
    m.x1 = Variable()
    m.constraint = m.x1**2 <= 5
    assert (
        str(m.constraint)
        == """<Constraint sense='<=' size=1 dimensions={} terms=2>
constraint: x1 * x1 <= 5"""
    )

    # Now with dimensions
    m.x2 = Variable({"x": [1, 2, 3]})
    m.constraint_2 = m.x2 * m.x1 <= 5
    assert (
        str(m.constraint_2)
        == """<Constraint sense='<=' size=3 dimensions={'x': 3} terms=6>
constraint_2[1]: x2[1] * x1 <= 5
constraint_2[2]: x2[2] * x1 <= 5
constraint_2[3]: x2[3] * x1 <= 5"""
    )


def test_to_str():
    expr = csvs_to_expr(
        """
    day,water_drank
    1,2.00000000001
    2,3
    3,4
"""
    )
    Config.float_to_str_precision = None
    assert str(expr) == "[1]: 2.00000000001\n[2]: 3\n[3]: 4"
    Config.float_to_str_precision = 6
    assert str(expr) == "[1]: 2\n[2]: 3\n[3]: 4"
    # repr() is what is used when the object is printed in the console
    assert (
        repr(expr)
        == "<Expression size=3 dimensions={'day': 3} terms=3>\n[1]: 2\n[2]: 3\n[3]: 4"
    )
    Config.float_to_str_precision = None
    assert (
        repr(expr)
        == "<Expression size=3 dimensions={'day': 3} terms=3>\n[1]: 2.00000000001\n[2]: 3\n[3]: 4"
    )


def test_write_lp(use_var_names, solver):
    if not solver.supports_write:
        pytest.skip(f"{solver.name} does not support writing LP files.")
    with TemporaryDirectory() as tmpdir:
        m = Model(solver=solver, use_var_names=use_var_names)
        cities = pl.DataFrame(
            {
                "city": ["Toronto", "Montreal", "Vancouver"],
                "country": ["CAN", "CAN", "CAN"],
                "rent": [1000, 800, 1200],
                "capacity": [100, 200, 150],
            }
        )
        m.population = Variable(cities[["country", "city"]])
        m.minimize = sum(cities[["country", "city", "rent"]] * m.population)
        m.total_pop = sum(m.population) >= 310
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
    if not solver.supports_write:
        pytest.skip(f"{solver.name} does not support writing solution files.")
    with TemporaryDirectory() as tmpdir:
        m = Model(solver=solver, use_var_names=use_var_names)
        cities = pl.DataFrame(
            {
                "city": ["Toronto", "Montreal", "Vancouver"],
                "country": ["CAN", "CAN", "CAN"],
                "rent": [1000, 800, 1200],
                "capacity": [100, 200, 150],
            }
        )
        m.population = Variable(cities[["country", "city"]])
        m.minimize = sum(cities[["country", "city", "rent"]] * m.population)
        m.total_pop = sum(m.population) >= 310
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
