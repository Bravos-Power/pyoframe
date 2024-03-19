from io import TextIOWrapper
from math import exp
from pathlib import Path
from tabnanny import check
from typing import List, Tuple

from tests.examples.diet_problem.model import main as main_diet
from tests.examples.diet_problem.model_gurobipy import main as main_diet_gurobipy
from tests.examples.facility_problem.model import main as main_facility
from tests.examples.facility_problem.model_gurobipy import (
    main as main_facility_gurobipy,
)
from tests.examples.cutting_stock_problem.model import main as main_cutting_stock


def test_diet_example():
    working_dir = Path("tests/examples/diet_problem/")
    result_dir = working_dir / "results"
    # Delete previous results
    if result_dir.exists():
        for file in result_dir.iterdir():
            file.unlink()

    main_diet_gurobipy(working_dir)
    main_diet(working_dir)

    check_sol_equal(result_dir / "diet-gurobipy.sol", result_dir / "diet.sol")


def test_facility_example():
    working_dir = Path("tests/examples/facility_problem/")
    result_dir = working_dir / "results"
    # Delete previous results
    if result_dir.exists():
        for file in result_dir.iterdir():
            file.unlink()

    main_facility_gurobipy(working_dir)
    main_facility(working_dir)

    # Check that two files results are equal
    check_sol_equal(result_dir / "facility-gurobipy.sol", result_dir / "facility.sol")


def test_cutting_stock_example():
    working_dir = Path("tests/examples/cutting_stock_problem/")
    result_dir = working_dir / "results"
    # Delete previous results
    if result_dir.exists():
        for file in result_dir.iterdir():
            file.unlink()

    main_cutting_stock(working_dir)

    check_integer_solutions_only(result_dir / "cutting_stock.sol")


def check_integer_solutions_only(sol_file):
    sol = parse_gurobi_sol(sol_file)
    for name, value in sol:
        assert value.is_integer(), f"Variable {name} has non-integer value {value}"


def check_sol_equal(expected_sol_file, actual_sol_file):
    # Remove comments and empty lines
    expected_result = parse_gurobi_sol(expected_sol_file)
    actual_result = parse_gurobi_sol(actual_sol_file)

    tol = 1e-8
    for (expected_name, expected_value), (actual_name, actual_value) in zip(
        expected_result, actual_result
    ):
        assert expected_name == actual_name
        assert expected_value - tol <= actual_value <= expected_value + tol


def parse_gurobi_sol(sol_file_path) -> List[Tuple[str, float]]:
    with open(sol_file_path) as f:
        sol = f.readlines()
    sol = [line.strip() for line in sol]
    sol = [line for line in sol if not (line.startswith("#") or line == "")]
    sol = sorted(sol)
    sol = [line.partition(" ") for line in sol]
    sol = [(name, float(value)) for name, _, value in sol]
    return sol
