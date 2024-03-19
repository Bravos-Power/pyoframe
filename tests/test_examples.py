from pathlib import Path
from typing import List, Tuple

from tests.examples.diet_problem.model import main as main_diet
from tests.examples.diet_problem.model_gurobipy import main as main_diet_gurobipy
from tests.examples.facility_problem.model import main as main_facility
from tests.examples.facility_problem.model_gurobipy import (
    main as main_facility_gurobipy,
)
from tests.examples.cutting_stock_problem.model import main as main_cutting_stock


def test_diet_example():
    input_dir = Path("tests/examples/diet_problem/input_data/")
    output_dir = Path("tmp/diet_problem/")
    expected_output = Path("tests/examples/diet_problem/results/")
    delete_dir(output_dir)

    main_diet(input_dir, output_dir)
    main_diet_gurobipy(input_dir, output_dir)

    check_files_equal(expected_output / "diet.lp", output_dir / "diet.lp")
    check_files_equal(expected_output / "diet.sol", output_dir / "diet.sol")
    check_sol_equal(expected_output / "diet-gurobipy.sol", output_dir / "diet.sol")


def test_facility_example():
    input_dir = Path("tests/examples/facility_problem/input_data/")
    output_dir = Path("tmp/facility_problem/")
    expected_output = Path("tests/examples/facility_problem/results/")

    delete_dir(output_dir)

    main_facility(input_dir, output_dir)
    main_facility_gurobipy(input_dir, output_dir)

    # Check that two files results are equal
    check_files_equal(expected_output / "facility.lp", output_dir / "facility.lp")
    check_files_equal(expected_output / "facility.sol", output_dir / "facility.sol")
    check_sol_equal(
        expected_output / "facility-gurobipy.sol", output_dir / "facility.sol"
    )


def test_cutting_stock_example():
    input_dir = Path("tests/examples/cutting_stock_problem/input_data/")
    output_dir = Path("tmp/cutting_stock/")
    expected_output = Path("tests/examples/cutting_stock_problem/results/")

    main_cutting_stock(input_dir, output_dir)

    check_files_equal(
        expected_output / "cutting_stock.lp", output_dir / "cutting_stock.lp"
    )
    check_integer_solutions_only(output_dir / "cutting_stock.sol")


def check_files_equal(file_expected: Path, file_actual: Path):
    with open(file_expected) as f:
        expected = f.readlines()
    with open(file_actual) as f:
        actual = f.readlines()
    for e, a in zip(expected, actual):
        assert e == a, f"Expected {e} but got {a}"


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


def delete_dir(dir: Path):
    if dir.exists():
        for file in dir.iterdir():
            file.unlink()
