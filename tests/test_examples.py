from dataclasses import dataclass
import importlib
import shutil
import pytest
from pathlib import Path
from typing import List, Tuple

import pyoframe as pf


@dataclass
class Example:
    folder_name: str
    integer_results_only: bool = False
    many_valid_solutions: bool = False


EXAMPLES = [
    Example("diet_problem"),
    Example("facility_problem"),
    Example(
        "cutting_stock_problem", integer_results_only=True, many_valid_solutions=True
    ),
]


@pytest.mark.parametrize(
    "example",
    EXAMPLES,
)
def test_examples(example: Example):
    example_dir = Path("tests/examples") / example.folder_name
    input_dir = example_dir / "input_data"
    expected_output_dir = example_dir / "results"
    working_dir = Path("tmp") / example.folder_name
    symbolic_output_dir = working_dir / "results"
    dense_output_dir = working_dir / "results_dense"

    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True)

    # Dynamically import the main function of the example
    main_module = importlib.import_module(f"tests.examples.{example.folder_name}.model")

    symbolic_solution_file = symbolic_output_dir / "pyoframe-problem.sol"
    dense_solution_file = dense_output_dir / "pyoframe-problem.sol"

    dense_result = main_module.main(
        input_dir, directory=dense_output_dir, solution_file=dense_solution_file
    )
    symbolic_result = main_module.main(
        input_dir,
        directory=symbolic_output_dir,
        solution_file=symbolic_solution_file,
        use_var_names=True,
    )
    check_results_equal(dense_result, symbolic_result)

    gurobi_module = None
    try:
        gurobi_module = importlib.import_module(
            f"tests.examples.{example.folder_name}.model_gurobipy"
        )
    except ImportError:
        pass

    if gurobi_module is not None:
        gurobi_module.main(input_dir, symbolic_output_dir)

    check_files_equal(
        expected_output_dir / "pyoframe-problem.lp",
        symbolic_output_dir / "pyoframe-problem.lp",
    )
    if not example.many_valid_solutions:
        check_files_equal(
            expected_output_dir / "pyoframe-problem.sol", symbolic_solution_file
        )
        if gurobi_module is not None:
            check_sol_equal(
                expected_output_dir / "gurobipy.sol", symbolic_solution_file
            )

    if example.integer_results_only:
        check_integer_solutions_only(symbolic_solution_file)
        check_integer_solutions_only(dense_solution_file)


def check_results_equal(result1, result2):
    assert result1.status == result2.status
    assert result1.solution.objective == result2.solution.objective


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
