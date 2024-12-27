from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import pytest
from polars.testing import assert_frame_equal


@dataclass
class Example:
    folder_name: str
    integer_results_only: bool = False
    many_valid_solutions: bool = False
    has_gurobi_version: bool = True
    check_params: Optional[Dict[str, Any]] = None
    skip_solvers: List[str] | None = None


@pytest.mark.parametrize(
    "example",
    [
        Example("diet_problem"),
        Example("facility_problem", check_params={"Method": 2}),
        Example(
            "cutting_stock_problem",
            integer_results_only=True,
            many_valid_solutions=True,
            has_gurobi_version=False,
        ),
        Example(
            "facility_location",
            has_gurobi_version=False,
            many_valid_solutions=True,
            skip_solvers=["highs"],
        ),
    ],
    ids=lambda x: x.folder_name,
)
def test_examples(solver, example: Example):
    """
    For each example, we
    1. Run it twice in a temp directory, once with use_var_names=True and the other time with =False.
    2. Check that their objective values are equal.
    3. Check that the .lp file is the same as in the example directory (only for use_var_names=True).
    4. Check that the .sol file is the same (only when many_valid_solutions=False).
    5. Check that all values in .sol are integers (when integer_results_only=True).
    6. Check that the .sol from the gurobipy version (if it exists) is the same.
    """
    if example.skip_solvers is not None and solver in example.skip_solvers:
        pytest.skip(f"Skipping {solver} for example {example.folder_name}")

    example_dir = Path("tests/examples") / example.folder_name
    input_dir = example_dir / "input_data"
    expected_output_dir = example_dir / "results"
    is_gurobi = solver == "gurobi"
    with TemporaryDirectory(prefix=example.folder_name) as working_dir_str:
        working_dir = Path(working_dir_str)
        symbolic_output_dir = working_dir / "results"
        dense_output_dir = working_dir / "results_dense"

        if working_dir.exists():
            shutil.rmtree(working_dir)
        working_dir.mkdir(parents=True)

        # Dynamically import the main function of the example
        main_module = importlib.import_module(
            f"tests.examples.{example.folder_name}.model"
        )

        if is_gurobi:
            symbolic_solution_file = symbolic_output_dir / "pyoframe-problem.sol"
            dense_solution_file = dense_output_dir / "pyoframe-problem.sol"

        symbolic_kwargs = dict(
            directory=symbolic_output_dir,
            use_var_names=True,
        )
        dense_kwargs = dict(directory=dense_output_dir)

        if input_dir.exists():
            symbolic_kwargs["input_dir"] = input_dir
            dense_kwargs["input_dir"] = input_dir

        symbolic_model = main_module.main(**symbolic_kwargs)
        dense_model = main_module.main(**dense_kwargs)
        assert symbolic_model.objective.value == dense_model.objective.value
        if is_gurobi:
            symbolic_model.write(symbolic_solution_file)
            dense_model.write(dense_solution_file)

        if example.check_params is not None and is_gurobi:
            for param, value in example.check_params.items():
                assert getattr(dense_model.params, param) == value
                assert getattr(symbolic_model.params, param) == value

        assert dense_model.objective.value == symbolic_model.objective.value
        check_results_dir_equal(
            expected_output_dir,
            symbolic_output_dir,
            check_sol=not example.many_valid_solutions and is_gurobi,
            check_lp=is_gurobi,
        )
        check_results_dir_equal(
            dense_output_dir,
            symbolic_output_dir,
            check_sol=not example.many_valid_solutions and is_gurobi,
            check_lp=False,
        )

        if example.integer_results_only and is_gurobi:
            check_integer_solutions_only(symbolic_solution_file)
            check_integer_solutions_only(dense_solution_file)

        gurobi_module = None
        if example.has_gurobi_version and is_gurobi:
            gurobi_module = importlib.import_module(
                f"tests.examples.{example.folder_name}.model_gurobipy"
            )

            gurobi_module.main(input_dir, symbolic_output_dir)
            if not example.many_valid_solutions:
                check_sol_equal(
                    expected_output_dir / "gurobipy.sol", symbolic_solution_file
                )


def check_results_dir_equal(expected_dir, actual_dir, check_sol, check_lp=True):
    for file in actual_dir.iterdir():
        assert (
            expected_dir / file.name
        ).exists(), f"File {file.name} not found in expected directory"

        expected = expected_dir / file.name
        if file.suffix == ".sol":
            if check_sol:
                check_sol_equal(expected, file)
        elif file.suffix == ".lp":
            if check_lp:
                check_lp_equal(expected, file)
        else:
            if check_sol:
                df1 = pl.read_csv(expected)
                df2 = pl.read_csv(file)
                assert_frame_equal(df1, df2, check_row_order=False)


def check_lp_equal(file_expected: Path, file_actual: Path):
    def keep_line(line):
        return "\\ Signature: 0x" not in line and line.strip() != ""

    with open(file_expected) as f:
        expected = "\n".join(filter(keep_line, f.readlines()))
    with open(file_actual) as f:
        actual = "\n".join(filter(keep_line, f.readlines()))
    assert (
        expected == actual
    ), f"LP files {file_expected} and {file_actual} are different"


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
        assert (
            expected_value - tol <= actual_value <= expected_value + tol
        ), f"Solution file does not match expected values {expected_sol_file}"


def parse_gurobi_sol(sol_file_path) -> List[Tuple[str, float]]:
    with open(sol_file_path, mode="r") as f:
        sol = f.readlines()
    sol = [line.strip() for line in sol]
    sol = [line for line in sol if not (line.startswith("#") or line == "")]
    sol = sorted(sol)
    sol = [line.partition(" ") for line in sol]
    sol = {name: float(value) for name, _, value in sol}
    if "ONE" in sol:
        assert sol["ONE"] == 1
        del sol["ONE"]
    return list(sol.items())
