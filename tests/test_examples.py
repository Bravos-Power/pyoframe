from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List, Optional, Tuple

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pyoframe as pf
from pyoframe.constants import SUPPORTED_SOLVERS


@dataclass
class Example:
    folder_name: str
    unique_solution: bool = True
    skip_solvers: List[str] | None = None

    def import_model_module(self):
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        return importlib.import_module(f"tests.examples.{self.folder_name}.model")

    def import_solve_func(self):
        return self.import_model_module().solve_model

    def get_results_path(self):
        path = Path("tests/examples") / self.folder_name / "results"
        assert path.exists(), (
            f"Results directory {path} does not exist. Working directory: {os.getcwd()}"
        )
        return path

    def get_solve_with_gurobipy(self) -> Optional[Any]:
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        try:
            return importlib.import_module(
                f"tests.examples.{self.folder_name}.model_gurobipy"
            ).main
        except ModuleNotFoundError:
            return None


EXAMPLES = [
    Example("diet_problem"),
    Example("facility_problem"),
    Example(
        "cutting_stock_problem",
        unique_solution=False,
    ),
    Example(
        "facility_location",
        unique_solution=False,
        skip_solvers=["highs"],  # Has quadratics
    ),
    Example("sudoku"),
    Example("pumped_storage"),
    Example("production_planning"),
]


def compare_results_dir(expected_dir, actual_dir):
    for file in actual_dir.iterdir():
        assert (expected_dir / file.name).exists(), (
            f"File {file.name} not found in expected directory"
        )

        expected = expected_dir / file.name
        if file.suffix == ".sol":
            check_sol_equal(expected, file)
        elif file.suffix == ".lp":
            check_lp_equal(expected, file)
        elif file.suffix == ".csv":
            df1 = pl.read_csv(expected)
            df2 = pl.read_csv(file)
            assert_frame_equal(df1, df2, check_row_order=False)
        else:
            raise ValueError(f"Unexpected file {file}")


def check_lp_equal(file_expected: Path, file_actual: Path):
    def keep_line(line):
        return "\\ Signature: 0x" not in line and line.strip() != ""

    with open(file_expected) as f1:
        with open(file_actual) as f2:
            for line1, line2 in zip(
                filter(keep_line, f1.readlines()), filter(keep_line, f2.readlines())
            ):
                assert line1.strip() == line2.strip(), (
                    f"LP files {file_expected} and {file_actual} are different"
                )


def check_integer_solutions_only(sol_file):
    sol = parse_sol(sol_file)
    for name, value in sol:
        assert value.is_integer(), f"Variable {name} has non-integer value {value}"


def check_sol_equal(expected_sol_file, actual_sol_file):
    # Remove comments and empty lines
    expected_result = parse_sol(expected_sol_file)
    actual_result = parse_sol(actual_sol_file)

    tol = 1e-8
    for (expected_name, expected_value), (actual_name, actual_value) in zip(
        expected_result, actual_result
    ):
        assert expected_name == actual_name, (
            f"Variable names do not match: {expected_name} != {actual_name}\n{expected_result}\n\n{actual_result}"
        )
        assert expected_value - tol <= actual_value <= expected_value + tol, (
            f"Solution file does not match expected values {expected_sol_file}"
        )


def parse_sol(sol_file_path) -> List[Tuple[str, float]]:
    with open(sol_file_path, mode="r") as f:
        sol = f.read()
    sol = sol.partition("\nHiGHS v1\n")[0]  # Cut out everything after this
    sol = [line.strip() for line in sol.split("\n")]
    sol = [line for line in sol if not (line.startswith("#") or line == "")]
    sol = [line.partition(" ") for line in sol]
    sol = sorted(sol, key=lambda x: x[0])
    sol_numeric = {}
    for name, _, value in sol:
        # So that comparisons with gurobipy work
        if name == "ONE":
            continue

        try:
            sol_numeric[name] = float(value)
        except ValueError:
            pass

    return list(sol_numeric.items())


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda x: x.folder_name)
def test_examples(example, solver, use_var_names):
    if example.skip_solvers and solver in example.skip_solvers:
        pytest.skip(f"Skipping {solver} for example {example.folder_name}")

    solver_func = example.import_solve_func()
    model = solver_func(use_var_names)
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        write_results(example, model, tmpdir)
        compare_results_dir(example.get_results_path(), tmpdir)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda x: x.folder_name)
def test_gurobi_model_matches(example, solver):
    if solver != "gurobi":
        pytest.skip("This test only runs for gurobi")
    gurobipy_solve = example.get_solve_with_gurobipy()
    if gurobipy_solve is None:
        pytest.skip("No gurobi model found")
    result = gurobipy_solve()
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        write_results_gurobipy(result, tmpdir)
        compare_results_dir(example.get_results_path(), tmpdir)


def write_results(example: Example, model, results_dir):
    readability = "pretty" if model.use_var_names else "machine"
    model.write(results_dir / f"problem-{model.solver_name}-{readability}.lp")

    if model.objective is not None:
        pl.DataFrame({"value": [model.objective.value]}).write_csv(
            results_dir / "objective.csv"
        )

    if example.unique_solution:
        model.write(results_dir / f"solution-{model.solver_name}-{readability}.sol")

        module = example.import_model_module()
        if hasattr(module, "write_solution"):
            module.write_solution(model, results_dir)
        else:
            for v in model.variables:
                v.solution.write_csv(results_dir / f"{v.name}.csv")  # type: ignore
            for c in model.constraints:
                try:
                    c.dual.write_csv(results_dir / f"{c.name}.csv")
                except RuntimeError as e:
                    if "Unable to retrieve attribute 'Pi'" in str(e):
                        pass
                    else:
                        raise e


def write_results_gurobipy(model_gpy, results_dir):
    model_gpy.write(str(results_dir / "solution-gurobi-pretty.sol"))
    pl.DataFrame({"value": [model_gpy.getObjective().getValue()]}).write_csv(
        results_dir / "objective.csv"
    )


if __name__ == "__main__":
    selection = int(
        input(
            "Choose which of the following results you'd like to rewrite.\n0: ALL\n"
            + "\n".join(
                str(i + 1) + ": " + example.folder_name
                for i, example in enumerate(EXAMPLES)
            )
            + "\n"
        )
    )

    if selection == 0:
        selection = EXAMPLES
    else:
        selection = [EXAMPLES[selection - 1]]

    for example in selection:
        results_dir = example.get_results_path()
        if results_dir.exists():
            shutil.rmtree(results_dir)
        solve_model = example.import_solve_func()
        for solver in SUPPORTED_SOLVERS:
            if example.skip_solvers and solver in example.skip_solvers:
                continue
            pf.Config.default_solver = solver
            for use_var_names in [True, False]:
                write_results(
                    example,
                    solve_model(use_var_names),
                    results_dir,
                )
