"""Module to automatically run and test the examples to ensure consistent results over time."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pyoframe as pf
from pyoframe._constants import SUPPORTED_SOLVERS, _Solver
from tests.util import get_tol_pl


@dataclass
class Example:
    folder_name: str
    unique_solution: bool = True
    is_mip: bool = False
    is_quadratically_constrained: bool = False
    is_non_convex: bool = False

    def supports_solver(self, solver: _Solver) -> bool:
        if self.is_mip and not solver.supports_integer_variables:
            return False
        if (
            self.is_quadratically_constrained
            and not solver.supports_quadratic_constraints
        ):
            return False
        if self.is_non_convex and not solver.supports_non_convex:
            return False
        return True

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

    def get_solve_with_gurobipy(self) -> Any | None:
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
    Example("facility_problem", is_mip=True),
    Example("cutting_stock_problem", unique_solution=False, is_mip=True),
    Example(
        "facility_location",
        unique_solution=False,
        is_mip=True,
        is_quadratically_constrained=True,
        is_non_convex=True,
    ),
    Example("sudoku", is_mip=True),
    Example("production_planning"),
    Example("portfolio_optim"),
    Example("pumped_storage", is_mip=True),
]


def compare_results_dir(expected_dir, test_dir, solver):
    for file in test_dir.iterdir():
        assert (expected_dir / file.name).exists(), (
            f"File {file.name} not found in expected directory"
        )

        expected = expected_dir / file.name
        if file.suffix == ".sol":
            check_sol_equal(expected, file)
        elif file.suffix == ".lp":
            if pf.Config.maintain_order:
                check_lp_equal(expected, file)
        elif file.suffix == ".csv":
            df1 = pl.read_csv(expected)
            df2 = pl.read_csv(file)
            assert_frame_equal(df1, df2, check_row_order=False, **get_tol_pl(solver))
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

    tol = 1e-8 if pf.Config.maintain_order else 1e-6
    for (expected_name, expected_value), (actual_name, actual_value) in zip(
        expected_result, actual_result
    ):
        assert expected_name == actual_name, (
            f"Variable names do not match: {expected_name} != {actual_name}\n{expected_result}\n\n{actual_result}"
        )
        assert expected_value - tol <= actual_value <= expected_value + tol, (
            f"Variable {actual_name} in solution file ({actual_value}) does not match expected value ({expected_value})"
        )


def parse_sol(sol_file_path) -> list[tuple[str, float]]:
    with open(sol_file_path) as f:
        sol = f.read()
    sol = sol.partition("\nHiGHS v1\n")[0]  # Cut out everything after this
    sol = sol.partition("\n# Dual solution values\n")[
        0
    ]  # Cut out everything after this too
    sol = [line.strip() for line in sol.split("\n")]
    sol = [line for line in sol if not (line.startswith("#") or line == "")]
    sol = [line.partition(" ") for line in sol]
    sol = sorted(sol, key=lambda x: x[0])
    sol_numeric = {}
    for name, _, value in sol:
        # So that comparisons with gurobipy work
        if name == "ONE":
            continue

        if name in sol_numeric:
            raise ValueError(f"Duplicate variable name {name} in solution file")

        try:
            sol_numeric[name] = float(value)
        except ValueError:
            pass

    return list(sol_numeric.items())


def pytest_generate_tests(metafunc):
    if "test_examples_config" in metafunc.fixturenames:
        metafunc.parametrize(
            "test_examples_config",
            [
                (True, True),
                (True, False),
                (False, True),
            ],
            ids=[
                "",
                "unordered",
                "unnamed",
            ],
        )


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda x: x.folder_name)
def test_examples(example, solver: _Solver, test_examples_config):
    use_var_names, maintain_order = test_examples_config

    if not example.supports_solver(solver):
        pytest.skip(
            f"Skipping example {example.folder_name} for solver {solver.name} due to unsupported features"
        )

    pf.Config.maintain_order = maintain_order
    pf.Config.default_solver = solver

    solver_func = example.import_solve_func()
    model = solver_func(use_var_names)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        write_results(example, model, tmpdir, solver)
        compare_results_dir(example.get_results_path(), tmpdir, solver)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda x: x.folder_name)
def test_gurobi_model_matches(example):
    gurobipy_solve = example.get_solve_with_gurobipy()
    if gurobipy_solve is None:
        pytest.skip("No gurobi model found")
    result = gurobipy_solve()
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        result.write(str(tmpdir / "solution-gurobi-pretty.sol"))
        pl.DataFrame({"value": [result.getObjective().getValue()]}).write_csv(
            tmpdir / "objective.csv"
        )
        compare_results_dir(example.get_results_path(), tmpdir, "gurobi")


def write_results(example: Example, model: pf.Model, results_dir, solver: _Solver):
    supports_write = solver.supports_write and (
        model.solver_uses_variable_names or not solver.accelerate_with_repeat_names
    )
    if supports_write:
        readability = "pretty" if model.solver_uses_variable_names else "machine"
        model.write(results_dir / f"problem-{model.solver.name}-{readability}.lp")

    if model.has_objective:
        pl.DataFrame({"value": [model.objective.value]}).write_csv(
            results_dir / "objective.csv"
        )

    if example.unique_solution:
        if supports_write:
            model.write(results_dir / f"solution-{model.solver.name}-{readability}.sol")

        module = example.import_model_module()
        if hasattr(module, "write_solution"):
            module.write_solution(model, results_dir)
        else:
            for v in model.variables:
                if hasattr(v.solution, "write_csv"):
                    v.solution.write_csv(results_dir / f"{v.name}.csv")
                else:
                    # Handle scalar values
                    pl.DataFrame({v.name: [v.solution]}).write_csv(
                        results_dir / f"{v.name}.csv"
                    )

            if solver.supports_duals and not example.is_mip:
                for c in model.constraints:
                    dual = c.dual
                    if hasattr(dual, "write_csv"):
                        dual.write_csv(results_dir / f"{c.name}.csv")
                    else:
                        # Handle scalar values
                        pl.DataFrame({c.name: [dual]}).write_csv(
                            results_dir / f"{c.name}.csv"
                        )


if __name__ == "__main__":
    problem_selection = int(
        input(
            "Choose which of the following results you'd like to rewrite.\n0: ALL\n"
            + "\n".join(
                str(i + 1) + ": " + example.folder_name
                for i, example in enumerate(EXAMPLES)
            )
            + "\n"
        )
    )

    solver_selection = int(
        input(
            "Choose which of the following solvers you'd like to rewrite.\n0: ALL\n"
            + "\n".join(
                str(i + 1) + ": " + solver.name
                for i, solver in enumerate(SUPPORTED_SOLVERS)
            )
            + "\n"
        )
    )

    if problem_selection == 0:
        problem_selection = EXAMPLES
    else:
        problem_selection = [EXAMPLES[problem_selection - 1]]

    solvers = (
        SUPPORTED_SOLVERS
        if solver_selection == 0
        else [SUPPORTED_SOLVERS[solver_selection - 1]]
    )

    for example in problem_selection:
        results_dir = example.get_results_path()
        if solver_selection == 0:
            if results_dir.exists():
                shutil.rmtree(results_dir)
        solve_model = example.import_solve_func()
        for solver in solvers:
            if not example.supports_solver(solver):
                continue
            pf.Config.default_solver = solver.name
            for use_var_names in [True, False]:
                write_results(example, solve_model(use_var_names), results_dir, solver)
