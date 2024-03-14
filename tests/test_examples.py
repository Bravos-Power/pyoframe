from pathlib import Path
from typing import List

from tests.examples.diet_model.model import main as main_diet
from tests.examples.diet_model.model_gurobipy import main as main_diet_gurobipy


def test_diet_example():
    working_dir = Path("tests/examples/diet_model/")
    result_dir = working_dir / "results"
    # Delete previous results
    if result_dir.exists():
        for file in result_dir.iterdir():
            file.unlink()

    main_diet_gurobipy(working_dir)
    main_diet(working_dir)

    # Check that two files results are equal
    with open(result_dir / "diet.sol") as f1, open(
        result_dir / "diet-gurobipy.sol"
    ) as f2:
        gurobi_sol_equal(f1.readlines(), f2.readlines())


def gurobi_sol_equal(sol1: List[str], sol2: List[str]):
    # Remove comments and empty lines
    sol1 = [line for line in sol1 if not (line.startswith("#") or line.strip() == "")]
    sol2 = [line for line in sol2 if not (line.startswith("#") or line.strip() == "")]

    # Sort lines
    sol1 = sorted(sol1)
    sol2 = sorted(sol2)

    # Compare
    for line1, line2 in zip(sol1, sol2):
        name1, _, value1 = line1.partition(" ")
        name2, _, value2 = line2.partition(" ")
        assert name1 == name2
        tol = 1e-8
        bound = float(value2)
        assert bound - tol <= float(value1) <= bound + tol
