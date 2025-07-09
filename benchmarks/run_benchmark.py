import argparse
import importlib
import os
from pathlib import Path
from typing import Collection

from pyoframe.constants import SUPPORTED_SOLVERS


def get_problems() -> Collection[str]:
    return [
        c.name
        for c in os.scandir(Path(__file__).parent)
        if c.is_dir() and not c.name.startswith("_")
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run_benchmark", description="Run a benchmark with a given solver."
    )
    parser.add_argument("problem", choices=get_problems())
    parser.add_argument("--solver", choices=SUPPORTED_SOLVERS)
    parser.add_argument("--size", type=int)
    parser.add_argument("--library")
    args = parser.parse_args()

    # import solve function dynamically
    solve_func = importlib.import_module(f"benchmarks.bm_{args.library.lower()}").solve

    solve_func(solver=args.solver, size=args.size)
