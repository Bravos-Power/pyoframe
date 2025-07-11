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
    parser.add_argument("--solver", choices=[s.name for s in SUPPORTED_SOLVERS])
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--library")
    args = parser.parse_args()

    # import solve function dynamically
    benchmark = importlib.import_module(
        f"{args.problem}.bm_{args.library.lower()}"
    ).Bench

    if (
        args.size is not None
        and benchmark.MAX_SIZE is not None
        and args.size > benchmark.MAX_SIZE
    ):
        raise ValueError(
            f"Size {args.size} exceeds maximum size {benchmark.MAX_SIZE} for problem {args.problem}."
        )

    benchmark(solver=args.solver, size=args.size).run()
