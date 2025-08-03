import argparse
import importlib
import os
import subprocess
from pathlib import Path
from typing import Collection

from pyoframe._constants import SUPPORTED_SOLVERS


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

    if args.library != "jump":
        # import solve function dynamically
        benchmark = importlib.import_module(
            f"benchmarks.{args.problem}.bm_{args.library.lower()}"
        ).Bench

        benchmark(solver=args.solver, size=args.size).run()
    else:
        subprocess.run(
            [
                "julia",
                "--project=.",
                f"src/{args.problem}/bm_jump.jl",
                args.solver,
                str(args.size),
            ],
            check=True,
        )
