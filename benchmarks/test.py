"""Calls run.py using the test configuration file."""

import argparse

from plot import plot_all
from run import read_config, run_all_benchmarks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse", action="store_true", help="Reuse past results if available."
    )
    parser.add_argument(
        "-p",
        "--problem",
        default=None,
        help="Only run a specific problem (e.g., 'energy_planning').",
    )
    parser.add_argument(
        "-l",
        "--library",
        default=None,
        help="Only run a specific library (e.g., 'pyoframe').",
    )
    args = parser.parse_args()

    config = read_config(name="config.test.yaml")

    if args.problem is not None:
        if args.problem not in config["problems"]:
            raise ValueError(
                f"Problem '{args.problem}' not found in config. Options are: {list(config['problems'].keys())}"
            )
        config["problems"] = {
            k: v for k, v in config["problems"].items() if k == args.problem
        }

    if args.library is not None:
        if args.library not in config["libraries"]:
            raise ValueError(
                f"Library '{args.library}' not found in config. Options are: {config['libraries']}"
            )
        config["libraries"] = [args.library]

    run_all_benchmarks(config, ignore_past_results=not args.reuse)

    plot_all(config_name="config.test.yaml")
