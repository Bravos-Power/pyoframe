"""Calls run.py using the test configuration file."""

import argparse

from plot import plot_all
from run import read_config, run_all_benchmarks

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--reuse", action="store_true", help="Reuse past results if available."
    )
    argparser.add_argument(
        "-p",
        "--problem",
        default=None,
        help="Only run a specific problem (e.g., 'energy_planning').",
    )
    argparser.add_argument(
        "-l",
        "--library",
        default=None,
        help="Only run a specific library (e.g., 'pyoframe').",
    )
    argparser.add_argument(
        "--config",
        type=str,
        default="config.test.yaml",
        help="Path to config YAML file.",
    )
    argparser.add_argument(
        "--skip-building-inputs", action="store_true", help="Skip building input files."
    )
    argparser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Fail the script if any benchmark fails.",
    )
    args = argparser.parse_args()

    config = read_config(name=args.config)

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

    run_all_benchmarks(
        config,
        ignore_past_results=not args.reuse,
        build_inputs=not args.skip_building_inputs,
        fail_on_error=args.fail_on_error,
    )

    plot_all(config_name="config.test.yaml")
