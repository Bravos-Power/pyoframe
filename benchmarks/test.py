"""Calls run.py using the test configuration file."""

import argparse

from plot import plot_all
from run import read_config, run_all_benchmarks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse", action="store_true", help="Reuse past results if available."
    )
    args = parser.parse_args()

    run_all_benchmarks(
        read_config(name="config.test.yaml"), ignore_past_results=not args.reuse
    )

    plot_all(config_name="config.test.yaml")
