"""Produces plots from benchmark results."""

import importlib
import math
import os
from pathlib import Path

import altair as alt
import polars as pl
import yaml

CWD = Path(__file__).parent
BENCHMARKS_DB = CWD / "results" / "benchmark_results.csv"

with open(CWD / "config.yaml") as f:
    config = yaml.safe_load(f)


def get_latest_data():
    df = pl.read_csv(BENCHMARKS_DB)
    df = df.filter(pl.col("error").is_null())
    df = df.group_by(["problem", "library", "solver", "size"]).agg(pl.col("*").last())
    df = df.with_columns((pl.col("time_s") / 60).alias("time_min"))

    return df


def normalize_results(results: pl.DataFrame):
    join_cols = ["size", "solver", "library"]

    pyoframe_results = results.filter(pl.col("library") == "pyoframe").drop("library")
    assert pyoframe_results.height > 0, (
        "Cannot normalize results: no pyoframe data found"
    )
    results = results.join(
        pyoframe_results,
        on=["size", "solver"],
    )

    results = results.select(
        *(
            join_cols
            + [
                pl.col(c) / pl.col(f"{c}_right")
                for c in ["time_s", "memory_uss_mb", "time_min"]
            ]
        )
    )

    return results


def plot(results: pl.DataFrame, output, problem, log_y=True, normalized=False):
    assert results.height > 0, "No results to plot"

    scale = "log" if log_y else "linear"
    x_label = "Number of variables"

    results = results.with_columns(
        pl.when(pl.col("library") == "pyoframe")
        .then(pl.lit("_pyoframe_"))
        .otherwise(pl.col("library"))
        .alias("library")
    )

    combined_plot = None
    for solver in results.get_column("solver").unique():
        solver_results = results.filter(pl.col("solver") == solver)

        tick_values = [
            10**i for i in range(int(math.ceil(math.log10(results["size"].max()))) + 1)
        ]
        x_axis = alt.Axis(values=tick_values, format="e")
        y_axis_time = y_axis_mem = alt.Undefined
        if log_y:
            max_time = solver_results["time_min"].max()
            max_mem = solver_results["memory_uss_mb"].max()

            time_ticks = [
                10**i for i in range(-2, int(math.ceil(math.log10(max_time))) + 1)
            ]
            y_axis_time = alt.Axis(values=time_ticks)
            mem_ticks = [10**i for i in range(int(math.ceil(math.log10(max_mem))) + 1)]
            y_axis_mem = alt.Axis(values=mem_ticks)

        left_plot = (
            alt.Chart(solver_results)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "size",
                    scale=alt.Scale(type="log"),
                    title=x_label,
                    axis=x_axis,
                ),
                y=alt.Y(
                    "time_min",
                    scale=alt.Scale(type=scale),
                    title="Time / Pyoframe time" if normalized else "Time (min)",
                    axis=y_axis_time,
                ),
                color="library",
            )
            .properties(title="Time to construct")
        )

        right_plot = (
            alt.Chart(solver_results)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "size", scale=alt.Scale(type="log"), title=x_label, axis=x_axis
                ),
                y=alt.Y(
                    "memory_uss_mb",
                    scale=alt.Scale(type=scale),
                    title="Memory / Pyoframe memory"
                    if normalized
                    else "Peak memory usage (USS, MB)",
                    axis=y_axis_mem,
                ),
                color="library",
            )
            .properties(title="Peak memory usage")
        )

        if combined_plot is None:
            combined_plot = left_plot | right_plot
        else:
            combined_plot &= left_plot | right_plot
    combined_plot.save(output)


if __name__ == "__main__":
    df = get_latest_data()
    for problem in config["problems"]:
        problem_df = df.filter(pl.col("problem") == problem)
        if problem_df.height == 0:
            continue

        if not os.path.exists(CWD / "results" / problem):
            os.makedirs(CWD / "results" / problem)

        problem_lib = importlib.import_module(problem)
        problem_lib_mapper = None
        try:
            problem_lib_mapper = problem_lib.size_to_num_variables
        except AttributeError:
            pass

        if problem_lib_mapper is not None:
            problem_df = problem_df.with_columns(
                problem_lib_mapper("size").alias("size")
            )

        plot(problem_df, CWD / "results" / problem / "combined_results.png", problem)
        normalized_df = normalize_results(problem_df)
        plot(
            normalized_df,
            CWD / "results" / problem / "normalized_results.png",
            problem,
            log_y=False,
            normalized=True,
        )
