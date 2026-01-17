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

    scale = "linear" if not normalized else "linear"
    x_label = "Number of variables"

    results = results.with_columns(
        pl.when(pl.col("library") == "pyoframe")
        .then(pl.lit("_pyoframe_"))
        .otherwise(pl.col("library"))
        .alias("library")
    )

    if not normalized:
        results = results.with_columns(pl.col("memory_uss_mb") / 1e3)

    combined_plot = None
    for solver in results.get_column("solver").unique():
        solver_results = results.filter(pl.col("solver") == solver)

        tick_values = [
            10**i
            for i in range(2, int(math.ceil(math.log10(results["size"].max()))) + 1)
        ]
        x_axis = alt.Axis(values=tick_values, format="e")
        y_axis_time = y_axis_mem = alt.Axis(labelExpr="datum.value + 'x'", grid=True)
        if not normalized:
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
                    else "Peak memory usage (USS, GB)",
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

    combined_plot = (
        combined_plot.configure_axis(
            labelFontSize=12, titleFontSize=14, labelFont="Arial", titleFont="Arial"
        )
        .configure_title(fontSize=18, font="Arial", anchor="middle")
        .configure_legend(labelFontSize=14, titleFontSize=16)
    )
    combined_plot.save(output)


def plot_all_summary():
    df = get_latest_data()
    for problem in config["problems"]:
        problem_df = df.filter(problem=problem)
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


def plot_memory_usage_over_time():
    for problem in config["problems"]:
        problem_mem_log_dir = CWD / "results" / problem / "mem_log"
        if not problem_mem_log_dir.exists():
            continue

        all_data = []

        for file in problem_mem_log_dir.glob("*.parquet"):
            file_terms = file.stem.split("_")
            day, time, library, solver, size = (
                file_terms[0],
                file_terms[1],
                file_terms[2],
                file_terms[3],
                file_terms[4],
            )
            df = pl.read_parquet(file)
            df = df.with_columns(
                timestamp=pl.lit(f"{day} {time}"),
                size=pl.lit(int(size)),
                library=pl.lit(library),
                solver=pl.lit(solver),
            )
            all_data.append(df)
        all_data_df = pl.concat(all_data)
        most_recent = all_data_df.group_by(["library", "solver", "size"]).agg(
            pl.col("timestamp").max()
        )
        only_most_recent = all_data_df.join(
            most_recent,
            on=["library", "solver", "size", "timestamp"],
            how="inner",
        )

        plt = None
        only_most_recent = only_most_recent.sort("size")

        only_most_recent = only_most_recent.with_columns(
            pl.col("uss_MiB", "vms_MiB", "rss_MiB") / 1024
        )

        for (size, solver), group in only_most_recent.group_by(
            ["size", "solver"], maintain_order=True
        ):
            if group.height == 0:
                continue
            panel = (
                alt.Chart(group)
                .mark_line(strokeWidth=1)
                .encode(
                    x=alt.X("time_s", title="Elapsed time (s)"),
                    y=alt.Y("uss_MiB", title="Memory usage (GiB, USS)"),
                    color="library:N",
                )
                .properties(title=f"Memory usage over time (N={size}, {solver})")
            )
            # add rss as dashed lines
            panel += (
                alt.Chart(group)
                .mark_line(strokeWidth=1, strokeDash=[2, 2])
                .encode(x=alt.X("time_s"), y=alt.Y("rss_MiB"), color="library:N")
            )
            keypoints = group.filter(pl.col("marker").is_not_null())
            if keypoints.height > 0:
                panel += keypoints.plot.scatter(
                    x="time_s",
                    y="uss_MiB",
                    color="library:N",
                    shape="marker:N",
                )

            if plt is None:
                plt = panel
            else:
                plt |= panel
        if plt is not None:
            plt.save(CWD / "results" / problem / "memory_usage_over_time.svg")


if __name__ == "__main__":
    plot_all_summary()
    plot_memory_usage_over_time()
