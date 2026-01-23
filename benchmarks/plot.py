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
    df = df.with_columns((pl.col("memory_uss_mb") / 1024).alias("memory_uss_GiB"))

    # Add normalization column
    join_cols = ["size", "solver", "problem"]
    pyoframe_results = df.filter(library="pyoframe").drop("library")
    assert pyoframe_results.height > 0, (
        "Cannot normalize results: no pyoframe data found"
    )
    df = df.join(
        pyoframe_results, on=join_cols, how="left", validate="m:1", suffix="_pyoframe"
    )

    df = df.with_columns(
        (pl.col("time_s") / pl.col("time_s_pyoframe")).alias("time_normalized"),
        (pl.col("memory_uss_GiB") / pl.col("memory_uss_GiB_pyoframe")).alias(
            "memory_uss_normalized"
        ),
    )

    return df


def plot(results: pl.DataFrame, output, problem):
    assert results.height > 0, "No results to plot"

    scale = "linear"
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
            10**i
            for i in range(2, int(math.ceil(math.log10(results["size"].max()))) + 1)
        ]
        x_axis = alt.Axis(values=tick_values, format="e")
        y_axis_time = y_axis_mem = alt.Axis(labelExpr="datum.value + 'x'", grid=True)

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
                    "time_normalized",
                    scale=alt.Scale(type=scale),
                    title="Time / Pyoframe time",
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
                    title="Memory / Pyoframe memory",
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


def plot_combined(results: pl.DataFrame, output):
    plot = None
    for (problem,), problem_df in results.group_by("problem"):
        chart = alt.Chart(problem_df).encode(color=alt.Color("library", legend=None))
        bins = [
            [1, 10, 30, 60, 120],
            [0.1, 0.5, 1, 5, 10, 100],
        ]
        xs = ["time_s_pyoframe", "memory_uss_GiB_pyoframe"]
        ys = ["time_normalized", "memory_uss_normalized"]
        titles = ["Time to construct", "Peak memory usage"]
        x_labels = ["Time for Pyoframe (seconds)", "Memory used by Pyoframe (GiB)"]
        y_labels = ["Time relative to Pyoframe", "Memory relative to Pyoframe"]
        row = None
        for x, y, bin, title, x_label, y_label in zip(
            xs, ys, bins, titles, x_labels, y_labels
        ):
            lines = (
                chart.mark_line(point=True)
                .encode(
                    alt.X(x).scale(type="log", bins=bin, nice=False).title(x_label),
                    alt.Y(y)
                    .axis(labelExpr="datum.value + 'x'", grid=True)
                    .title(y_label),
                )
                .properties(title=title)
            )
            labels = chart.encode(
                alt.X(f"max({x})"),
                alt.Y(y, aggregate=alt.ArgmaxDef(argmax=x)),
                text="library",
            ).mark_text(align="left", dx=4)
            facet = lines + labels
            if row is None:
                row = facet
            else:
                row |= facet
        if plot is None:
            plot = row
        else:
            plot &= row

    plot.save(output)


def plot_combined_v2(results: pl.DataFrame, output):
    plot = None
    for (problem,), problem_df in results.group_by("problem"):
        chart = alt.Chart(problem_df).encode(
            color=alt.condition(
                alt.datum.library == "pyoframe",
                alt.value("black"),
                alt.Color("library", legend=None),
            )
        )

        xs = ["time_s_pyoframe", "memory_uss_GiB_pyoframe"]
        ys = ["time_normalized", "memory_uss_normalized"]
        titles = ["Time to construct", "Peak memory usage"]
        y_labels = ["Time relative to Pyoframe", "Memory relative to Pyoframe"]
        units = ["sec", "GiB"]
        row = None
        for x, y, title, y_label, unit in zip(xs, ys, titles, y_labels, units):
            tick_values = [10**i for i in range(1, 9)]
            lines = (
                chart.mark_line(point=True)
                .encode(
                    alt.X("size")
                    .scale(type="log")
                    .title("Number of variables")
                    .axis(grid=False, format="~s", values=tick_values),
                    alt.Y(y)
                    .axis(labelExpr="datum.value + 'x'", grid=True)
                    .title(y_label),
                )
                .properties(title=title)
            )
            labels = chart.encode(
                alt.X("max(size)"),
                alt.Y(y, aggregate=alt.ArgmaxDef(argmax=x)),
                text="library",
            ).mark_text(align="left", dx=4, fontSize=12)

            pyoframe_data = problem_df.filter(library="pyoframe")
            pyoframe_data = pyoframe_data.with_columns(
                pl.col(x).round_sig_figs(1).map_elements(lambda v: f"{v:g} {unit}")
            )
            pyoframe_labels = (
                alt.Chart(pyoframe_data)
                .encode(alt.X("size"), alt.Y(y), alt.Text(x))
                .mark_text(align="center", dy=-10, fontSize=12)
            )

            pyoframe_background = pyoframe_labels.mark_text(
                align="center",
                stroke="white",
                strokeWidth=5,
                strokeJoin="round",
                strokeOpacity=0.6,
                dy=-10,
                fontSize=12,
            )
            facet = lines + pyoframe_background + pyoframe_labels + labels
            if row is None:
                row = facet
            else:
                row |= facet
        if plot is None:
            plot = row
        else:
            plot &= row

    plot.save(output)


def plot_all_summary():
    df = get_latest_data()
    for (solver,), solver_df in df.group_by("solver"):
        plot_combined(solver_df, CWD / "results" / f"results_{solver}.svg")
        plot_combined_v2(solver_df, CWD / "results" / f"results_v2_{solver}.svg")
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

        plot(problem_df, CWD / "results" / problem / "results.svg", problem)


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

            if group["process_name"].n_unique() > 1:
                group = group.group_by("time_s", "library").agg(
                    pl.col("uss_MiB", "rss_MiB", "vms_MiB", "num_threads").sum(),
                    pl.col("marker").first(),
                )

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
            # panel += (
            #     alt.Chart(group)
            #     .mark_line(strokeWidth=1, strokeDash=[2, 2])
            #     .encode(x=alt.X("time_s"), y=alt.Y("rss_MiB"), color="library:N")
            # )
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
