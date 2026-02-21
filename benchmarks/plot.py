"""Produces plots from benchmark results."""

import os
from pathlib import Path

import altair as alt
import polars as pl
import yaml


def get_latest_data(base_path: Path):
    df = pl.read_csv(base_path / "benchmark_results.csv")
    df = df.filter(pl.col("error").is_null())
    df = df.group_by(["problem", "library", "solver", "size"]).agg(pl.col("*").last())
    df = df.with_columns((pl.col("max_memory_uss_mb") / 1024).alias("memory_uss_GiB"))
    df = df.with_columns(
        construct_time_s=pl.col("total_time_s") - pl.col("solve_time_s")
    )

    # Add normalization column
    join_cols = ["size", "solver", "problem"]
    pyoframe_results = df.filter(library="pyoframe").drop("library")
    assert pyoframe_results.height > 0, (
        f"Cannot normalize results: no pyoframe data found\n{df}"
    )
    df = df.join(
        pyoframe_results, on=join_cols, how="left", validate="m:1", suffix="_pyoframe"
    )

    df = df.with_columns(
        (pl.col("construct_time_s") / pl.col("construct_time_s_pyoframe")).alias(
            "construct_time_normalized"
        ),
        (pl.col("memory_uss_GiB") / pl.col("memory_uss_GiB_pyoframe")).alias(
            "memory_uss_normalized"
        ),
    )

    return df


def plot_combined(results: pl.DataFrame, output):
    panels = [[], []]
    for (problem,), problem_df in results.group_by("problem", maintain_order=True):
        chart = alt.Chart(problem_df).encode(
            color=alt.condition(
                alt.datum.library == "pyoframe",
                alt.value("black"),
                alt.Color("library", legend=None),
            )
        )

        lib_names = ["construct_time_s_pyoframe", "memory_uss_GiB_pyoframe"]
        ys = ["construct_time_normalized", "memory_uss_normalized"]
        titles = ["Time to construct", "Peak memory usage"]
        units = ["sec", "GiB"]
        max_ys = [10, 5]
        for label, y, title, unit, panel_col, max_y in zip(
            lib_names, ys, titles, units, panels, max_ys
        ):
            tick_values = [10**i for i in range(1, 9)]
            lines = (
                chart.mark_line(point=True, clip=True)
                .encode(
                    alt.X("num_variables")
                    .scale(type="log")
                    .title("Number of variables")
                    .axis(grid=False, format="~s", values=tick_values),
                    alt.Y(y)
                    .axis(labelExpr="datum.value + 'x'", grid=True)
                    .title(problem)
                    .scale(domain=[0, max_y]),
                )
                .properties(title=title)
            )
            lib_names = chart.encode(
                alt.X("max(num_variables)"),
                alt.Y(y, aggregate=alt.ArgmaxDef(argmax=label)),
                text="library",
            ).mark_text(align="left", dx=4, fontSize=12)

            pyoframe_data = problem_df.filter(library="pyoframe")
            pyoframe_data = pyoframe_data.with_columns(
                pl.col(label).round_sig_figs(1).map_elements(lambda v: f"{v:g} {unit}")
            )
            pf_labels = (
                alt.Chart(pyoframe_data)
                .encode(alt.X("num_variables"), alt.Y(y), alt.Text(label))
                .mark_text(align="center", dy=-10, fontSize=12)
            )

            pf_labels_background = pf_labels.mark_text(
                align="center",
                stroke="white",
                strokeWidth=5,
                strokeJoin="round",
                strokeOpacity=0.6,
                dy=-10,
                fontSize=12,
            )
            facet = lines + pf_labels_background + pf_labels + lib_names

            panel_col.append(facet)

    columns = [alt.vconcat(*panel_col) for panel_col in panels]

    y_labels = [
        alt.Chart(pl.DataFrame({"y": [0.5]}))
        .mark_text(
            text=title,
            angle=270,
            align="center",
            baseline="middle",
            fontSize=16,
            dx=10,
        )
        .encode(y=alt.Y("y", axis=None).scale(domain=[0, 1]))
        .properties(width=20)
        for title in [
            "Time relative to Pyoframe",
            "Memory relative to Pyoframe",
        ]
    ]

    plot = y_labels[0] | columns[0] | y_labels[1] | columns[1]

    plot.configure_view(stroke=None).save(output)


def plot_solve_time(df, output_path):
    df = df.filter(pl.col("solve_time_s") > 0)

    # TODO normalize


def plot_all_summary(base_path: Path, config):
    df = get_latest_data(base_path)

    for (solver,), solver_df in df.group_by("solver"):
        plot_solve_time(solver_df, base_path / f"solve_time_{solver}.svg")
        plot_combined(solver_df, base_path / f"results_{solver}.svg")
    for problem in config["problems"]:
        problem_df = df.filter(problem=problem)
        if problem_df.height == 0:
            continue

        if not os.path.exists(base_path / problem):
            os.makedirs(base_path / problem)


def plot_memory_usage_over_time(base_path: Path, config):
    for problem in config["problems"]:
        problem_mem_log_dir = base_path / problem / "mem_log"
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
            plt.save(base_path / problem / "memory_usage_over_time.svg")


def plot_all(config_name="config.yaml"):
    cwd = Path(__file__).parent
    with open(cwd / config_name) as f:
        config = yaml.safe_load(f)
    base_path = cwd / "results" / config["name"]

    plot_all_summary(base_path, config)
    plot_memory_usage_over_time(base_path, config)


if __name__ == "__main__":
    plot_all()
