"""Script to plot the benchmark results for a select group of benchmark problems."""

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import warnings
    from pathlib import Path

    import marimo as mo  # keep for it to run headless
    import matplotlib.pyplot as plt
    import polars as pl
    from matplotlib.patches import Patch

    return Patch, Path, pl, plt, mo, warnings


@app.cell
def _(Path):
    RESULTS_FOLDER = Path(__file__).parent / "results/main_v3"
    BENCHMARK_PROBLEMS = {
        ("simple_problem", 10_000_000): "Trivial\nData\nProblem",
        ("energy_planning_capacity_expansion", 168): "Capacity\nExpansion\nProblem",
        (
            "energy_planning_security_constrained_dispatch",
            48,
        ): "Electricity\nDispatch\nProblem",
        ("facility_location", 128): "Facility\nLocation\nProblem*",
    }
    return BENCHMARK_PROBLEMS, RESULTS_FOLDER


@app.cell
def _(BENCHMARK_PROBLEMS, RESULTS_FOLDER, pl):
    latest_runs = pl.read_csv(f"{RESULTS_FOLDER}/benchmark_results.csv")
    latest_runs

    # Only include gurobi for now
    latest_runs = latest_runs.filter(solver="gurobi")

    # keep only latest result
    latest_runs = latest_runs.sort("date").unique(
        subset=["problem", "library", "size"], keep="last", maintain_order=True
    )

    # remove errors
    latest_runs = latest_runs.filter(pl.col("error").is_null())

    # filter only relevant problem / sizes
    latest_runs = latest_runs.filter(
        pl.concat_str("problem", pl.col("size").cast(pl.Utf8), separator="_").is_in(
            [f"{problem}_{size}" for problem, size in BENCHMARK_PROBLEMS]
        )
    )

    # Convert to GB
    latest_runs = latest_runs.with_columns(
        (pl.col("max_memory_uss_mb") / 1024).alias("max_memory_uss_gb"),
        (pl.col("max_solver_memory_uss_mb") / 1024).alias("max_solver_memory_uss_gb"),
    )

    latest_runs
    return (latest_runs,)


@app.cell
def _(RESULTS_FOLDER, latest_runs, pl, warnings):
    MARKERS_TO_IGNORE = ["1_START", "3b_GUROBI_PRESOLVED", "6_DONE"]

    def extract_data(problem, date, library, size, solver, total_time, solve_time):
        df = pl.read_parquet(
            f"{RESULTS_FOLDER}/{problem}/mem_log/{date}_{library}_{solver}_{size}.parquet"
        )

        # events are only on main thread
        df = df.filter(process_name="main").drop("process_name")

        # keep only relevant columns
        df = df.select("time_s", "events")

        # drop uneventful rows and explode
        df = raw_df = df.filter(pl.col("events").list.len() > 0).explode("events")

        # only GUROBI_END should appear twice
        assert (
            df.filter(pl.col("events") != "4_GUROBI_END")
            .get_column("events")
            .is_unique()
            .all()
        )

        # keep latest value
        df = df.filter(pl.col("time_s") == pl.col("time_s").max().over("events"))

        # find appropriate total time
        done_time = df.filter(events="6_DONE").get_column("time_s").item()
        if done_time >= total_time:
            assert (done_time - total_time) / done_time < 0.01, (
                f"done_time ({done_time}) should be within 1% of total_time ({total_time})"
            )
            total_time = done_time

        # filter out presolve (not relevant)
        df = df.filter(~pl.col("events").is_in(MARKERS_TO_IGNORE))

        # add end time
        df = pl.concat(
            [df, pl.DataFrame({"time_s": [total_time], "events": ["7_ENDED"]})]
        )

        # sort and compute difference
        df = df.sort("events")
        df = df.select(
            description=pl.col("events").replace_strict(
                {
                    # "1_START": "startup",
                    "2_SOLVE": "build",
                    "3_GUROBI_START": "convert_to_gurobi",
                    "4_GUROBI_END": "solve",
                    "5_SOLVE_RETURNED": "convert_back",
                    # "6_DONE": "postprocess",
                    "7_ENDED": "postprocess",
                }
            ),
            elapsed=pl.col("time_s").diff().fill_null(pl.col("time_s")),
        )
        assert (df.get_column("elapsed") >= 0).all(), (
            f"elapsed time should be non-negative ({problem, library, size}): {df}"
        )

        # Check solve time
        solve_time_df = df.filter(description="solve")["elapsed"]
        if not solve_time_df.is_empty():
            if abs(solve_time_df.item() - solve_time) / total_time >= 0.05:
                warnings.warn(
                    f"solve time ({solve_time}) should match ({problem, library, size}): {raw_df}"
                )

        # Add metadata
        df = df.select(
            problem=pl.lit(problem),
            library=pl.lit(library),
            size=pl.lit(size),
            solver=pl.lit(solver),
            description=pl.col("description"),
            elapsed=pl.col("elapsed"),
        )

        return df

    data = [
        extract_data(*args)
        for args in latest_runs.select(
            "problem",
            "date",
            "library",
            "size",
            "solver",
            "total_time_s",
            "solve_time_s",
        ).iter_rows()
    ]
    # data = data[0:1]
    data = pl.concat(data, how="diagonal")

    data = data.join(
        latest_runs.filter(library="pyoframe").select(
            "problem", "size", "total_time_s", "num_variables"
        ),
        on=["problem", "size"],
    )

    data = data.with_columns(
        elapsed_normalized=pl.col("elapsed") / pl.col("total_time_s")
    ).drop("total_time_s")

    data
    return (data,)


@app.cell
def _(BENCHMARK_PROBLEMS, Patch, RESULTS_FOLDER, data, latest_runs, pl, plt):
    LIBRARY_LABELS = {
        "pyoframe": "Pyoframe",
        "pyoptinterface": "PyOptInterface",
        "gurobipy": "Gurobipy",
        "jump": "JuMP",
        "ampl": "AMPL",
        "pyomo": "Pyomo",
        "linopy": "Linopy",
        "cvxpy": "CVXPY",
        "pulp": "PuLP",
    }

    COLORS = {
        "build": "gray",
        "convert_to_gurobi": "lightgray",
        "solve": "white",
        "convert_back": "lightgray",
        "postprocess": "gray",
    }
    order = {description: i for i, description in enumerate(COLORS.keys())}

    plt.rcParams.update(
        {
            "font.size": 7,
            "font.sans-serif": ["Helvetica", "Nimbus Sans"],
            "axes.axisbelow": True,
            "axes.labelsize": 7,
            "xtick.labelsize": 5,
            # tick size
            "xtick.major.size": 2,
            # spline width
            "axes.linewidth": 0.5,
        }
    )

    fig, axes = plt.subplots(
        ncols=2,
        nrows=1,
        figsize=(7.086, 5),  # 180mm x 90mm
    )

    ax_time = axes[0]
    ax_memory = axes[1]

    EXTRA_SPACING = 4
    PROBLEM_SPACING = 6
    PADDING = 4
    HEIGHT = 4
    BAR_TEXT_OFFSET = 0.12

    bar_kwargs = dict(
        height=HEIGHT,
        edgecolor="black",
        zorder=2,
        linewidth=0.5,
    )

    def label_kwargs(lib):
        return dict(
            va="center",
            ha="left",
            fontsize=6,
            color="#DC2626" if lib == "pyoframe" else "black",
            fontweight="bold" if lib == "pyoframe" else "normal",
        )

    _y = PADDING
    for (problem, _), problem_label in reversed(BENCHMARK_PROBLEMS.items()):
        _y_start = _y
        df_problem = data.filter(problem=problem)

        # Plot time
        df_problem = df_problem.sort(
            pl.col("elapsed").sum().over("library"), descending=True
        )
        for i, ((library,), bar) in enumerate(
            df_problem.group_by("library", maintain_order=True)
        ):
            _y_time = _y + i * (PROBLEM_SPACING)
            bar = bar.sort(pl.col("description").replace_strict(order))
            _base = 0
            _total_time = bar.get_column("elapsed").sum()
            _total_time_normalized = bar.get_column("elapsed_normalized").sum()

            for description, elapsed in bar.select(
                "description",
                "elapsed_normalized",
            ).iter_rows():
                ax_time.barh(
                    width=elapsed,
                    y=_y_time,
                    left=_base,
                    color=COLORS[description],
                    **bar_kwargs,
                )
                _base += elapsed

            if library == "pyoframe":
                _total_time_str = (
                    f"{int(_total_time)}s"
                    if _total_time <= 120
                    else f"{(_total_time / 60):.1f}min"
                )
                _total_time_str = ", " + _total_time_str
            else:
                _total_time_str = ""
            ax_time.text(
                _total_time_normalized + BAR_TEXT_OFFSET,
                _y_time,
                LIBRARY_LABELS[library]
                + f" ({_total_time_normalized:.1f}x{_total_time_str})",
                **label_kwargs(library),
                zorder=3,
                # backgroundcolor="white",
                # remove padding around text
                # bbox=dict(facecolor="white", edgecolor="none", pad=0.0)
            )
            # _y += PROBLEM_SPACING

        # plot memory
        df_memory = latest_runs.filter(problem=problem)
        df_memory = df_memory.sort(pl.col("max_memory_uss_gb"), descending=True)
        pyoframe_max = (
            df_memory.filter(library="pyoframe").get_column("max_memory_uss_gb").item()
        )
        for i, ((library,), bar) in enumerate(
            df_memory.group_by("library", maintain_order=True)
        ):
            y_memory = _y + i * (PROBLEM_SPACING)
            total_memory = bar.get_column("max_memory_uss_gb").item()
            memory_normalized = total_memory / pyoframe_max
            solver_memory_normalized = (
                bar.get_column("max_solver_memory_uss_gb").fill_null(0).item()
                / pyoframe_max
            )
            build_memory = memory_normalized - solver_memory_normalized

            ax_memory.barh(
                width=solver_memory_normalized, y=y_memory, color="white", **bar_kwargs
            )
            ax_memory.barh(
                width=build_memory,
                y=y_memory,
                left=solver_memory_normalized,
                color=COLORS["build"],
                **bar_kwargs,
            )
            if library == "pyoframe":
                _total_mem_str = f"{int(total_memory)}GB"
                _total_mem_str = ", " + _total_mem_str
            else:
                _total_mem_str = ""
            ax_memory.text(
                memory_normalized + BAR_TEXT_OFFSET,
                y_memory,
                LIBRARY_LABELS[library]
                + f" ({memory_normalized:.1f}x{_total_mem_str})",
                **label_kwargs(library),
            )

        _y += PROBLEM_SPACING * (i + 1)
        num_variables_mil = round(
            df_problem.get_column("num_variables").mode().mean() / 1e6, 1
        )
        ax_time.text(
            -BAR_TEXT_OFFSET,
            (_y_start + _y - PROBLEM_SPACING) / 2,
            problem_label + "\n" + f"(n={num_variables_mil:g}M)",
            va="center",
            ha="right",
            fontsize=7,
        )
        _y += EXTRA_SPACING
    _y -= EXTRA_SPACING + PROBLEM_SPACING
    ax_time.set_xlabel("End-to-End Execution Time\n(relative to Pyoframe)")
    ax_memory.set_xlabel("Peak Memory Usage\n(relative to Pyoframe)")
    for ax in axes:
        ax.set_yticks([])
        ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_ylim(0, _y + PADDING)

    LEGEND_HANDLES_TIME = [
        (COLORS["solve"], "Gurobi"),
        (COLORS["convert_to_gurobi"], "Overhead (conversions)"),
        (COLORS["build"], "Overhead (modeling code)"),
    ]
    LEGEND_HANDLES_MEMORY = [
        (COLORS["solve"], "Gurobi"),
        (COLORS["build"], "Overhead (modeling framework)"),
    ]

    for ax, LEGEND_HANDLES, x_offset in zip(
        axes, [LEGEND_HANDLES_TIME, LEGEND_HANDLES_MEMORY], [1.15, 1.3]
    ):
        ax.legend(
            handles=[
                Patch(
                    facecolor=color, edgecolor="black", label=description, linewidth=0.5
                )
                for color, description in LEGEND_HANDLES
            ],
            loc="upper right",
            frameon=True,
            bbox_to_anchor=(x_offset, 1),
            fontsize=6,
        )

    fig.savefig(
        f"{RESULTS_FOLDER}/benchmark_results_plot.png", bbox_inches="tight", dpi=300
    )
    fig
    return


if __name__ == "__main__":
    app.run()
