"""Script to plot the benchmark results for a select group of benchmark problems."""

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import polars as pl

    return pl, plt


@app.cell
def _():
    RESULTS_FOLDER = "results/main"
    BENCHMARK_PROBLEMS = {
        ("facility_location", 128): "Facility\nLocation\nProblem*",
        ("simple_problem", 10_000_000): "Trivial\nData\nProblem",
        ("energy_planning_capacity_expansion", 168): "Capacity\nExpansion\nProblem",
        (
            "energy_planning_security_constrained_dispatch",
            48,
        ): "Electricity\nDispatch\nProblem",
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

    # TODO REMOVE
    # ignore AMPL for now
    # latest_runs = latest_runs.filter(pl.col("library") != "ampl")

    latest_runs
    return (latest_runs,)


@app.cell
def _(RESULTS_FOLDER, latest_runs, pl):
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
            assert abs(solve_time_df.item() - solve_time) / total_time < 0.05, (
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
def _(BENCHMARK_PROBLEMS, RESULTS_FOLDER, data, pl, plt):
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
    # ax_memory = axes[1]

    EXTRA_SPACING = 4
    PROBLEM_SPACING = 6
    PADDING = 4
    HEIGHT = 4
    BAR_TEXT_OFFSET = 0.12

    _y = PADDING
    for (problem, _), problem_label in reversed(BENCHMARK_PROBLEMS.items()):
        _y_start = _y
        df_problem = data.filter(problem=problem)
        df_problem = df_problem.sort(
            pl.col("elapsed").sum().over("library"), descending=True
        )
        for (library,), bar in df_problem.group_by("library", maintain_order=True):
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
                    y=_y,
                    height=HEIGHT,
                    left=_base,
                    label=library,
                    color=COLORS[description],
                    edgecolor="black",
                    zorder=2,
                    linewidth=0.5,
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
                _y,
                LIBRARY_LABELS[library]
                + f" ({_total_time_normalized:.1f}x{_total_time_str})",
                va="center",
                ha="left",
                fontsize=6,
                color="#DC2626" if library == "pyoframe" else "black",
                fontweight="bold" if library == "pyoframe" else "normal",
                zorder=3,
                # backgroundcolor="white",
                # remove padding around text
                # bbox=dict(facecolor="white", edgecolor="none", pad=0.0)
            )
            _y += PROBLEM_SPACING
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
    ax_time.set_ylim(0, _y + PADDING)
    ax_time.set_xlabel("End-to-End Execution Time\n(normalized to Pyoframe)")

    for ax in axes:
        # remove y labels
        ax.set_yticks([])
        # make x axis gap size 1
        ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 1))
        # remove right and top spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # add vertical gridlines
        # ax.grid(axis="x", color="gray", linewidth=0.5, alpha=0.5, zorder=0)

    fig.savefig(
        f"{RESULTS_FOLDER}/benchmark_results_plot.png", bbox_inches="tight", dpi=300
    )
    fig
    return


if __name__ == "__main__":
    app.run()
