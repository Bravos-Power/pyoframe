"""Script to plot the benchmark results for a select group of benchmark problems."""

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import polars as pl
    from matplotlib.patches import Patch
    from matplotlib.transforms import Affine2D
    from scipy.stats import t

    return Affine2D, Patch, Path, pl, plt, t


@app.cell
def _():
    VERSION = 7
    CONFIDENCE_INTERVAL = 0.95
    return CONFIDENCE_INTERVAL, VERSION


@app.cell
def _(Path):
    RESULTS_FOLDER = Path(__file__).parent / "results/main"
    BENCHMARK_PROBLEMS = {
        ("simple_problem", 10_000_000): "Trivial\nData\nProblem",
        ("energy_planning_capacity_expansion", 168): "Capacity\nExpansion\nProblem",
        (
            "energy_planning_security_constrained_dispatch",
            24,
        ): "Electricity\nDispatch\nProblem",
        ("facility_location", 128): "Facility\nLocation\nProblem*",
    }
    return BENCHMARK_PROBLEMS, RESULTS_FOLDER


@app.cell
def _(BENCHMARK_PROBLEMS, RESULTS_FOLDER, VERSION, pl):
    latest_runs = pl.read_csv(f"{RESULTS_FOLDER}/benchmark_results.csv")
    latest_runs = latest_runs.cast(
        {
            "version": pl.Int32,
            "convert_from_solver_s": pl.Float64,
            "convert_to_solver_s": pl.Float64,
            "presolve_time_s": pl.Float64,
        }
    )
    latest_runs = latest_runs.drop(
        "num_constraints", "num_nonzeros", "barrier_iterations", "objective_value"
    )

    if VERSION is not None:
        latest_runs = latest_runs.filter(version=VERSION)
    latest_runs = latest_runs.drop("version")

    # Non-errors and gurobi
    latest_runs = latest_runs.filter(pl.col("error").is_null(), solver="gurobi").drop(
        "error", "solver"
    )

    # keep only latest result
    latest_runs = (
        latest_runs.sort("date")
        .unique(
            subset=["problem", "library", "size", "seed"],
            keep="last",
            maintain_order=True,
        )
        .drop("date")
    )

    assert latest_runs["solver_version"].n_unique() == 1, (
        "Multiple Gurobi versions found"
    )

    # filter only relevant problem / sizes
    latest_runs = latest_runs.filter(
        pl.concat_str("problem", pl.col("size").cast(pl.Utf8), separator="_").is_in(
            [f"{problem}_{size}" for problem, size in BENCHMARK_PROBLEMS]
        )
    )

    # Convert to GB
    latest_runs = latest_runs.with_columns(
        (pl.col("max_memory_uss_mb") / 1024).alias("max_memory"),
        (pl.col("max_solver_memory_uss_mb") / 1024).alias("memory_solve"),
    ).drop("max_memory_uss_mb", "max_solver_memory_uss_mb")

    # Remove solve_time for facility location
    latest_runs = latest_runs.with_columns(
        solve_time_s=pl.when(problem="facility_location")
        .then(None)
        .otherwise(pl.col("solve_time_s"))
    )

    # Compute memory overhead
    latest_runs = latest_runs.with_columns(
        memory_model=pl.col("max_memory") - pl.col("memory_solve").fill_null(0)
    )

    # Compute time overhead
    latest_runs = latest_runs.rename({"solve_time_s": "time_solve"})
    latest_runs = latest_runs.with_columns(
        time_convert=pl.col("convert_from_solver_s").fill_null(0)
        + pl.col("convert_to_solver_s").fill_null(0),
    ).with_columns(
        time_model=pl.col("total_time_s")
        - pl.col("time_convert")
        - pl.col("time_solve").fill_null(0)
    )

    # Correct for differences in presolve_time_s between libraries
    presolve_attributable_to_lib = pl.col("presolve_time_s") - pl.col(
        "presolve_time_s"
    ).mean().over("problem", "size")
    latest_runs = latest_runs.with_columns(
        pl.col("time_convert") + presolve_attributable_to_lib.fill_null(0),
        pl.col("time_solve") - presolve_attributable_to_lib.fill_null(0),
        presolve_correction=presolve_attributable_to_lib,
    )

    latest_runs
    return (latest_runs,)


@app.cell
def _(CONFIDENCE_INTERVAL, latest_runs, pl, t):
    data = latest_runs.select(
        "problem",
        "library",
        "size",
        "seed",
        "time_solve",
        "time_convert",
        "time_model",
        "memory_model",
        "memory_solve",
    )
    data = data.unpivot(
        index=["problem", "size", "library", "seed"], value_name="amount"
    )
    data = data.filter(pl.col("amount").is_not_null())
    data = data.with_columns(
        pl.col("variable").str.split("_").list.get(0).alias("metric"),
        pl.col("variable").str.split("_").list.get(1).alias("type"),
    ).drop("variable")

    data_solver = (
        data.filter(type="solve")
        .group_by(["problem", "size", "metric", "type"])
        .agg(
            library=pl.lit("gurobi"),
            amount=pl.col("amount").mean(),
            amount_std=pl.col("amount").std(),
            n=pl.len(),
        )
    )

    data_overhead = (
        data.filter(pl.col("type") != "solve")
        .group_by(["problem", "size", "metric", "type", "library"])
        .agg(pl.col("amount").mean())
    )

    data_err = (
        data.filter(pl.col("type") != "solve")
        .group_by(["problem", "size", "metric", "library", "seed"])
        .agg(pl.col("amount").sum())
        .group_by(["problem", "size", "metric", "library"])
        .agg(amount_std=pl.col("amount").std(), n=pl.len())
    )

    data_overhead = data_overhead.join(
        data_err,
        on=["problem", "size", "metric", "library"],
        how="left",
        validate="m:1",
    )

    data = pl.concat([data_solver, data_overhead], how="diagonal")

    def t_stat(n):
        return t.ppf((1 + CONFIDENCE_INTERVAL) / 2, n - 1)

    data = data.with_columns(
        amount_ci=pl.when(pl.col("n") >= 3).then(
            pl.col("n").map_elements(t_stat, pl.Float64)
            * (pl.col("amount_std") / pl.col("n").sqrt())
        )
    ).drop("amount_std", "n")

    # Normalize to Pyoframe total time
    data = data.join(
        data_overhead.filter(library="pyoframe")
        .group_by("problem", "size", "metric")
        .agg(pl.col("amount").sum()),
        on=["problem", "size", "metric"],
        how="left",
        validate="m:1",
        suffix="_pyoframe",
    )
    data = data.with_columns(
        (pl.col("amount", "amount_ci") / pl.col("amount_pyoframe")).name.suffix(
            "_normalized"
        )
    ).drop("amount_pyoframe")

    # Join num_variables
    data = data.join(
        latest_runs.group_by("problem", "size").agg(pl.col("num_variables").median()),
        on=["problem", "size"],
        how="left",
        validate="m:1",
    )

    data
    return (data,)


@app.cell
def _(Affine2D, BENCHMARK_PROBLEMS, Patch, RESULTS_FOLDER, data, pl, plt):
    LIBRARY_LABELS = {
        "gurobi": "Gurobi",
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
        "solve": "white",
        "model": "gray",
        "convert": "lightgray",
    }
    # TITLES = {"time": "Time", "memory": "Peak Memory Usage"}
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
            # tick width
            "xtick.major.width": 0.5,
        }
    )

    PROBLEM_SPACING = 0.22
    BAR_HEIGHT = 0.15
    BAR_SPACING = 0.05
    SOLVER_EXTRA_SPACING = 0.04
    PADDING = 0.1

    # TITLE_PADDING = 5
    WSPACE = 0.38

    BAR_TEXT_OFFSET = 8
    Y_LABEL_OFFSET = 0.3
    Y_AXIS_EXTENSION = 0.07

    AXIS_LABEL_FONT_SIZE = 7
    # TITLE_FONTSIZE = 7

    NUM_PROBLEMS = data.unique("problem").height
    NUM_BARS = data.unique(["problem", "library"]).height

    expected_height = (
        BAR_HEIGHT * (NUM_BARS - NUM_PROBLEMS)
        + SOLVER_EXTRA_SPACING * NUM_PROBLEMS
        + PROBLEM_SPACING * (NUM_PROBLEMS - 1)
        + 2 * PADDING
    )

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(7.086, expected_height),  # 180mm x 90mm
    )

    # set width space between subplots
    fig.subplots_adjust(wspace=WSPACE)

    bar_kwargs = dict(
        height=BAR_HEIGHT - BAR_SPACING, edgecolor="black", zorder=2, linewidth=0.5
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
        df_row = data.filter(problem=problem)

        if df_row.is_empty():
            continue

        num_vars = df_row.get_column("num_variables").unique().item()
        num_vars_str = f"{num_vars / 1e6:.1f}M"

        for column, ax in zip(["time", "memory"], axes):
            df_panel = df_row.filter(metric=column)

            df_panel = df_panel.sort(
                pl.col("type") == "solve",
                pl.col("amount").sum().over("library"),
                descending=[False, True],
            )
            for i, ((library,), bar) in enumerate(
                df_panel.group_by("library", maintain_order=True)
            ):
                _y_bar = (
                    _y
                    + i * (BAR_HEIGHT)
                    + (SOLVER_EXTRA_SPACING if library == "gurobi" else 0)
                )
                bar = bar.sort(pl.col("type").replace_strict(order))
                _base = 0
                _total_amount = bar.get_column("amount").sum()
                _total_amount_normalized = bar.get_column("amount_normalized").sum()
                _x_label = _total_amount_normalized
                ci = bar.get_column("amount_ci_normalized").unique().item()

                for description, amount, ci in bar.select(
                    "type", "amount_normalized", "amount_ci_normalized"
                ).iter_rows():
                    ax.barh(
                        width=amount,
                        y=_y_bar,
                        left=_base,
                        color=COLORS[description],
                        **bar_kwargs,
                    )
                    _base += amount

                if ci is not None:
                    ax.errorbar(
                        x=_base,
                        y=_y_bar,
                        xerr=ci,
                        fmt="none",
                        ecolor="black",
                        elinewidth=0.5,
                        capsize=1,
                        capthick=0.5,
                    )
                    _x_label += ci

                if library in ("pyoframe", "gurobi"):
                    if column == "time":
                        _total_amount_str = (
                            f"{int(_total_amount)}s"
                            if _total_amount <= 120
                            else f"{(_total_amount / 60):.1f}min"
                        )
                    else:
                        _total_amount_str = f"{int(_total_amount)}GB"
                    _total_amount_str = ", " + _total_amount_str
                else:
                    _total_amount_str = ""
                ax.text(
                    _x_label,
                    _y_bar,
                    LIBRARY_LABELS[library]
                    + f" ({_total_amount_normalized:.1f}x{_total_amount_str})",
                    **label_kwargs(library),
                    zorder=3,
                    transform=ax.transData + Affine2D().translate(BAR_TEXT_OFFSET, 0),
                    # backgroundcolor="white",
                    # remove padding around text
                    # bbox=dict(facecolor="white", edgecolor="none", pad=0.0)
                )

        _y += BAR_HEIGHT * i + SOLVER_EXTRA_SPACING
        axes[0].text(
            -Y_LABEL_OFFSET,
            (_y_start + _y - BAR_HEIGHT) / 2,
            problem_label + "\n" + f"(n={num_vars_str})",
            va="center",
            ha="right",
            fontsize=AXIS_LABEL_FONT_SIZE,
        )
        for ax in axes:
            ax.plot(
                [0, 0],
                [_y_start - Y_AXIS_EXTENSION, _y + Y_AXIS_EXTENSION],
                linewidth=1,
                color="black",
            )
        _y += PROBLEM_SPACING
    _y -= PROBLEM_SPACING
    _y += PADDING
    assert (_y - expected_height) / expected_height < 0.001, (
        f"Estimated height {expected_height} does not match actual height {_y}"
    )
    axes[0].set_xlabel("Time\n(relative to Pyoframe)", fontsize=AXIS_LABEL_FONT_SIZE)
    axes[1].set_xlabel(
        "Memory usage\n(relative to Pyoframe)", fontsize=AXIS_LABEL_FONT_SIZE
    )
    for ax, column in zip(axes, ["time", "memory"]):
        ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1))
        # ax.set_title(TITLES[column], fontsize=TITLE_FONTSIZE, pad=TITLE_PADDING)
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, _y)

    LEGEND_HANDLES_TIME = [
        (COLORS["solve"], "Solver (for reference)"),
        (COLORS["model"], "Overhead (model code)"),
        (COLORS["convert"], "Overhead (conversions)"),
    ]
    LEGEND_HANDLES_MEMORY = [
        (COLORS["solve"], "Solver (for reference)"),
        (COLORS["model"], "Overhead"),
    ]

    for ax, LEGEND_HANDLES, x_offset, y_offset in zip(
        axes, [LEGEND_HANDLES_TIME, LEGEND_HANDLES_MEMORY], [1.3, 1.3], [1, 1]
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
            bbox_to_anchor=(x_offset, y_offset),
            fontsize=6,
        )

    fig.savefig(
        f"{RESULTS_FOLDER}/benchmark_results_plot.png", bbox_inches="tight", dpi=300
    )
    fig
    return


if __name__ == "__main__":
    app.run()
