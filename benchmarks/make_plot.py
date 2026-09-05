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

    return Affine2D, Patch, Path, pl, plt


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
def _(BENCHMARK_PROBLEMS, RESULTS_FOLDER, pl):
    data = pl.read_csv(f"{RESULTS_FOLDER}/benchmark_results_processed.csv")

    # Non-errors and gurobi
    data = data.filter(pl.col("error").is_null()).drop("error")

    # filter only relevant problem / sizes
    data = data.filter(
        pl.concat_str("problem", "size", separator="_").is_in(
            [f"{problem}_{size}" for problem, size in BENCHMARK_PROBLEMS]
        )
    )

    data = data.with_columns(
        time_model=pl.col("time_overhead") - pl.col("time_convert"),
        memory_model=pl.col("memory_overhead"),
    )

    data = data.unpivot(
        index=["problem", "size", "library", "seed", "num_variables"],
        on=[
            "time_solver",
            "memory_solver",
            "time_model",
            "memory_model",
            "time_convert",
        ],
    )
    data = data.filter(pl.col("value").is_not_null())
    data = data.with_columns(
        pl.col("variable").str.split("_").list.get(0).alias("metric"),
        pl.col("variable").str.split("_").list.get(1).alias("type"),
    ).drop("variable")

    data_solver = (
        data.filter(type="solver")
        .group_by(["problem", "size", "metric", "type"])
        .agg(
            library=pl.lit("gurobi"),
            value=pl.col("value").mean(),
            value_min=pl.col("value").min(),
            value_max=pl.col("value").max(),
        )
    )

    data_overhead = (
        data.filter(pl.col("type") != "solver")
        .group_by(["problem", "size", "metric", "library", "type"])
        .agg(value=pl.col("value").mean())
    )

    data_overhead_error = (
        data.filter(pl.col("type") != "solver")
        .group_by(["problem", "size", "metric", "library", "seed"])
        .agg(pl.col("value").sum())
        .group_by(["problem", "size", "metric", "library"])
        .agg(
            value_min=pl.col("value").min(),
            value_max=pl.col("value").max(),
        )
    )
    data_overhead = data_overhead.join(
        data_overhead_error,
        on=["problem", "size", "metric", "library"],
        how="left",
        validate="m:1",
    )

    data = pl.concat([data_solver, data_overhead], how="diagonal").join(
        data.select("problem", "size", "num_variables").unique(),
        on=["problem", "size"],
        how="left",
        validate="m:1",
    )

    # Normalize to Pyoframe total time
    data = data.join(
        data_overhead.filter(library="pyoframe")
        .group_by("problem", "size", "metric")
        .agg(pl.col("value").sum()),
        on=["problem", "size", "metric"],
        how="left",
        validate="m:1",
        suffix="_pyoframe",
    )
    data = data.with_columns(
        (
            pl.col("value", "value_min", "value_max") / pl.col("value_pyoframe")
        ).name.suffix("_normalized")
    ).drop("value_pyoframe")

    data = data.sort("problem", "size", "metric", "library", "type")
    data.write_csv(RESULTS_FOLDER / "plot_data.csv")
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
        "solver": "white",
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
                pl.col("type") == "solver",
                pl.col("value").sum().over("library"),
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
                _total_amount = bar.get_column("value").sum()
                _total_amount_normalized = bar.get_column("value_normalized").sum()
                amount_min = bar.get_column("value_min_normalized").unique().item()
                amount_max = bar.get_column("value_max_normalized").unique().item()
                _x_label = amount_max

                for description, amount in bar.select(
                    "type", "value_normalized"
                ).iter_rows():
                    ax.barh(
                        width=amount,
                        y=_y_bar,
                        left=_base,
                        color=COLORS[description],
                        **bar_kwargs,
                    )
                    _base += amount

                if amount_min / _base != _base or amount_max != _base:
                    ax.errorbar(
                        x=_base,
                        y=_y_bar,
                        xerr=[[_base - amount_min], [amount_max - _base]],
                        fmt="none",
                        ecolor="black",
                        elinewidth=0.5,
                        capsize=1,
                        capthick=0.5,
                    )

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
        # ax.set_title(TITLES[column], fontsize=TITLE_FONTSIZE, pad=TITLE_PADDING)
        ax.set_yticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0, _y)

    LEGEND_HANDLES_TIME = [
        (COLORS["solver"], "Solver (for reference)"),
        (COLORS["model"], "Overhead (model code)"),
        (COLORS["convert"], "Overhead (conversions)"),
    ]
    LEGEND_HANDLES_MEMORY = [
        (COLORS["solver"], "Solver (for reference)"),
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
