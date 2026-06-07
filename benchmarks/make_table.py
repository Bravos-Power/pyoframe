"""Script to create benchmarking table from results CSV."""

import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    from math import log
    from pathlib import Path

    import great_tables as gt
    import matplotlib as mpl
    import polars as pl

    return Path, gt, log, mpl, pl


@app.cell
def _():
    RESULTS_FILE = "results/main/benchmark_results.csv"
    return (RESULTS_FILE,)


@app.cell
def _(RESULTS_FILE, pl):
    results_raw = pl.read_csv(RESULTS_FILE)
    results_raw
    return (results_raw,)


@app.cell
def _(pl, results_raw):
    results_latest = results_raw

    results_latest = results_latest.cast({"max_solver_memory_uss_mb": pl.Float64})

    # Only include gurobi for now
    results_latest = results_latest.filter(solver="gurobi").drop("solver")

    # keep only latest result
    results_latest = (
        results_latest.sort("date")
        .unique(subset=["problem", "library", "size"], keep="last", maintain_order=True)
        .drop("date")
    )

    # keep only timeout errors
    results_latest = results_latest.filter(
        (pl.col("error") == "TIMEOUT") | pl.col("error").is_null()
    )
    results_latest = results_latest.with_columns(
        total_time_s=pl.when(error="TIMEOUT").then(None).otherwise("total_time_s"),
        solve_time_s=pl.when(error="TIMEOUT").then(None).otherwise("solve_time_s"),
    )

    # convert to GiB
    results_latest = results_latest.with_columns(
        max_memory_uss_gib=pl.col("max_memory_uss_mb") / 1024,
        max_solver_memory_uss_gib=pl.col("max_solver_memory_uss_mb") / 1024,
    ).drop("max_memory_uss_mb", "max_solver_memory_uss_mb")

    # remove invalid solver_memory_uss_gib values
    results_latest = results_latest.with_columns(
        pl.col("max_solver_memory_uss_gib").replace(0, None)
    )

    results_latest
    return (results_latest,)


@app.cell
def _(pl, results_latest):
    results = results_latest

    # Replace zero solve_time with N/A
    results = results.with_columns(
        pl.when(pl.col("solve_time_s") == 0)
        .then(pl.lit(None))
        .otherwise(pl.col("solve_time_s"))
        .alias("solve_time_s")
    )

    # Determine median number of variables, and min solve time
    results = results.with_columns(
        pl.col("num_variables").median().over("problem", "size"),
        min_solve_time_s=pl.col("solve_time_s").min().over("problem", "size"),
        solver_benchmark_memory_gib=pl.col("max_solver_memory_uss_gib")
        .median()
        .over("problem", "size"),
    )

    # compute time and memory overhead
    results = results.with_columns(
        overhead_time_s=pl.when(pl.col("min_solve_time_s").is_not_null())
        .then(pl.col("total_time_s") - pl.col("min_solve_time_s"))
        .otherwise(pl.col("total_time_s")),
        overhead_memory_uss_gib=pl.when(
            pl.col("solver_benchmark_memory_gib").is_not_null()
        )
        .then(pl.col("max_memory_uss_gib") - pl.col("solver_benchmark_memory_gib"))
        .otherwise("max_memory_uss_gib"),
    )

    # compute overhead relative to solve time
    results = results.with_columns(
        overhead_time_relative_solve=(
            pl.col("total_time_s") / pl.col("min_solve_time_s")
        ),
        overhead_memory_relative_solve=(
            pl.col("max_memory_uss_gib") / pl.col("solver_benchmark_memory_gib")
        ),
    )

    # Drop problems with 10K or less variables
    # results = results.filter(pl.col("num_variables") > 10_000)

    # Only keep relevant columns for the table
    results = results.select(
        "problem",
        "library",
        "size",
        "overhead_time_s",
        "max_memory_uss_gib",
        "num_variables",
        "min_solve_time_s",
        "solver_benchmark_memory_gib",
        "overhead_memory_uss_gib",
        "overhead_time_relative_solve",
        "overhead_memory_relative_solve",
        "error",
    )

    # Merge pyoframe results to get relative overheads
    pyoframe_results = results.filter(library="pyoframe")
    results = results.join(
        pyoframe_results.select(
            "problem", "size", "overhead_time_s", "overhead_memory_uss_gib"
        ),
        on=["problem", "size"],
        how="left",
        suffix="_pyoframe",
    )
    results = results.with_columns(
        overhead_time_relative=pl.col("overhead_time_s")
        / pl.col("overhead_time_s_pyoframe"),
        memory_relative=pl.col("overhead_memory_uss_gib")
        / pl.col("overhead_memory_uss_gib_pyoframe"),
    )

    def round_two_sig_figs(val):
        if val >= 10:
            return f"{val:.0f}"
        if val >= 1:
            return f"{val:.1f}"
        return f"{val:.2f}"

    def human_format(num):
        for unit in ["", "k", "M", "B", "T"]:
            if abs(num) < 1000:
                return f"{num:.0f}{unit}"
            num /= 1000
        return f"{num:.0f}P"

    def format_time(val_s):
        if val_s < 1:
            return f"{val_s * 1000:.0f} ms"
        elif val_s < 60:
            return round_two_sig_figs(val_s) + " s"
        else:
            return round_two_sig_figs(val_s / 60) + " min"

    def format_memory(val_gib):
        if val_gib * 1024 < 1:
            return round_two_sig_figs(val_gib * 1024 * 1024) + " kB"
        elif val_gib < 1:
            return round_two_sig_figs(val_gib * 1024) + " MB"
        else:
            return round_two_sig_figs(val_gib) + " GB"

    # Round seconds to 1 decimal place
    results = results.with_columns(
        min_solve_time_s_pretty=pl.col("min_solve_time_s")
        .map_elements(format_time, pl.String)
        .fill_null("N/A**"),
        solver_benchmark_memory_gib_pretty=pl.col("solver_benchmark_memory_gib")
        .map_elements(format_memory, pl.String)
        .fill_null("N/A**"),
        time=pl.concat_str(
            pl.lit("<span style='font-weight: bold"),
            # pl.col("overhead_time_color"),
            pl.lit(";'>"),
            pl.col("overhead_time_relative").map_elements(
                round_two_sig_figs, pl.String
            ),
            pl.lit("x</span>"),
            pl.when(pl.col("overhead_time_relative_solve").is_not_null())
            .then(
                pl.concat_str(
                    pl.lit("<br/><span style='color: grey;'>("),
                    pl.col("overhead_time_relative_solve").map_elements(
                        round_two_sig_figs, pl.String
                    ),
                    pl.lit("x)</span>"),
                )
            )
            .otherwise(pl.lit("")),
        ),
        memory=pl.concat_str(
            pl.lit("<span style='font-weight: bold"),
            # pl.col("memory_color"),
            pl.lit(";'>"),
            pl.col("memory_relative").map_elements(round_two_sig_figs, pl.String),
            pl.lit("x</span>"),
            pl.when(pl.col("overhead_memory_relative_solve").is_not_null())
            .then(
                pl.concat_str(
                    pl.lit("<br/><span style='color: grey;'>("),
                    pl.col("overhead_memory_relative_solve").map_elements(
                        round_two_sig_figs, pl.String
                    ),
                    pl.lit("x)</span>"),
                )
            )
            .otherwise(pl.lit("")),
        ),
        size=pl.concat_str(
            pl.col("num_variables").map_elements(human_format, pl.String),
            pl.when(problem="simple_problem")
            .then(pl.lit(""))
            .otherwise(
                pl.concat_str(
                    pl.lit("<br/><span style='color: grey;'>(n="),
                    pl.col("size"),
                    pl.lit(")</span>"),
                )
            ),
        ),
    )

    # Handle timeout
    results = results.with_columns(
        time=pl.when(error="TIMEOUT").then(pl.lit("TO")).otherwise(pl.col("time")),
        memory=pl.when(error="TIMEOUT").then(pl.lit("TO")).otherwise(pl.col("memory")),
    )

    # Rename problems for better display
    results = results.with_columns(
        pl.col("library")
        .str.to_titlecase()
        .replace(
            {
                "Jump": "JuMP",
                "Ampl": "AMPL",
                "Pulp": "PuLP",
                "Pyoptinterface": "PyOptInterface",
                "Cvxpy": "CVXPY",
            }
        ),
        problem_name=pl.col("problem").replace_strict(
            {
                "simple_problem": "Trivial Problem (with data)",
                "energy_planning_capacity_expansion": "Electrical Grid Capacity Expansion Problem",
                "energy_planning_security_constrained_dispatch": "Electrical Grid Dispatch Problem",
                "facility_location": "Facility Location Problem (no data)",
            }
        ),
        problem_order=pl.col("problem").replace_strict(
            {
                "facility_location": 0,
                "simple_problem": 1,
                "energy_planning_capacity_expansion": 2,
                "energy_planning_security_constrained_dispatch": 3,
            }
        ),
        library_order=pl.col("library").replace_strict(
            {
                "pyoframe": 0,
                "pyoptinterface": 1,
                "gurobipy": 2,
                "jump": 3,
                "linopy": 4,
                "ampl": 5,
                "pyomo": 6,
                "cvxpy": 7,
                "pulp": 8,
            }
        ),
    )
    results = results.sort(["problem_order", "library_order", "num_variables"])

    results
    return results, round_two_sig_figs


@app.cell
def _(Path, RESULTS_FILE, gt, log, mpl, pl, results):
    results_table = results

    n_libraries = results["library"].n_unique()

    # Pivot
    results_table = results_table.select(
        "problem",
        "problem_name",
        "library",
        "size",
        "time",
        "memory",
        "min_solve_time_s_pretty",
        "solver_benchmark_memory_gib_pretty",
    )
    results_table = results_table.pivot(
        on="library",
        index=[
            "problem",
            "problem_name",
            "size",
            "min_solve_time_s_pretty",
            "solver_benchmark_memory_gib_pretty",
        ],
    ).fill_null("NI")

    # Reshuffle column order
    cols = (
        ["problem", "problem_name", "size", "min_solve_time_s_pretty"]
        + [c for c in results_table.columns if c.startswith("time_")]
        + ["solver_benchmark_memory_gib_pretty"]
        + [c for c in results_table.columns if c.startswith("memory")]
    )
    results_table = results_table.select(cols)

    # Add N/A for linopy
    results_table = results_table.with_columns(
        time_Linopy=pl.when(problem="facility_location")
        .then(pl.lit("NS"))
        .otherwise("time_Linopy"),
        memory_Linopy=pl.when(problem="facility_location")
        .then(pl.lit("NS"))
        .otherwise("memory_Linopy"),
    )

    _col_names = {c: c.split("_")[-1] for c in results_table.columns if c != "problem"}
    _col_names["min_solve_time_s_pretty"] = "Best Gurobi Solve Time"
    _col_names["solver_benchmark_memory_gib_pretty"] = "Gurobi Memory Usage*"

    table = (
        gt.GT(results_table.drop("problem"))
        .tab_stub(rowname_col="size", groupname_col="problem_name")
        .tab_stubhead(
            label=gt.html(
                "Number of variables<br/><span style='color: grey;'>(Problem size)</span>"
            )
        )
        .tab_spanner(
            gt.html(
                "<span style='font-weight: bold;'>Time overhead relative to Pyoframe</span><br><span style='color: grey;'>(Increase in solve time due to modeling interface)</span>"
            ),
            columns=[c for c in results_table.columns if c.startswith("time_")],
        )
        .tab_spanner(
            gt.html(
                "<span style='font-weight: bold;'>Memory overhead relative to Pyoframe</span><br><span style='color: grey;'>(Increase in peak memory usage due to modeling interface)</span>"
            ),
            columns=[c for c in results_table.columns if c.startswith("memory_")],
        )
        .cols_label(_col_names)
        .tab_style(
            style=gt.style.borders(sides=["left", "right"]),
            locations=gt.loc.body(columns=n_libraries + 3),
        )
        .tab_style(
            style=gt.style.borders(sides="left"),
            locations=gt.loc.body(columns=3),
        )
        .cols_label_rotate()
        .tab_options(row_striping_background_color="white", data_row_padding="0.5")
        .cols_align(
            align="right",
        )
        .tab_source_note(
            gt.html(
                """
                k = thousand; M = million; ms = milliseconds; s = seconds; min = minutes; kB = 1024 bytes; MB = 1,024² bytes; GB = 1,024³ bytes
                <br/>TO = Timeout (benchmark did not complete within the 20 minute time limit)
                <br/>NS = Not Supported (Linopy does not support quadratic constraints)
                <br/>NI = Not Implemented (CVXPY and PuLP were not implemented for all benchmarks to limit the benchmarking scope)
                <br/>* Gurobi's memory usage is estimated by taking the median memory usage across all benchmark runs.
                <br/>** The facility location benchmark developed by the JuMP and PyOptInterface authors does not involve solving the optimization problem.<br/>Only the time and memory needed to construct the problem is measured.
                """
            )
        )
    )

    vmin = 1 / 3
    vmax = 3
    color_norm = mpl.colors.Normalize(vmin=log(vmin), vmax=log(vmax), clip=True)
    anchors = [(vmin, "#A5D6A7"), (1, "white"), (vmax, "#EF9A9A")]
    color_map = mpl.colors.LinearSegmentedColormap.from_list(
        "green_red",
        [
            ((log(v) - log(vmin)) / (log(vmax) - log(vmin)), color)
            for v, color in anchors
        ],
    )
    for row_i, (problem, size) in enumerate(
        results.select("problem", "size").unique(maintain_order=True).iter_rows()
    ):
        for col_i, (library,) in enumerate(
            results.select("library").unique(maintain_order=True).iter_rows()
        ):
            color = results.filter(problem=problem, size=size, library=library)

            for metric, offset in [
                ("overhead_time_relative", 3),
                ("memory_relative", 3 + n_libraries + 1),
            ]:
                if color.is_empty():
                    color_hex = "white"
                elif color[metric].item() is None:
                    color_hex = "lightgrey"
                else:
                    amount = color.select(metric).item()

                    color_hex = mpl.colors.to_hex(color_map(color_norm(log(amount))))

                table = table.tab_style(
                    style=gt.style.fill(color_hex),
                    locations=gt.loc.body(
                        rows=row_i,
                        columns=col_i + offset,
                    ),
                )

    table.save(
        Path(RESULTS_FILE).parent / "benchmark_results_table.png",
        web_driver="edge",
        scale=2,
    )
    table.write_raw_html(Path(RESULTS_FILE).parent / "benchmark_results_table.html")
    table
    return (n_libraries,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw time and memory values
    """)
    return


@app.cell
def _(
    Path,
    RESULTS_FILE,
    format_numeric,
    gt,
    n_libraries,
    pl,
    results,
    round_two_sig_figs,
):
    _results_table = results

    # Pivot
    _results_table = _results_table.select(
        "problem",
        "problem_name",
        "library",
        "error",
        pl.col("size").str.replace("<br/>", " ", literal=True),
        format_numeric("min_solve_time_s", fill_null="N/A*"),
        format_numeric("solver_benchmark_memory_gib", fill_null="N/A*"),
        time=pl.col("overhead_time_s").map_elements(round_two_sig_figs, pl.String),
        memory=pl.col("overhead_memory_uss_gib").map_elements(
            round_two_sig_figs, pl.String
        ),
    )

    _results_table = _results_table.with_columns(
        time=pl.when(error="TIMEOUT").then(pl.lit("TO")).otherwise(pl.col("time")),
        memory=pl.when(error="TIMEOUT").then(pl.lit("TO")).otherwise(pl.col("memory")),
    ).drop("error")

    _results_table = _results_table.pivot(
        on="library",
        index=[
            "problem",
            "problem_name",
            "size",
            "min_solve_time_s",
            "solver_benchmark_memory_gib",
        ],
    ).fill_null("NI")

    # Reshuffle column order
    _cols = (
        ["problem", "problem_name", "size", "min_solve_time_s"]
        + [c for c in _results_table.columns if c.startswith("time_")]
        + ["solver_benchmark_memory_gib"]
        + [c for c in _results_table.columns if c.startswith("memory")]
    )
    _results_table = _results_table.select(_cols)

    # Add N/A for linopy
    _results_table = _results_table.with_columns(
        time_Linopy=pl.when(problem="facility_location")
        .then(pl.lit("NS"))
        .otherwise("time_Linopy"),
        memory_Linopy=pl.when(problem="facility_location")
        .then(pl.lit("NS"))
        .otherwise("memory_Linopy"),
    )

    _col_names = {c: c.split("_")[-1] for c in _results_table.columns if c != "problem"}
    _col_names["min_solve_time_s"] = "Gurobi Solve Time (s)"
    _col_names["solver_benchmark_memory_gib"] = "Gurobi Memory Usage (GB)"
    # _col_names = {}

    _table = (
        gt.GT(_results_table.drop("problem"))
        .tab_stub(rowname_col="size", groupname_col="problem_name")
        .tab_stubhead(
            label=gt.html(
                "Number of variables<br/><span style='color: grey;'>(Problem size)</span>"
            )
        )
        .tab_spanner(
            gt.html("<span style='font-weight: bold;'>Time overhead (s)</span>"),
            columns=[c for c in _results_table.columns if c.startswith("time_")],
        )
        .tab_spanner(
            gt.html("<span style='font-weight: bold;'>Memory overhead (GB)</span>"),
            columns=[c for c in _results_table.columns if c.startswith("memory_")],
        )
        .cols_label(_col_names)
        .tab_style(
            style=gt.style.borders(sides=["left", "right"]),
            locations=gt.loc.body(columns=n_libraries + 3),
        )
        .tab_style(
            style=gt.style.borders(sides="left"),
            locations=gt.loc.body(columns=3),
        )
        .cols_label_rotate()
        .tab_options(row_striping_background_color="white", data_row_padding="0.5")
        .cols_align(
            align="right",
        )
        .tab_source_note(
            gt.html(
                """
                k = thousand; M = million; s = seconds; GB = 1,024³ bytes
                <br/>TO = Timeout (benchmark did not complete within the 20 minute time limit)
                <br/>NS = Not Supported (Linopy does not support quadratic constraints)
                <br/>NI = Not Implemented (CVXPY and PuLP were not implemented for all benchmarks to limit the benchmarking scope)
                <br/>* The facility location benchmark developed by the JuMP and PyOptInterface authors does not involve solving the optimization problem.<br/>Only the time and memory needed to construct the problem is measured.
                """
            )
        )
    )

    _table.save(
        Path(RESULTS_FILE).parent / "benchmark_results_table_raw.png",
        web_driver="edge",
        scale=2,
    )
    _table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solve time and memory usage for different runs
    """)
    return


@app.cell
def _(Path, RESULTS_FILE, pl, results_latest):
    import altair as alt

    df_solver = results_latest

    df_solver.select(
        "problem", "library", "size", "solve_time_s", "max_solver_memory_uss_gib"
    )

    # compute relative difference of solve_time_s relative to median
    df_solver = df_solver.with_columns(
        relative_solve_time_s=pl.col("solve_time_s")
        / pl.col("solve_time_s").median().over("problem", "size"),
        relative_memory_gib=pl.col("max_solver_memory_uss_gib")
        / pl.col("max_solver_memory_uss_gib").median().over("problem", "size"),
    )

    df_solver = df_solver.unpivot(
        ["relative_solve_time_s", "relative_memory_gib"],
        index=["problem", "library", "size"],
        variable_name="metric",
        value_name="relative_value",
    )

    df_solver = df_solver.with_columns(
        pl.col("metric").replace(
            {
                "relative_solve_time_s": "Gurobi Solve Time",
                "relative_memory_gib": "Gurobi Memory Usage",
            }
        ),
        pl.col("problem").replace(
            {
                "energy_planning_capacity_expansion": "Electrical Grid Capacity Expansion Problem",
                "energy_planning_security_constrained_dispatch": "Electrical Grid Dispatch Problem",
                "simple_problem": "Trivial Problem",
            }
        ),
    )

    _fig = df_solver.plot.scatter(
        x=alt.X(
            "relative_value:Q",
            scale=alt.Scale(type="log"),
            title="Normalized value",
        ),
        y=alt.Y("library:O", title=""),
        column=alt.Column("metric", title=""),
        color=alt.Color(
            "problem:N", title="Benchmark", legend=alt.Legend(labelLimit=200)
        ),
    )

    _fig.save(Path(RESULTS_FILE).parent / "solver_performance.pdf")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Utils
    """)
    return


@app.cell
def _(pl, round_two_sig_figs):
    def format_numeric(col_name, sig_figs=2, fill_null=None):
        assert sig_figs == 2, "Only 2 significant figures is currently supported"
        return (
            pl.col(col_name)
            .map_elements(round_two_sig_figs, pl.String)
            .fill_null(fill_null)
        )

    return (format_numeric,)


if __name__ == "__main__":
    app.run()
