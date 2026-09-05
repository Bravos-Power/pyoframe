"""Script to create benchmarking table from results CSV."""

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    from math import log
    from pathlib import Path

    import great_tables as gt
    import marimo as mo
    import matplotlib as mpl
    import polars as pl

    return Path, gt, log, mo, mpl, pl


@app.cell
def _(Path):
    RESULTS_FOLDER = Path(__file__).parent / "results/main"
    return (RESULTS_FOLDER,)


@app.cell
def _(RESULTS_FOLDER, pl):
    results_raw = pl.read_csv(RESULTS_FOLDER / "benchmark_results_processed.csv")
    results_raw
    return (results_raw,)


@app.cell
def _(pl, results_raw):
    results = results_raw

    # Determine median solve time
    results = results.with_columns(
        time_solver_median=pl.col("time_solver").median().over("problem", "size")
    )

    # average across seeds
    results = results.group_by("problem", "library", "size").agg(
        pl.col("time_overhead", "memory_overhead").mean(),
        pl.col("num_variables", "time_solver_median", "memory_solver_median")
        .unique()
        .item(),
        pl.col("error").drop_nulls().unique().item(allow_empty=True),
    )

    # compute overhead relative to solve time
    results = results.with_columns(
        overhead_time_relative_solve=(
            pl.col("time_overhead") / pl.col("time_solver_median") + 1
        ),
        memory_overhead_relative_solve=(
            pl.col("memory_overhead") / pl.col("memory_solver_median") + 1
        ),
    )

    # Drop problems with 10K or less variables
    # results = results.filter(pl.col("num_variables") > 10_000)

    # Only keep relevant columns for the table
    results = results.select(
        "problem",
        "library",
        "size",
        "time_overhead",
        "num_variables",
        "time_solver_median",
        "memory_solver_median",
        "memory_overhead",
        "overhead_time_relative_solve",
        "memory_overhead_relative_solve",
        "error",
    )

    # Merge pyoframe results to get relative overheads
    pyoframe_results = results.filter(library="pyoframe")
    results = results.join(
        pyoframe_results.select("problem", "size", "time_overhead", "memory_overhead"),
        on=["problem", "size"],
        how="left",
        suffix="_pyoframe",
    )
    results = results.with_columns(
        overhead_time_relative=pl.col("time_overhead")
        / pl.col("time_overhead_pyoframe"),
        memory_relative=pl.col("memory_overhead") / pl.col("memory_overhead_pyoframe"),
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
        time_solver_median_pretty=pl.col("time_solver_median")
        .map_elements(format_time, pl.String)
        .fill_null("N/A*"),
        memory_solver_median_pretty=pl.col("memory_solver_median")
        .map_elements(format_memory, pl.String)
        .fill_null("N/A*"),
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
            pl.when(pl.col("memory_overhead_relative_solve").is_not_null())
            .then(
                pl.concat_str(
                    pl.lit("<br/><span style='color: grey;'>("),
                    pl.col("memory_overhead_relative_solve").map_elements(
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
                "simple_problem": "Trivial Data Problem",
                "energy_planning_capacity_expansion": "Electrical Grid Capacity Expansion Problem",
                "energy_planning_security_constrained_dispatch": "Electrical Grid Dispatch Problem",
                "facility_location": "Facility Location Problem (no data, from JuMP paper)",
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
                "ampl": 4,
                "pyomo": 5,
                "linopy": 6,
                "cvxpy": 7,
                "pulp": 8,
            }
        ),
    )
    results = results.sort(["problem_order", "library_order", "num_variables"])

    results
    return results, round_two_sig_figs


@app.cell
def _(RESULTS_FOLDER, gt, log, mpl, pl, results):
    results_table = results

    vmin, vmax = 1 / 3, 3
    color_min, color_max = "#A5D6A7", "#EF9A9A"
    legend_block_size = "20px"

    n_libraries = results["library"].n_unique()

    # Pivot
    results_table = results_table.select(
        "problem",
        "problem_name",
        "library",
        "size",
        "time",
        "memory",
        "time_solver_median_pretty",
        "memory_solver_median_pretty",
    )
    results_table = results_table.pivot(
        on="library",
        index=[
            "problem",
            "problem_name",
            "size",
            "time_solver_median_pretty",
            "memory_solver_median_pretty",
        ],
    ).fill_null("NI")

    # Reshuffle column order
    cols = (
        ["problem", "problem_name", "size", "time_solver_median_pretty"]
        + [
            c
            for c in results_table.columns
            if c.startswith("time")
            if c != "time_solver_median_pretty"
        ]
        + ["memory_solver_median_pretty"]
        + [
            c
            for c in results_table.columns
            if c.startswith("memory")
            if c != "memory_solver_median_pretty"
        ]
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
    _col_names["time_solver_median_pretty"] = "Gurobi Solve Time"
    _col_names["memory_solver_median_pretty"] = "Gurobi Memory Usage"

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
            columns=[
                c
                for c in results_table.columns
                if c.startswith("time_") and c != "time_solver_median_pretty"
            ],
        )
        .tab_spanner(
            gt.html(
                "<span style='font-weight: bold;'>Memory overhead relative to Pyoframe</span><br><span style='color: grey;'>(Increase in peak memory usage due to modeling interface)</span>"
            ),
            columns=[
                c
                for c in results_table.columns
                if c.startswith("memory_") and c != "memory_solver_median_pretty"
            ],
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
                f"""
            <span style="font-size: 18px;">
            <span style="
                display:inline-block;
                width:{legend_block_size};
                height:{legend_block_size};
                background:{color_min};
                border:1px solid #aaa;
                vertical-align:middle;
            "></span>
            &nbsp;Less than a third of Pyoframe's overhead (≤ 1/3×)
            &nbsp;&nbsp;&nbsp;

            <span style="
                display:inline-block;
                width:{legend_block_size};
                height:{legend_block_size};
                background:white;
                border:1px solid #aaa;
                vertical-align:middle;
            "></span>
            &nbsp;Same overhead as Pyoframe (1x)
            &nbsp;&nbsp;&nbsp;

            <span style="
                display:inline-block;
                width:{legend_block_size};
                height:{legend_block_size};
                background:{color_max};
                border:1px solid #aaa;
                vertical-align:middle;
            "></span>
            &nbsp;More than triple Pyoframe's overhead (≥3×)
            </span>
            """
            )
        )
        .tab_source_note(
            gt.html(
                """
                k = thousand; M = million; ms = milliseconds; s = seconds; min = minutes; kB = 1024 bytes; MB = 1,024² bytes; GB = 1,024³ bytes
                <br/>TO = Timeout (benchmark did not complete within the 20 minute time limit)
                <br/>NS = Not Supported (Linopy does not support quadratic constraints)
                <br/>NI = Not Implemented (CVXPY and PuLP were not implemented for all benchmarks to limit the benchmarking scope)
                <br/>* The facility location benchmark developed by the JuMP and PyOptInterface authors does not involve solving the optimization problem.<br/>Only the time and memory needed to construct the problem is measured.
                """
            )
        )
    )

    color_norm = mpl.colors.Normalize(vmin=log(vmin), vmax=log(vmax), clip=True)
    anchors = [(vmin, color_min), (1, "white"), (vmax, color_max)]
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
        RESULTS_FOLDER / "benchmark_results_table.png",
        web_driver="edge",
        scale=2,
    )
    table.write_raw_html(RESULTS_FOLDER / "benchmark_results_table.html")
    table
    return n_libraries, results_table


@app.cell
def _(results_table):
    results_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw time and memory values
    """)
    return


@app.cell
def _(
    RESULTS_FOLDER,
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
        format_numeric("time_solver_median", fill_null="N/A*"),
        format_numeric("memory_solver_median", fill_null="N/A*"),
        time=pl.col("time_overhead").map_elements(round_two_sig_figs, pl.String),
        memory=pl.col("memory_overhead").map_elements(round_two_sig_figs, pl.String),
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
            "time_solver_median",
            "memory_solver_median",
        ],
    ).fill_null("NI")

    # Reshuffle column order
    _cols = (
        ["problem", "problem_name", "size", "time_solver_median"]
        + [
            c
            for c in _results_table.columns
            if c.startswith("time_")
            if c != "time_solver_median"
        ]
        + ["memory_solver_median"]
        + [
            c
            for c in _results_table.columns
            if c.startswith("memory")
            if c != "memory_solver_median"
        ]
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
    _col_names["time_solver_median"] = "Gurobi Solve Time (s)"
    _col_names["memory_solver_median"] = "Gurobi Memory Usage (GB)"
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
        RESULTS_FOLDER / "benchmark_results_table_raw.png",
        web_driver="edge",
        scale=2,
    )
    _table
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
