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
    results = results_raw

    # Only include gurobi for now
    results = results.filter(solver="gurobi").drop("solver")

    # keep only latest result
    results = results.sort("date").unique(
        subset=["problem", "library", "size"], keep="last", maintain_order=True
    )

    # remove errors
    results = results.filter(pl.col("error").is_null()).drop("error")

    # compute overhead as total_time - solve_time
    results = results.with_columns(
        overhead_time_s=pl.col("total_time_s") - pl.col("solve_time_s")
    )

    # Replace zero solve_time with N/A
    results = results.with_columns(
        pl.when(pl.col("solve_time_s") == 0)
        .then(pl.lit(None))
        .otherwise(pl.col("solve_time_s"))
        .alias("solve_time_s")
    )

    # convert to GiB
    results = results.with_columns(
        max_memory_uss_gib=pl.col("max_memory_uss_mb") / 1024
    )

    # Determine mode number of variables, and average solve time
    results = results.with_columns(
        pl.col("num_variables").mode().first().over("problem", "size"),
        pl.col("solve_time_s").mean().over("problem", "size").alias("avg_solve_time_s"),
    )

    # Drop problems with 10K or less variables
    results = results.filter(pl.col("num_variables") > 10_000)

    # Only keep relevant columns for the table
    results = results.select(
        "problem",
        "library",
        "size",
        "overhead_time_s",
        "max_memory_uss_gib",
        "num_variables",
        "avg_solve_time_s",
    )

    # Merge pyoframe results to get relative overheads
    pyoframe_results = results.filter(library="pyoframe")
    results = results.join(
        pyoframe_results.select(
            "problem", "size", "overhead_time_s", "max_memory_uss_gib"
        ),
        on=["problem", "size"],
        how="left",
        suffix="_pyoframe",
    )
    results = results.with_columns(
        overhead_time_relative=pl.col("overhead_time_s")
        / pl.col("overhead_time_s_pyoframe"),
        memory_relative=pl.col("max_memory_uss_gib")
        / pl.col("max_memory_uss_gib_pyoframe"),
    )

    # determine color
    results = results.with_columns(
        overhead_time_color=pl.when(pl.col("overhead_time_relative") < 0.9)
        .then(pl.lit("green"))
        .when(pl.col("overhead_time_relative") <= 1.1)
        .then(pl.lit("black"))
        .when(pl.col("overhead_time_relative") > 5)
        .then(pl.lit("red"))
        .otherwise(pl.lit("darkred")),
        memory_color=pl.when(pl.col("memory_relative") < 0.9)
        .then(pl.lit("green"))
        .when(pl.col("memory_relative") <= 1.1)
        .then(pl.lit("black"))
        .when(pl.col("memory_relative") > 5)
        .then(pl.lit("red"))
        .otherwise(pl.lit("darkred")),
    )

    def round_two_sig_figs(val):
        if val >= 10:
            return f"{val:.0f}"
        if val >= 1:
            return f"{val:.1f}"
        return f"{val:.2f}"

    def human_format(num):
        for unit in ["", "K", "M", "B", "T"]:
            if abs(num) < 1000:
                return f"{num:.0f}{unit}"
            num /= 1000
        return f"{num:.0f}P"

    # Round seconds to 1 decimal place
    results = results.with_columns(
        pl.col("avg_solve_time_s")
        .map_elements(round_two_sig_figs, pl.String)
        .fill_null("N/A"),
        time=pl.concat_str(
            pl.lit("<span style='font-weight: bold"),
            # pl.col("overhead_time_color"),
            pl.lit(";'>"),
            pl.col("overhead_time_relative").map_elements(
                round_two_sig_figs, pl.String
            ),
            pl.lit("x</span><br/><span style='color: grey;'>("),
            pl.col("overhead_time_s").map_elements(round_two_sig_figs, pl.String),
            pl.lit(")</span>"),
        ),
        memory=pl.concat_str(
            pl.lit("<span style='font-weight: bold"),
            # pl.col("memory_color"),
            pl.lit(";'>"),
            pl.col("memory_relative").map_elements(round_two_sig_figs, pl.String),
            pl.lit("x</span><br/><span style='color: grey;'>("),
            pl.col("max_memory_uss_gib").map_elements(round_two_sig_figs, pl.String),
            pl.lit(")</span>"),
        ),
        size=pl.concat_str(
            pl.col("num_variables").map_elements(human_format, pl.String),
            pl.when(problem="simple_problem")
            .then(pl.lit(""))
            .otherwise(
                pl.concat_str(
                    pl.lit("<br><span style='color: grey;'>(n="),
                    pl.col("size"),
                    pl.lit(")</span>"),
                )
            ),
        ),
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
        pl.col("problem").replace_strict(
            {
                "simple_problem": "Simple Problem",
                "energy_planning_capacity_expansion": "Electrical Grid Capacity Expansion Problem",
                "energy_planning_security_constrained_dispatch": "Electrical Grid Security-Constrained Dispatch Problem",
                "facility_location": "Facility Location Problem (from JuMP paper)",
            }
        ),
        problem_order=pl.col("problem").replace_strict(
            {
                "simple_problem": 0,
                "energy_planning_capacity_expansion": 1,
                "energy_planning_security_constrained_dispatch": 2,
                "facility_location": 3,
            }
        ),
    )
    results = results.sort(["problem_order"])

    results
    return (results,)


@app.cell
def _(Path, RESULTS_FILE, gt, log, mpl, results):
    results_table = results

    # Pivot
    results_table = results_table.select(
        "problem", "library", "size", "time", "memory", "avg_solve_time_s"
    )
    results_table = results_table.pivot(
        on="library", index=["problem", "size", "avg_solve_time_s"]
    ).fill_null("—")

    col_names = {c: c.split("_")[-1] for c in results_table.columns}
    col_names["avg_solve_time_s"] = "Gurobi Solve Time (s)"

    table = (
        gt.GT(results_table)
        .tab_stub(rowname_col="size", groupname_col="problem")
        .tab_stubhead(
            label=gt.html(
                "Number of variables<br/><span style='color: grey;'>(Problem size)</span>"
            )
        )
        .tab_spanner(
            gt.html(
                "Overhead Time Relative to Pyoframe<br><span style='color: grey;'>(Overhead in s)</span>"
            ),
            columns=[c for c in results_table.columns if c.startswith("time")],
        )
        .tab_spanner(
            gt.html(
                "Peak Memory Usage Relative to Pyoframe<br><span style='color: grey;'>(Peak Memory in GiB)</span>"
            ),
            columns=[c for c in results_table.columns if "memory" in c],
        )
        .cols_label(col_names)
        .tab_style(
            style=gt.style.borders(sides="left"),
            locations=gt.loc.body(columns=results["library"].n_unique() + 3),
        )
        .tab_style(
            style=gt.style.borders(sides="left"),
            locations=gt.loc.body(columns=3),
        )
        .cols_label_rotate()
        # .opt_row_striping(row_striping=False)
        .tab_options(row_striping_background_color="white", data_row_padding="0.5")
        .cols_align(
            align="right",
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
        offset = 3
        for col_i, (library,) in enumerate(
            results.select("library").unique(maintain_order=True).iter_rows()
        ):
            color = results.filter(problem=problem, size=size, library=library)

            for i, metric in enumerate(["overhead_time_relative", "memory_relative"]):
                if color.is_empty():
                    color_hex = "white"
                else:
                    amount = color.select(metric).item()

                    color_hex = mpl.colors.to_hex(color_map(color_norm(log(amount))))

                table = table.tab_style(
                    style=gt.style.fill(color_hex),
                    locations=gt.loc.body(
                        rows=row_i,
                        columns=col_i + offset + i * results["library"].n_unique(),
                    ),
                )

    table.save(
        Path(RESULTS_FILE).parent / "benchmark_results_table.pdf", web_driver="edge"
    )
    table.write_raw_html(Path(RESULTS_FILE).parent / "benchmark_results_table.html")
    table
    return


if __name__ == "__main__":
    app.run()
