"""Script to preprocess the benchmark results and generate descriptive statistics."""

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import polars as pl
    from utils import LIBRARY_NAME_MAP, PROBLEM_NAME_MAP

    return LIBRARY_NAME_MAP, PROBLEM_NAME_MAP, Path, alt, pl


@app.cell
def _(Path):
    VERSION = 7
    RESULTS_FOLDER = Path(__file__).parent / "results/main"
    return RESULTS_FOLDER, VERSION


@app.cell
def _(RESULTS_FOLDER, VERSION, pl):
    data_raw = pl.read_csv(
        RESULTS_FOLDER / "benchmark_results.csv", infer_schema_length=1000
    )
    data_raw = data_raw.filter(version=VERSION).drop("version")
    data_raw
    return (data_raw,)


@app.cell
def _(data_raw, pl):
    data = data_raw

    # Check only using gurobi
    assert (data["solver"] == "gurobi").all(), (
        "Data contains results from solvers other than gurobi."
    )
    data = data.drop("solver")

    # Check solver version is consistent
    assert data.filter(pl.col("error").is_null())["solver_version"].n_unique() == 1, (
        "Multiple solver versions found in the data."
    )
    data = data.drop("solver_version")

    data = data.drop("num_nonzeros", "num_constraints", "objective_value", "note")

    # Get latest
    data = data.filter(
        pl.col("date")
        == pl.col("date").max().over("seed", "problem", "size", "library")
    ).drop("date")

    # Improve column names
    data = data.rename(
        {
            "max_memory_uss_mb": "memory_total",
            "max_solver_memory_uss_mb": "memory_solver",
            "total_time_s": "time_total",
            "solve_time_s": "time_solver",
            "presolve_time_s": "time_presolve",
        }
    )

    # Convert to GB
    data = data.with_columns(
        pl.col("memory_total", "memory_solver") / 1024.0,
    )

    # Keep only timeout errors
    data = data.filter((pl.col("error") == "TIMEOUT") | pl.col("error").is_null())

    # Filter out partial timeouts
    has_timeout = (
        (pl.col("error") == "TIMEOUT").any().over(["problem", "library", "size"])
    )
    data = data.with_columns(
        pl.when(has_timeout).then(None).otherwise(pl.col(c)).alias(c)
        for c in [
            "time_total",
            "time_solver",
            "memory_total",
            "memory_solver",
            "time_presolve",
        ]
    )

    # Compute conversion
    data = data.with_columns(
        time_convert=pl.col("convert_to_solver_s") + pl.col("convert_from_solver_s")
    ).drop("convert_to_solver_s", "convert_from_solver_s")

    # Change 0 to None for solver in facility location
    data = data.with_columns(
        pl.when(problem="facility_location")
        .then(None)
        .otherwise("time_solver")
        .alias("time_solver"),
    )
    # Change 0 to None for solver memory (zero is invalid)
    data = data.with_columns(pl.col("memory_solver").replace(0, None))

    # Compute overhead
    data = data.with_columns(
        time_overhead=pl.col("time_total") - pl.col("time_solver"),
        time_solver_minus_presolve=pl.col("time_solver") - pl.col("time_presolve"),
        memory_overhead=pl.col("memory_total") - pl.col("memory_solver"),
        memory_solver_median=pl.col("memory_solver").median().over("problem", "size"),
        time_solver_median=pl.col("time_solver").median().over("problem", "size"),
    ).with_columns(
        memory_overhead_computed=pl.col("memory_total")
        - pl.col("memory_solver_median"),
        time_overhead_computed=pl.col("time_total") - pl.col("time_solver_median"),
    )

    # Correct for presolve
    # data = data.with_columns(
    #     time_presolve_lib = pl.col("time_presolve") - pl.col("time_presolve").mean().over("problem", "size")
    # ).with_columns(
    #     time_solver_corrected = pl.col("time_solver") - pl.col("time_presolve_lib"),
    #     time_overhead_corrected = pl.col("time_overhead") + pl.col("time_presolve_lib"),
    #     time_convert_corrected = pl.col("time_convert") + pl.col("time_presolve_lib"),
    #     time_overhead_computed_corrected = pl.col("time_overhead_computed") + pl.col("time_presolve_lib")
    # )

    # Move facility location into overhead
    data = data.with_columns(
        pl.when(problem="facility_location")
        .then("time_total")
        .otherwise("time_overhead")
        .alias("time_overhead"),
        pl.when(problem="facility_location")
        .then(None)
        .otherwise("time_total")
        .alias("time_total"),
        pl.when(problem="facility_location")
        .then("memory_total")
        .otherwise("memory_overhead")
        .alias("memory_overhead"),
        pl.when(problem="facility_location")
        .then(None)
        .otherwise("memory_total")
        .alias("memory_total"),
        pl.when(problem="facility_location")
        .then("memory_total")
        .otherwise("memory_overhead_computed")
        .alias("memory_overhead_computed"),
    )

    # Standardize number of variables
    data = data.with_columns(pl.col("num_variables").median().over("problem", "size"))

    data
    return (data,)


@app.cell
def _(data, pl):
    data_analysis = data
    data_analysis = data_analysis.filter(pl.col("error").is_null())
    assert "facility_location" in data_analysis["problem"].unique().to_list(), (
        "Facility location problem not found in the data."
    )
    data_analysis = data_analysis.filter(pl.col("problem") != "facility_location")
    data_analysis = data_analysis.unpivot(
        index=["problem", "size", "library", "seed"],
        on=[
            "time_total",
            "time_solver",
            "time_solver_minus_presolve",
            "time_overhead",
            "time_overhead_computed",
            "memory_total",
            "memory_solver",
            "memory_overhead",
            "memory_overhead_computed",
            "time_presolve",
        ],
    )

    data_analysis = data_analysis.filter(pl.col("value").is_not_null())

    data_analysis = data_analysis.with_columns(
        pl.col("variable").str.split("_").list.get(0).alias("type"),
        pl.col("variable").str.split("_").list.slice(1).list.join("_").alias("metric"),
    ).drop("variable")

    data_analysis
    return (data_analysis,)


@app.cell
def _(RESULTS_FOLDER, data_analysis, pl):
    _df = data_analysis

    _df = _df.group_by("problem", "size", "library", "type", "metric").agg(
        mean=pl.col("value").mean(),
        n=pl.len(),
        variability=(pl.col("value").max() - pl.col("value").min())
        / pl.col("value").mean(),
        cv=pl.col("value").std() / pl.col("value").mean(),
    )

    assert _df["n"].max() == 3, "Grouping messed up, should have 3 per group"

    _df_summary = _df.group_by("type", "metric").agg(
        (100 * pl.col("cv").quantile(0.5)).round(1).alias("50"),
        (100 * pl.col("cv").quantile(0.95)).round(1).alias("95"),
    )

    _df_summary = _df_summary.pivot(on="type", index="metric").sort(
        pl.col("metric").replace(
            {
                "total": "A",
                "solver": "B",
                "presolve": "B1",
                "overhead": "C",
                "overhead_computed": "D",
            }
        )
    )
    _df_summary = _df_summary.select(
        "metric", "50_time", "95_time", "50_memory", "95_memory"
    )
    _df_summary.write_csv(RESULTS_FOLDER / "variability.csv")
    _df_summary
    return


@app.cell
def _(
    LIBRARY_NAME_MAP,
    PROBLEM_NAME_MAP,
    RESULTS_FOLDER,
    alt,
    data_analysis,
    pl,
):
    _df = data_analysis
    _df = _df.filter(metric="solver")
    _df = _df.with_columns(
        benchmark_median=pl.col("value")
        .median()
        .over("problem", "size", "metric", "type")
    )
    _df = _df.filter(
        ~(
            ((pl.col("type") == "time") & (pl.col("benchmark_median") < 1))
            | ((pl.col("type") == "memory") & (pl.col("benchmark_median") < 0.5))
        )
    )
    _df = _df.with_columns(
        bias=(pl.col("value") / pl.col("benchmark_median")) - 1,
    )
    _df = _df.with_columns(
        pl.col("library").replace(LIBRARY_NAME_MAP),
        pl.col("problem").replace(PROBLEM_NAME_MAP),
    )
    _df = _df.sort("type", descending=True)
    _plt = None
    TITLES = {"time": "Solver Time (≥ 1 s)", "memory": "Solver Memory Usage (≥0.5 GB)"}

    for (_type,), _df_panel in _df.group_by("type", maintain_order=True):
        _is_left = _type == "time"
        _range = _df_panel["bias"].abs().max() * 1.01
        _plt_panel = _df_panel.plot.scatter(
            x=alt.X(
                "bias",
                title="Relative Distance from Median",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[-_range, _range]),
            ),
            y=alt.Y(
                "library",
                title="Library" if _is_left else "",
                axis=alt.Axis(labels=_is_left),
            ),
            color=alt.Color("problem", title="Benchmark"),
            tooltip=["problem", "size", "library", "metric", "type", "value", "bias"],
        ).properties(
            title=TITLES[_type],
        ) + alt.Chart().mark_rule().encode(x=alt.datum(0))
        _plt = _plt_panel if _plt is None else _plt | _plt_panel
    _plt.save(RESULTS_FOLDER / "bias.pdf")
    _plt.save(RESULTS_FOLDER / "bias.png")
    _plt
    return


@app.cell
def _(RESULTS_FOLDER, data):
    data_output = data

    data_output = data_output.select(
        "problem",
        "size",
        "library",
        "seed",
        "error",
        "num_variables",
        "time_overhead",
        "time_solver",
        "time_convert",
        "memory_solver",
        "memory_solver_median",
        memory_overhead="memory_overhead_computed",
    ).sort("problem", "size", "library", "seed")

    data_output.write_csv(RESULTS_FOLDER / "benchmark_results_processed.csv")
    data_output
    return


if __name__ == "__main__":
    app.run()
