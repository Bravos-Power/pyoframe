"""Produces plots from benchmark results."""

from pathlib import Path

import altair as alt
import polars as pl

from benchmark_utils import mock_snakemake


def collect_benchmarks(input_files) -> pl.DataFrame:
    assert input_files, "No input files provided to plot_results.py"
    dfs = []
    for input_file in input_files:
        input_file = Path(input_file)
        name = input_file.name.partition(".")[0].split("_")
        library, solver, size, metric = name[0], name[1], int(name[2]), name[3]
        if metric != "time":
            continue

        with open(input_file) as f:
            ts = f.readlines()
        if ts[0].strip().startswith("timeout"):
            continue
        t_avg = sum(float(t) for t in ts) / len(ts)

        mem_file = input_file.parent / f"{library}_{solver}_{size}_mem.tsv"
        mem_df = pl.read_csv(mem_file, separator="\t").mean()
        mem_df = (
            mem_df.with_columns(pl.lit(t_avg).alias("s"))
            .drop("h:m:s", "cpu_time")
            .with_columns(
                library=pl.lit(library),
                size=pl.lit(size),
                solver=pl.lit(solver),
            )
        )

        dfs.append(mem_df)

    return pl.concat(dfs)


def normalize_results(results: pl.DataFrame):
    join_cols = ["size", "solver", "library"]
    non_join_cols = [col for col in results.columns if col not in join_cols]

    pyoframe_results = results.filter(pl.col("library") == "pyoframe").drop("library")
    assert pyoframe_results.height > 0, (
        "Cannot normalize results: no pyoframe data found"
    )
    results = results.join(
        pyoframe_results,
        on=["size", "solver"],
    )

    results = results.select(
        *(join_cols + [pl.col(c) / pl.col(f"{c}_right") for c in non_join_cols])
    )

    return results


def plot(results: pl.DataFrame, output):
    assert results.height > 0, "No results to plot"
    if results.get_column("max_pss").sum() == 0:
        results = results.rename({"max_uss": "memory"})
    else:
        results = results.rename({"max_pss": "memory"})

    combined_plot = None
    for solver in results.get_column("solver").unique():
        solver_results = results.filter(pl.col("solver") == solver)

        left_plot = (
            alt.Chart(solver_results)
            .mark_line(point=True)
            .encode(
                x=alt.X("size", scale=alt.Scale(type="log")),
                y=alt.Y("s", scale=alt.Scale(type="log")),
                color="library",
            )
            .properties(title=f"Time to construct (log-log, {solver})")
        )

        right_plot = (
            alt.Chart(solver_results)
            .mark_line(point=True)
            .encode(
                x=alt.X("size", scale=alt.Scale(type="log")),
                y=alt.Y("memory", scale=alt.Scale(type="log")),
                color="library",
            )
            .properties(title=f"Peak memory usage (log-log, {solver})")
        )

        if combined_plot is None:
            combined_plot = left_plot | right_plot
        else:
            combined_plot &= left_plot | right_plot
    combined_plot.save(output)


if __name__ == "__main__":
    if "snakemake" not in globals() or hasattr(snakemake, "mock"):  # noqa: F821
        snakemake = mock_snakemake("plot_results", problem="facility_location")

    results = collect_benchmarks(snakemake.input)
    results.write_csv(snakemake.output[0])

    plot(results, snakemake.output[1])

    results = normalize_results(results)

    plot(results, snakemake.output[2])
