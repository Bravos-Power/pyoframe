from pathlib import Path

import altair as alt
import polars as pl

from benchmarks.utils import mock_snakemake


def collect_benchmarks(input_files):
    assert input_files, "No input files provided to plot_results.py"
    dfs = []
    for input_file in input_files:
        input_file = Path(input_file)
        name = input_file.name.partition(".")[0].split("_")
        library = name[0]
        solver = name[1]
        size = int(name[2])
        dfs.append(
            pl.read_csv(input_file, separator="\t")
            .mean()  # TODO also compute std(). Maybe at the plotting stage.
            .with_columns(
                library=pl.lit(library),
                size=pl.lit(size),
                solver=pl.lit(solver),
                # problem=pl.lit(problem),
            )
        )
    return pl.concat(dfs)


def normalize_results(results: pl.DataFrame):
    join_cols = ["size", "solver", "library"]
    non_join_cols = [col for col in results.columns if col not in join_cols]

    pyoframe_results = results.filter(pl.col("library") == "pyoframe").drop("library")
    results = results.join(
        pyoframe_results,
        on=["size", "solver"],
    )

    results = results.select(
        *(join_cols + [pl.col(c) / pl.col(f"{c}_right") for c in non_join_cols])
    )

    return results


def plot(results):
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
            .encode(x="size", y="s", color="library")
            .properties(title=f"Time to construct ({solver})")
        )

        right_plot = (
            alt.Chart(solver_results)
            .mark_line(point=True)
            .encode(x="size", y="memory", color="library")
            .properties(title=f"Peak memory usage (MB, {solver})")
        )

        if combined_plot is None:
            combined_plot = left_plot | right_plot
        else:
            combined_plot &= left_plot | right_plot
    combined_plot.save(snakemake.output[1])


if __name__ == "__main__":
    if "snakemake" not in globals() or hasattr(snakemake, "mock"):  # noqa: F821
        snakemake = mock_snakemake("plot_results", problem="facility_location")

    results = collect_benchmarks(snakemake.input)
    results.write_csv(snakemake.output[0])

    # results = normalize_results(results)

    plot(results)
