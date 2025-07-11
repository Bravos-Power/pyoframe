from pathlib import Path

import polars as pl

from benchmarks.util import mock_snakemake


def collect_benchmarks(input_files, problem):
    dfs = []
    for input_file in input_files:
        input_file = Path(input_file)
        if problem != input_file.parent.parent.name:
            continue
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


if __name__ == "__main__":
    if "snakemake" not in globals() or hasattr(snakemake, "mock"):  # noqa: F821
        snakemake = mock_snakemake("plot_results", problem="facility_location")

    problem = snakemake.wildcards.problem

    results = collect_benchmarks(snakemake.input, problem)
    results.write_csv(snakemake.output[0])

    combined_plot = None
    for solver in results.get_column("solver").unique():
        solver_results = results.filter(pl.col("solver") == solver)

        left_plot = solver_results.plot.line(
            x="size", y="s", color="library"
        ).properties(title=f"Time for {problem} with {solver}")

        right_plot = solver_results.plot.line(
            x="size", y="max_uss", color="library"
        ).properties(title=f"Peak memory (MB) for {problem} with {solver}")

        if combined_plot is None:
            combined_plot = left_plot | right_plot
        else:
            combined_plot &= left_plot | right_plot
    combined_plot.save(snakemake.output[1])
