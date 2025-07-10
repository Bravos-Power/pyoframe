from pathlib import Path

import polars as pl
from energy_model.scripts.util import mock_snakemake


def collect_benchmarks(input_files):
    dfs = []
    for input_file in input_files:
        input_file = Path(input_file)
        problem = input_file.parent.parent.name
        name = input_file.name.partition(".")[0].split("_")
        library = name[0]
        solver = name[1]
        size = name[2]
        dfs.append(
            pl.read_csv(input_file, separator="\t")
            .mean()  # TODO also compute std(). Maybe at the plotting stage.
            .with_columns(
                problem=pl.lit(problem),
                library=pl.lit(library),
                solver=pl.lit(solver),
                size=pl.lit(size),
            )
        )
    return pl.concat(dfs)


if __name__ == "__main__":
    if "snakemake" not in globals() or hasattr(snakemake, "mock"):  # noqa: F821
        snakemake = mock_snakemake("collect_benchmarks")

    result = collect_benchmarks(snakemake.input)
    result.write_csv(snakemake.output[0])
