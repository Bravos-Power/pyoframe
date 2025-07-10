import polars as pl

from benchmarks.energy_planning.scripts.util import mock_snakemake

if __name__ == "__main__":
    if "snakemake" not in globals() or hasattr(snakemake, "mock"):  # noqa: F821
        snakemake = mock_snakemake(
            "plot_results", problem="facility_location", solver="gurobi"
        )

    problem = snakemake.wildcards.problem
    solver = snakemake.wildcards.solver

    results = pl.read_csv(snakemake.input[0])
    results = results.filter(pl.col("problem") == problem, pl.col("solver") == solver)

    results.plot.line(x="size", y="s", color="library").properties(
        title=f"Time for {problem} with {solver}"
    ).save(snakemake.output.time_plot)

    results.plot.line(x="size", y="max_uss", color="library").properties(
        title=f"Peak memory (MB) for {problem} with {solver}"
    ).save(snakemake.output.mem_plot)
