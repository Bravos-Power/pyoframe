import polars as pl

from scripts.utils import mock_snakemake

if __name__ == "__main__":
    if "snakemake" not in globals() or hasattr(snakemake, "mock"):  # noqa: F821
        snakemake = mock_snakemake("solve_energy_problem")
    pl.DataFrame().write_parquet(snakemake.output[0])
