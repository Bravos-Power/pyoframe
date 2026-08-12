"""Script to generate the input_data folder based on the benchmark files."""

import shutil
from pathlib import Path

import polars as pl

BENCHMARK_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / "benchmarks/src/energy_planning/model_data"
)
INPUT_DIR = Path(__file__).parent / "starter_code" / "input_data"


def main():
    copy_file(
        "generators.parquet",
        transform=lambda df: df.with_columns(
            pl.col("type").replace(
                {
                    "Other Natural Gas": "Natural Gas",
                    "CSP": "Solar",
                    "Solar PV": "Solar",
                }
            )
        )
        .filter(pl.col("type") != "IMPORT")
        .drop("hourly_overhead_per_MW_capacity", "PlantAndGenID"),
    )
    copy_file(
        "variable_capacity_factors.parquet",
        reduce_datetime=True,
        transform=lambda df: df.with_columns(
            type=pl.col("vcf_type").replace({"solar": "Solar", "wind": "Wind"})
        ).drop("vcf_type"),
    )
    copy_file("lines_simplified.parquet", dest_name="lines.parquet")
    copy_file("loads.parquet", reduce_datetime=True)

    shutil.make_archive("starter_code_for_dispatch_challenge", "zip", "starter_code")


def copy_file(name, *, dest_name=None, reduce_datetime=False, transform=None):
    if dest_name is None:
        dest_name = name

    df = pl.read_parquet(BENCHMARK_DIR / name)

    if reduce_datetime:
        df = df.filter(
            pl.col("datetime").dt.month() == 1, pl.col("datetime").dt.day() == 1
        )

    if transform is not None:
        df = transform(df)

    # Sort for consistency
    df = df.sort(df.columns)
    df.write_parquet(INPUT_DIR / dest_name)


if __name__ == "__main__":
    main()
