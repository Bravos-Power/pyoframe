"""Script to filter the input data to only Jan 1st (24 hours)."""

from pathlib import Path

import polars as pl

INPUT_DIR = Path(__file__).parent / "starter_code" / "input_data"

FILES_TO_REDUCE = ["loads.parquet", "variable_capacity_factors.parquet"]


def script(dry_run=True):
    for file in FILES_TO_REDUCE:
        file = INPUT_DIR / file

        df = pl.read_parquet(file)
        assert "datetime" in df.columns, f"datetime not in {file}"

        n = df.height

        df = df.filter(
            pl.col("datetime").dt.month() == 1, pl.col("datetime").dt.day() == 1
        )  # Jan 1st

        print(
            f"Will remove {(n - df.height) / n:.1%} rows from {file} ({n} -> {df.height})..."
        )

        if not dry_run:
            # Save the reduced dataframe
            df.write_parquet(file)
            print("Removed")


if __name__ == "__main__":
    script(dry_run=False)
