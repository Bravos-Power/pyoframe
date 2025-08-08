"""Tests for the branch outage distribution factor script.

I made up an example and manualy checked the results were as expected.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from benchmarks.utils import run_notebook

ROOT = Path(__file__).parent.parent.parent.parent.parent


def test_mini_grid():
    """Testing the following grid.

    Assume reactances are all 1 except for line 3 where reactance is 0.5.

              4
             ➚↓↘
           1➚ ↓ ↘
           ➚  ↓4 ↘7
          ➚   ↓   ↘
         3→→→→5→→→→→7
          ↘ 3 ↑ 6 ↗
          2↘  ↑5 ↗8
            ↘ ↑ ↗
             ↘↑↗
              6

    """
    lines = pl.DataFrame(
        schema=["line_id", "from_bus", "to_bus", "reactance"],
        data=[
            [1, 3, 4, 1],
            [2, 3, 6, 1],
            [3, 3, 5, 2],
            [4, 4, 5, 1],
            [5, 6, 5, 1],
            [6, 5, 7, 1],
            [7, 4, 7, 1],
            [8, 6, 7, 1],
        ],
        orient="row",
    ).with_columns(is_leaf=pl.lit(False))

    # Save mini grid data to a temporary file
    with TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        lines.write_parquet(tmpdirname / "lines.parquet")

        run_notebook(
            notebook_path=ROOT
            / "src/energy_planning/scripts/compute_power_transfer_dist_facts.ipynb",
            working_directory=Path(tmpdirname),
            first_cell="""
from unittest.mock import Mock
snakemake = Mock()
snakemake.input = Mock()
snakemake.input.lines = "lines.parquet"
snakemake.output = ["ptdf.parquet"]
""",
        )

        run_notebook(
            notebook_path=ROOT
            / "src/energy_planning/scripts/compute_branch_outage_dist_facts.ipynb",
            working_directory=Path(tmpdirname),
            first_cell="""
from unittest.mock import Mock
snakemake = Mock()
snakemake.input = Mock()
snakemake.input.ptdf = "ptdf.parquet"
snakemake.input.lines = "lines.parquet"
snakemake.output = ["bodf.parquet"]
""",
        )

        bodf = pl.read_parquet(tmpdirname / "bodf.parquet")

    assert set(bodf.columns) == {"outage", "line", "factor"}
    bodf = bodf.sort("outage", "line").select(["outage", "line", "factor"])
    assert_frame_equal(
        bodf,
        pl.read_csv(Path(__file__).parent / "expected_bodfs.csv"),
        check_dtypes=False,
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
