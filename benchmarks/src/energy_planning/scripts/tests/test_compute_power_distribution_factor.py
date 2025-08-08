"""Tests for the compute power distribution factor script.

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

    Assume reactances are all 1 except for line 4 where reactance is 0.5.

              4
             ➚↓↘
           2➚ ↓ ↘8
           ➚  ↓5 ↘
          ➚   ↓   ↘  10
    1→→→→3→→→→5→→→→→7→→→→→8
       1  ↘ 4 ↑ 7 ↗
          3↘  ↑6 ↗9
            ↘ ↑ ↗
             ↘↑↗
              6

    """
    lines = pl.DataFrame(
        schema=["line_id", "from_bus", "to_bus", "reactance"],
        data=[
            [1, 1, 3, 1],
            [2, 3, 4, 1],
            [3, 3, 6, 1],
            [4, 3, 5, 2],
            [5, 4, 5, 1],
            [6, 6, 5, 1],
            [7, 5, 7, 1],
            [8, 4, 7, 1],
            [9, 6, 7, 1],
            [10, 7, 8, 1],
        ],
        orient="row",
    )

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
snakemake.output = ["pdf.parquet"]
""",
        )

        pdf = pl.read_parquet(tmpdirname / "pdf.parquet")

    assert set(pdf.columns) == {"injection", "line", "factor"}
    pdf = pdf.sort("injection", "line").select(["injection", "line", "factor"])
    # pdf.write_csv(ROOT / "pdf.csv")
    assert_frame_equal(
        pdf,
        pl.read_csv(Path(__file__).parent / "expected_pdfs.csv"),
        check_dtypes=False,
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
