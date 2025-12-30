"""Tests for src/pyoframe/_param.py."""

from tempfile import NamedTemporaryFile

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pyoframe as pf


def test_load_file():
    data = pl.DataFrame({"day": ["Mon", "Tue", "Wed"], "hours": [8, 9, 7]})
    param_data = pf.Param(data).data

    with NamedTemporaryFile(suffix=".csv") as tmp:
        data.write_csv(tmp.name)
        param = pf.Param(tmp.name)
        assert_frame_equal(param.data, param_data)

    with NamedTemporaryFile(suffix=".parquet") as tmp:
        data.write_parquet(tmp.name)
        param = pf.Param(tmp.name)
        assert_frame_equal(param.data, param_data)

    with NamedTemporaryFile(suffix=".feather") as tmp:
        data.write_ipc(tmp.name)
        with pytest.raises(NotImplementedError):
            pf.Param(tmp.name)
