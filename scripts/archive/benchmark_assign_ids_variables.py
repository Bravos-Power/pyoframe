"""A throwaway script used in the past to test different versions of assign_ids."""

import time

import polars as pl
import pyoptinterface as poi
from pyoptinterface import gurobi

import pyoframe as pf
from pyoframe._constants import (
    KEY_TYPE,
    VAR_KEY,
)


def assign_ids(
    poi_model: gurobi.Model, data, use_listcomp=True, no_kwargs=True, add_name=False
):
    lb = -1e100
    ub = 1e100

    kwargs = {}
    # kwargs["lb"] = float(lb)
    # kwargs["ub"] = float(ub)
    # kwargs["domain"] = poi.VariableDomain.Continuous
    domain = poi.VariableDomain.Continuous

    poi_add_var = poi_model.add_variable

    if use_listcomp:
        if no_kwargs:
            if add_name:
                ids = [
                    poi_add_var(domain, lb, ub, "v").index for _ in range(data.height)
                ]
                df = data.with_columns(pl.Series(ids, dtype=KEY_TYPE).alias(VAR_KEY))
            else:
                ids = [poi_add_var(domain, lb, ub).index for _ in range(data.height)]
                df = data.with_columns(pl.Series(ids, dtype=KEY_TYPE).alias(VAR_KEY))
        else:
            ids = [poi_add_var(**kwargs).index for _ in range(data.height)]
            df = data.with_columns(pl.Series(ids, dtype=KEY_TYPE).alias(VAR_KEY))
    else:
        df = data.with_columns(pl.lit(0).alias(VAR_KEY).cast(KEY_TYPE)).with_columns(
            pl.col(VAR_KEY).map_elements(
                lambda _: poi_add_var(**kwargs).index, return_dtype=KEY_TYPE
            )
        )
    return df


def main():
    Ns = [100, 10_000, 1_000_000, 10_000_000]

    for N in Ns:
        m = pf.Model(solver="gurobi")
        data = pf.Set(x=range(N)).data
        t1 = time.time()
        assign_ids(m.poi, data)
        t2 = time.time()
        list_time = t2 - t1

        t1 = time.time()
        assign_ids(m.poi, data, add_name=True)
        t2 = time.time()
        numpy_time = t2 - t1

        print(f"N={N}\toriginal: {list_time:.4f}s\tnew: {numpy_time:.4f}s")


if __name__ == "__main__":
    main()
