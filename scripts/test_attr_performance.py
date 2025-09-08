"""Script to test the performance of two approaches to retrieving the solution of a variable.

Developed on Sep 2nd, 2025. Compares the use of Polars' map_elements vs. Python's list comprehension.

On Windows and Python 13, performance was comparable, if not a bit better for the list comprehension approach.

Note that the Polars deprecation error is not reproduced here since it only arises when map_elements throws an error.
"""

import time

import polars as pl
import pyoptinterface as poi
from pyoptinterface.gurobi import Model

from pyoframe._constants import KEY_TYPE, VAR_KEY


def get_attr(n, model, use_map_elements):
    name = "Value"
    col_name = name

    try:
        name = poi.VariableAttribute[name]
        getter = model.get_variable_attribute
    except KeyError:
        getter = model.get_variable_raw_attribute

    data = pl.DataFrame().with_columns(
        pl.int_range(1, n + 1).alias("dim1"),
        pl.int_range(n).alias(VAR_KEY).cast(KEY_TYPE),
    )

    t1 = time.time()

    if use_map_elements:
        res = data.with_columns(
            pl.col(VAR_KEY)
            .map_elements(lambda v_id: getter(poi.VariableIndex(v_id), name))
            .alias(col_name)
        ).select(["dim1"] + [col_name])
    else:
        ids = data.get_column(VAR_KEY).to_list()
        attr = [getter(poi.VariableIndex(v_id), name) for v_id in ids]
        res = data.with_columns(pl.Series(attr).alias(col_name)).select(
            ["dim1"] + [col_name]
        )

    t2 = time.time()

    print(res)

    return t2 - t1


def main(Ns=[3, 10, 100, 1000, 10_000, 100_000, 1_000_000]):
    results = []
    for n in Ns:
        model = Model()
        for i in range(n):
            model.add_variable(lb=i, ub=i)
        model.optimize()
        r1 = get_attr(n, model, use_map_elements=False)
        r2 = get_attr(n, model, use_map_elements=True)
        r3 = get_attr(n, model, use_map_elements=False)
        r4 = get_attr(n, model, use_map_elements=True)
        results.append(f"!N={n}:\tnew={r1:.4g}\t{r3:.4g}\told={r2:.4g}\t{r4:.2g}")

    print("\n".join(results))


if __name__ == "__main__":
    main()
