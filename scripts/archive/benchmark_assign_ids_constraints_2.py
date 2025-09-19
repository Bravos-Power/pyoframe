"""A throwaway script used in the past to test different versions of assign_ids.

Learnings:
- The fastest way to group the data by dimension is using a sort, even if
    the data should maintain its group ordering.
"""

import time
from itertools import pairwise

import polars as pl
import pyoptinterface as poi

import pyoframe as pf
from pyoframe._constants import (
    COEF_KEY,
    CONSTRAINT_KEY,
    KEY_TYPE,
    RESERVED_COL_KEYS,
    VAR_KEY,
)


def assign_ids(
    poi_model,
    data: pl.DataFrame,
    maintain_order=True,
    maintain_initial_order=False,
    use_zip=False,  # slower when true
    use_iterrows=True,  # faster when true
    smart_sort=False,
):
    sense = poi.ConstraintSense.LessEqual

    key_cols = [COEF_KEY, VAR_KEY]
    dimensions = [col for col in data.columns if col not in RESERVED_COL_KEYS]
    add_constraint = poi_model._add_linear_constraint
    name = "C"
    create_expression = poi.ScalarAffineFunction

    if not maintain_order or not maintain_initial_order:
        t1 = time.time()
        if not smart_sort:
            # TODO don't shuffle entries! sort using the current order.
            #   or is it better to be sorted by keys?
            df = data.sort(dimensions, maintain_order=maintain_order)
            df_unique = df.select(dimensions).unique(maintain_order=True)
        else:
            df_unique = data.select(dimensions).unique(maintain_order=maintain_order)
            df = (
                data.join(
                    df_unique.with_row_index(), on=dimensions, maintain_order="left"
                )
                .sort("index", maintain_order=maintain_order)
                .drop("index")
            )
        t2 = time.time()
        coefs = df.get_column(COEF_KEY).to_list()
        vars = df.get_column(VAR_KEY).to_list()
        split = (
            df.lazy()
            .with_row_index()
            .filter(pl.struct(dimensions).is_first_distinct())
            .select("index")
            .collect()
            .to_series()
            .to_list()
        )
        split.append(df.height)

        df = df_unique

        # Note: list comprehension was slightly faster than using polars map_elements
        # Note 2: not specifying the argument name (`expr=`) was also slightly faster.
        # Note 3: we could have merged the if-else using an expansion operator (*) but that is slow.
        # Note 4: using kwargs is slow and including the constant term for linear expressions is faster.
        # GRBaddconstr uses sprintf when no name or "" is given. sprintf is slow. As such, we specify "C" as the name.
        # Specifying "" is the same as not specifying anything, see pyoptinterface:
        # https://github.com/metab0t/PyOptInterface/blob/6d61f3738ad86379cff71fee77077d4ea919f2d5/lib/gurobi_model.cpp#L338
        ids = [
            add_constraint(
                create_expression(coefs[s0:s1], vars[s0:s1], 0), sense, 0, name
            ).index
            for s0, s1 in pairwise(split)
        ]
    else:
        df = data.group_by(dimensions, maintain_order=maintain_order).all()
        if not use_iterrows:
            coefs = df.get_column(COEF_KEY).to_list()
            vars = df.get_column(VAR_KEY).to_list()
        else:
            df_vals = df.select(*key_cols)
        df = df.drop(*key_cols)
        # Note 2: not specifying the argument name (`expr=`) was also slightly faster.
        # Note 3: we could have merged the if-else using an expansion operator (*) but that is slow.
        # Note 4: using kwargs is slow and including the constant term for linear expressions is faster.
        # GRBaddconstr uses sprintf when no name or "" is given. sprintf is slow. As such, we specify "C" as the name.
        # Specifying "" is the same as not specifying anything, see pyoptinterface:
        # https://github.com/metab0t/PyOptInterface/blob/6d61f3738ad86379cff71fee77077d4ea919f2d5/lib/gurobi_model.cpp#L338
        if use_iterrows:
            t1 = time.time()
            ids = [
                add_constraint(create_expression(c, v, 0), sense, 0, name).index
                for c, v in df_vals.iter_rows()
            ]
            t2 = time.time()
        else:
            if use_zip:
                ids = [
                    add_constraint(create_expression(c, v, 0), sense, 0, name).index
                    for c, v in zip(coefs, vars)
                ]

            else:
                t1 = time.time()
                ids = [
                    add_constraint(
                        create_expression(coefs[i], vars[i], 0), sense, 0, name
                    ).index
                    for i in range(df.height)
                ]
                t2 = time.time()

    df = df.with_columns(pl.Series(ids, dtype=KEY_TYPE).alias(CONSTRAINT_KEY))

    return t2 - t1


def main():
    Ns = [100, 10_000, 1_000_000]
    repeat = 5

    SHUFFLE = True

    for N in Ns:
        m = pf.Model(solver="gurobi")
        m.X = pf.Variable(pf.Set(x=range(N), z=range(3)))
        expr = pf.sum("z", m.X) * 1.4
        if SHUFFLE:
            expr = pf.Expression(expr.data.sample(expr.data.height, shuffle=True))
        for i in range(repeat):
            t1 = time.time()
            dt1 = assign_ids(m.poi, expr.data, maintain_order=True)
            t2 = time.time()
            list_time = t2 - t1

            t1 = time.time()
            dt2 = assign_ids(m.poi, expr.data, smart_sort=True)
            t2 = time.time()
            numpy_time = t2 - t1

            print(
                f"N={N}: {list_time:.4f}s ({dt1:.4f}s) | {numpy_time:.4f}s ({dt2:.4f}s) | ratio={numpy_time / list_time:.2f} ({dt2 / dt1:.2f})"
            )


if __name__ == "__main__":
    main()
