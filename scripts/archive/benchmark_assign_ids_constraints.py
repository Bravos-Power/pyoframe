"""A throwaway script used in the past to test different versions of assign_ids."""

import time
from itertools import pairwise
from typing import Any

import numpy as np
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
    data,
    use_numpy=False,
    map_elements=False,
    cache=False,
    incl_name=True,
    specify_args=False,
    use_kwargs=False,
    include_const=True,
    use_pairwise=True,
    cache_lists=False,
):
    if use_kwargs:
        kwargs: dict[str, Any] = dict(sense=poi.ConstraintSense.LessEqual, rhs=0)

    add_constraint = poi_model._add_linear_constraint
    if use_numpy:
        create_expr = poi.ScalarAffineFunction.from_numpy
    else:
        create_expr = poi.ScalarAffineFunction

    dimensions = [col for col in data.columns if col not in RESERVED_COL_KEYS]

    df = data.sort(dimensions)
    if use_numpy:
        coefs = df.get_column(COEF_KEY).to_numpy()
        vars = df.get_column(VAR_KEY).to_numpy()
    else:
        coefs = df.get_column(COEF_KEY).to_list()
        vars = df.get_column(VAR_KEY).to_list()
    df = df.drop(COEF_KEY, VAR_KEY)
    split = (
        df.lazy()
        .with_row_index()
        .filter(pl.struct(dimensions).is_first_distinct())
        .select("index")
        .collect()
        .to_series()
    )

    df = df.unique(maintain_order=True)

    if use_numpy:
        split = split.to_numpy()
        # It is even slower without the .copy()
        coefs = [coef.copy() for coef in np.array_split(coefs, split[1:])]
        vars = [var.copy() for var in np.array_split(vars, split[1:])]
        exprs = [
            create_expr(coefficients=coefs[i], variables=vars[i])
            for i in range(len(coefs))
        ]
        ids = [add_constraint(expr=expr, **kwargs).index for expr in exprs]
    else:
        split = split.to_list()
        split.append(data.height)
        if map_elements:
            df = (
                df.lazy()
                .with_row_index()
                .with_columns(
                    pl.col("index")
                    .map_elements(
                        (
                            lambda i: add_constraint(
                                expr=create_expr(
                                    coefficients=coefs[split[i] : split[i + 1]],
                                    variables=vars[split[i] : split[i + 1]],
                                ),
                                **kwargs,
                            ).index
                        ),
                        return_dtype=KEY_TYPE,
                    )
                    .alias(CONSTRAINT_KEY)
                )
                .drop("index")
                .collect()
            )
        else:
            if cache:
                expr = [
                    create_expr(
                        coefficients=coefs[split[i] : split[i + 1]],
                        variables=vars[split[i] : split[i + 1]],
                    )
                    for i in range(df.height)
                ]
                ids = [add_constraint(expr=expr, **kwargs).index for expr in expr]
            else:
                if specify_args:
                    ids = [
                        add_constraint(
                            create_expr(
                                coefficients=coefs[split[i] : split[i + 1]],
                                variables=vars[split[i] : split[i + 1]],
                            ),
                            **kwargs,
                        ).index
                        for i in range(df.height)
                    ]
                else:
                    if use_kwargs:
                        ids = [
                            add_constraint(
                                create_expr(
                                    coefs[split[i] : split[i + 1]],
                                    vars[split[i] : split[i + 1]],
                                ),
                                **kwargs,
                            ).index
                            for i in range(df.height)
                        ]
                    else:
                        sense = poi.ConstraintSense.LessEqual
                        if include_const:
                            if incl_name:
                                if use_pairwise:
                                    if cache_lists:
                                        chunked = (
                                            (coefs[s0:s1], vars[s0:s1])
                                            for s0, s1 in pairwise(split)
                                        )
                                        ids = [
                                            add_constraint(
                                                create_expr(coef, var),
                                                sense,
                                                0,
                                                "c",  # give space as name
                                            ).index
                                            for coef, var in chunked
                                        ]
                                    else:
                                        ids = [
                                            add_constraint(
                                                create_expr(coefs[s0:s1], vars[s0:s1]),
                                                sense,
                                                0,
                                                "c",  # give space as name
                                            ).index
                                            for s0, s1 in pairwise(split)
                                        ]
                                else:
                                    ids = [
                                        add_constraint(
                                            create_expr(
                                                coefs[split[i] : split[i + 1]],
                                                vars[split[i] : split[i + 1]],
                                            ),
                                            sense,
                                            0,
                                            "c",  # give space as name
                                        ).index
                                        for i in range(df.height)
                                    ]
                            else:
                                ids = [
                                    add_constraint(
                                        create_expr(
                                            coefs[split[i] : split[i + 1]],
                                            vars[split[i] : split[i + 1]],
                                        ),
                                        sense,
                                        0,
                                    ).index
                                    for i in range(df.height)
                                ]
                        else:
                            ids = [
                                add_constraint(
                                    create_expr(
                                        coefs[split[i] : split[i + 1]],
                                        vars[split[i] : split[i + 1]],
                                        0,
                                    ),
                                    sense,
                                    0,
                                ).index
                                for i in range(df.height)
                            ]

            df = df.with_columns(pl.Series(ids).alias(CONSTRAINT_KEY).cast(KEY_TYPE))


def main():
    Ns = [100, 10_000, 1_000_000]

    for N in Ns:
        m = pf.Model(solver="gurobi")
        m.X = pf.Variable(pf.Set(x=range(N), z=range(3)))
        expr = pf.sum("z", m.X) * 1.4
        t1 = time.time()
        assign_ids(m.poi, expr.data)
        t2 = time.time()
        list_time = t2 - t1

        t1 = time.time()
        assign_ids(m.poi, expr.data, cache_lists=True)
        t2 = time.time()
        numpy_time = t2 - t1

        print(f"N={N}\tlists: {list_time:.4f}s\tmape: {numpy_time:.4f}s")


if __name__ == "__main__":
    main()
