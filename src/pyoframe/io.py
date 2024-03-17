"""
Source code heavily based on the `linopy` package by Fabian Hofmann.
Copyright 2015-2021 PyPSA Developers
Copyright 2021-2023 Fabian Hofmann
Copyright 2024 Bravos Energía
MIT License

Module containing all import/export functionalities.
"""

from io import TextIOWrapper
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import TYPE_CHECKING

from pyoframe.constraints import Expression
from pyoframe.dataframe import COEF_KEY
from pyoframe.dataframe import VAR_KEY
from pyoframe.var_mapping import DEFAULT_MAP, VariableMapping

if TYPE_CHECKING:
    from pyoframe.model import Model

import polars as pl


def _expression_vars_to_string(
    expr: Expression, var_map: VariableMapping = DEFAULT_MAP, sort=True
) -> pl.DataFrame:
    result = expr.variable_terms
    if sort:
        result = result.sort(by=VAR_KEY)

    result = var_map.map_vars(result)
    dimensions = expr.dimensions

    result = result.with_columns(
        result=pl.concat_str(
            pl.when(pl.col(COEF_KEY) < 0).then(pl.lit("")).otherwise(pl.lit("+")),
            COEF_KEY,
            pl.lit(" "),
            VAR_KEY,
            pl.lit(" "),
        )
    ).drop(COEF_KEY, VAR_KEY)

    if dimensions:
        result = result.group_by(dimensions, maintain_order=True).agg(
            pl.col("result").str.concat(delimiter="")
        )
    else:
        result = result.select(pl.col("result").str.concat(delimiter=""))

    return result


def objective_to_file(m: "Model", f: TextIOWrapper, var_map):
    """
    Write out the objective of a model to a lp file.
    """
    objective = m.objective
    assert objective is not None, "No objective set."

    f.write(f"{objective.sense}\n\nobj:\n\n")
    result = objective.expr.to_str(var_map=var_map, include_header=False)
    f.writelines(result)


def constraints_to_file(m: "Model", f: TextIOWrapper, var_map):
    if not m.constraints:
        return

    f.write("\n\ns.t.\n\n")
    constraints = m.constraints
    # if log:
    #     constraints = tqdm(
    #         list(constraints),
    #         desc="Writing constraints.",
    #         colour=TQDM_COLOR,
    #     )

    for constraint in constraints:
        rhs = constraint.constant_terms.with_columns(pl.col(COEF_KEY) * -1).rename(
            {COEF_KEY: "rhs"}
        )
        lhs = constraint.to_str_table(var_map=var_map, include_const_term=False)
        constr_str = pl.concat([lhs, rhs], how="align")
        constr_str = constr_str.select(
            pl.concat_str(
                "expr",
                pl.lit(f" {constraint.sense.value} "),
                "rhs",
                pl.lit("\n"),
            )
        ).to_series()
        f.writelines(constr_str)


def bounds_to_file(m: "Model", f, var_map):
    """
    Write out variables of a model to a lp file.
    """
    if not m.variables:
        return

    f.write("\n\nbounds\n\n")

    for variable in m.variables:
        lb = "-inf" if variable.lb is None else f"{variable.lb:+.12g}"
        ub = "inf" if variable.ub is None else f"{variable.ub:+.12g}"

        df = var_map.map_vars(variable.data)

        df = df.select(
            result=pl.concat_str(pl.lit(f"{lb} <= "), VAR_KEY, pl.lit(f" <= {ub}\n"))
        ).to_series()
        df = df.str.concat(delimiter="")

        f.writelines(df.item())


# def binaries_to_file(m: "Model", f, log=False):
#     """
#     Write out binaries of a model to a lp file.
#     """

#     names = m.variables.binaries
#     if not len(list(names)):
#         return

#     f.write("\n\nbinary\n\n")
#     if log:
#         names = tqdm(
#             list(names),
#             desc="Writing binary variables.",
#             colour=TQDM_COLOR,
#         )

#     batch = []  # to store batch of lines
#     for name in names:
#         df = m.variables[name].flat

#         for label in df.labels.values:
#             batch.append(f"x{label}\n")
#             batch = handle_batch(batch, f)

#     if batch:  # write the remaining lines
#         f.writelines(batch)


# def integers_to_file(m: "Model", f, log=False, integer_label="general"):
#     """
#     Write out integers of a model to a lp file.
#     """
#     names = m.variables.integers
#     if not len(list(names)):
#         return

#     f.write(f"\n\n{integer_label}\n\n")
#     if log:
#         names = tqdm(
#             list(names),
#             desc="Writing integer variables.",
#             colour=TQDM_COLOR,
#         )

#     batch = []  # to store batch of lines
#     for name in names:
#         df = m.variables[name].flat

#         for label in df.labels.values:
#             batch.append(f"x{label}\n")
#             batch = handle_batch(batch, f)

#     if batch:  # write the remaining lines
#         f.writelines(batch)


def to_file(
    m: "Model", fn: str | Path | None, integer_label="general", use_var_names=False
) -> Path:
    """
    Write out a model to a lp file.
    """
    if fn is None:
        with NamedTemporaryFile(
            prefix="linoframe-problem-", suffix=".lp", mode="w", delete=False
        ) as f:
            fn = f.name

    fn = Path(fn)
    assert fn.suffix == ".lp", f"File format `{fn.suffix}` not supported."

    if fn.exists():
        fn.unlink()

    var_map = m.var_map if use_var_names else DEFAULT_MAP

    with open(fn, mode="w") as f:
        objective_to_file(m, f, var_map)
        constraints_to_file(m, f, var_map)
        bounds_to_file(m, f, var_map)
        # binaries_to_file(m, f)
        # integers_to_file(m, f, integer_label=integer_label)
        f.write("end\n")

    return fn
