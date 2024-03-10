"""
Source code heavily based on the `linopy` package by Fabian Hofmann.
Copyright 2015-2021 PyPSA Developers
Copyright 2021-2023 Fabian Hofmann
Copyright 2024 Bravos Energía
MIT License

Module containing all import/export functionalities.
"""

from io import TextIOWrapper
import logging
from tempfile import NamedTemporaryFile
import time
from pathlib import Path
import typing

from tqdm import tqdm
from convop.expressions import COEF_KEY, CONST_KEY, VAR_KEY, Expression

if typing.TYPE_CHECKING:
    from convop.model import Model

import polars as pl


logger = logging.getLogger(__name__)

_BATCH_SIZE = 5000

ufunc_kwargs = dict(vectorize=True)

TQDM_COLOR = "#80bfff"


def handle_batch(batch, f):
    """
    Write out a batch to a file and reset the batch.
    """
    if len(batch) >= _BATCH_SIZE:
        f.writelines(batch)  # write out a batch
        batch = []  # reset batch
    return batch


def _expression_vars_to_string(expr: Expression, sort=True) -> pl.DataFrame:
    result = expr.variables
    if sort:
        result = result.sort(by=VAR_KEY)
    dimensions = expr.dimensions

    result = result.with_columns(
        result=pl.concat_str(
            pl.when(pl.col(COEF_KEY) < 0).then(pl.lit("")).otherwise(pl.lit("+")),
            pl.col(COEF_KEY).cast(pl.String),
            pl.lit(" x"),
            pl.col(VAR_KEY).cast(pl.String),
            pl.lit("\n"),
        )
    ).drop(COEF_KEY, VAR_KEY)

    if dimensions:
        result = result.group_by(dimensions).agg(
            pl.col("result").str.concat(delimiter="")
        )
    else:
        result = result.select(pl.col("result").str.concat(delimiter=""))

    return result


def objective_to_file(m: "Model", f: TextIOWrapper, log=False):
    """
    Write out the objective of a model to a lp file.
    """
    if log:
        logger.info("Writing objective.")

    objective = m.objective
    assert objective is not None, "No objective set."

    f.write(f"{objective.sense}\n\nobj:\n\n")
    assert objective.expr.constants.is_empty, "Objective cannot have constant terms."
    result = _expression_vars_to_string(objective.expr, sort=True)
    f.writelines(result.item())


def constraints_to_file(m: "Model", f: TextIOWrapper, log=False):
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
        lhs = _expression_vars_to_string(constraint.lhs).rename({"result": "lhs"})
        rhs = constraint.lhs.constants.with_columns(pl.col(CONST_KEY) * -1)
        expression = pl.concat([lhs, rhs], how="align")
        expression = expression.select(
            result=pl.concat_str(
                pl.lit(f"c{constraint.name}:\n"),
                pl.col("lhs"),
                pl.lit(f" {constraint.sense.value} "),
                pl.col(CONST_KEY).cast(pl.String),
                pl.lit("\n\n"),
            )
        ).to_series()
        f.writelines(expression)


def bounds_to_file(m: "Model", f, log=False):
    """
    Write out variables of a model to a lp file.
    """
    if not m.variables:
        return

    f.write("\n\nbounds\n\n")
    # if log:
    #     names = tqdm(
    #         list(names),
    #         desc="Writing continuous variables.",
    #         colour=TQDM_COLOR,
    #     )

    for variable in m.variables:
        lb = "-inf" if variable.lb is None else f"{variable.lb:+.12g}"
        ub = "inf" if variable.ub is None else f"{variable.ub:+.12g}"

        df = variable.data.select(
            result=pl.concat_str(
                pl.lit(f"{lb} <= x"),
                pl.col(VAR_KEY).cast(pl.String),
                pl.lit(f" <= {ub}\n"),
            )
        ).to_series()
        df = df.str.concat(delimiter="")

        f.writelines(df.item())


def binaries_to_file(m: "Model", f, log=False):
    """
    Write out binaries of a model to a lp file.
    """

    names = m.variables.binaries
    if not len(list(names)):
        return

    f.write("\n\nbinary\n\n")
    if log:
        names = tqdm(
            list(names),
            desc="Writing binary variables.",
            colour=TQDM_COLOR,
        )

    batch = []  # to store batch of lines
    for name in names:
        df = m.variables[name].flat

        for label in df.labels.values:
            batch.append(f"x{label}\n")
            batch = handle_batch(batch, f)

    if batch:  # write the remaining lines
        f.writelines(batch)


def integers_to_file(m: "Model", f, log=False, integer_label="general"):
    """
    Write out integers of a model to a lp file.
    """
    names = m.variables.integers
    if not len(list(names)):
        return

    f.write(f"\n\n{integer_label}\n\n")
    if log:
        names = tqdm(
            list(names),
            desc="Writing integer variables.",
            colour=TQDM_COLOR,
        )

    batch = []  # to store batch of lines
    for name in names:
        df = m.variables[name].flat

        for label in df.labels.values:
            batch.append(f"x{label}\n")
            batch = handle_batch(batch, f)

    if batch:  # write the remaining lines
        f.writelines(batch)


def to_file(m: "Model", fn: str | Path | None, integer_label="general") -> Path:
    """
    Write out a model to a lp file.
    """
    if fn is None:
        with NamedTemporaryFile(
            prefix="linoframe-problem-",
            suffix=".lp",
            mode="w",
            delete=False,
        ) as f:
            fn = f.name

    fn = Path(fn)
    assert fn.suffix == ".lp", f"File format `{fn.suffix}` not supported."

    if fn.exists():
        fn.unlink()

    with open(fn, mode="w") as f:
        start = time.time()

        objective_to_file(m, f)
        constraints_to_file(m, f)
        bounds_to_file(m, f)
        # binaries_to_file(m, f)
        # integers_to_file(m, f, integer_label=integer_label)
        f.write("end\n")

        logger.info(" Writing time: %s", round(time.time() - start, 2))

    return fn
