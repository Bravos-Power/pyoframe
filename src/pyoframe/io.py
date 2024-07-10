"""
Module containing all import/export functionalities.
"""

import sys
import time
from io import TextIOWrapper
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Iterable, Optional, TypeVar, Union

from pyoframe.constants import CONST_TERM, VAR_KEY, ObjSense
from pyoframe.core import Constraint, Variable
from pyoframe.io_mappers import (
    Base36ConstMapper,
    Base36VarMapper,
    IOMappers,
    Mapper,
    NamedMapper,
    NamedVariableMapper,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Model

import polars as pl

T = TypeVar("T")


def io_progress_bar(
    iterable: Iterable[T],
    prefix: str = "",
    suffix: str = "",
    length: int = 50,
    fill: str = "█",
    update_every: int = 1,
):
    """
    Display progress bar for I/O operations.
    """
    try:
        total = len(iterable)
    except TypeError:
        total = None

    start_time = time.time()

    def print_progress(iteration: int):
        if total is not None:
            percent = f"{100 * (iteration / float(total)):.1f}"
            filled_length = int(length * iteration // total)
            bar = fill * filled_length + "-" * (length - filled_length)
        else:
            percent = "N/A"
            bar = fill * (iteration % length) + "-" * (length - (iteration % length))
        elapsed_time = time.time() - start_time
        if iteration > 0:
            estimated_total_time = (
                elapsed_time * (total / iteration) if total else elapsed_time
            )
            estimated_remaining_time = estimated_total_time - elapsed_time
            eta = time.strftime("%H:%M:%S", time.gmtime(estimated_remaining_time))
        else:
            eta = "Estimating..."  # pragma: no cover
        sys.stdout.write(
            f'\r{prefix} |{bar}| {percent}% Complete ({iteration}/{total if total else "?"}) ETA: {eta} {suffix}'
        )
        sys.stdout.flush()

    for i, item in enumerate(iterable):
        yield item
        if (i + 1) % update_every == 0 or total is None or i == total - 1:
            print_progress(i + 1)

    sys.stdout.write("\n")
    sys.stdout.flush()


def objective_to_file(m: "Model", f: TextIOWrapper, var_map):
    """
    Write out the objective of a model to a lp file.
    """
    if m.objective is None:
        return
    objective_sense = "minimize" if m.sense == ObjSense.MIN else "maximize"
    f.write(f"{objective_sense}\n\nobj:\n\n")
    result = m.objective.to_str(
        var_map=var_map, include_prefix=False, include_const_variable=True
    )
    f.write(result)


def constraints_to_file(m: "Model", f: TextIOWrapper, var_map, const_map):
    for constraint in create_section(
        io_progress_bar(
            m.constraints, prefix="Writing constraints to file", update_every=5
        ),
        f,
        "s.t.",
    ):
        f.write(constraint.to_str(var_map=var_map, const_map=const_map) + "\n")


def bounds_to_file(m: "Model", f, var_map):
    """
    Write out variables of a model to a lp file.
    """
    if (m.objective is not None and m.objective.has_constant) or len(m.variables) != 0:
        f.write("\n\nbounds\n\n")
    if m.objective is not None and m.objective.has_constant:
        const_term_df = pl.DataFrame(
            {VAR_KEY: [CONST_TERM]}, schema={VAR_KEY: pl.UInt32}
        )
        f.write(f"{var_map.apply(const_term_df).item()} = 1\n")

    for variable in io_progress_bar(
        m.variables, prefix="Writing bounds to file", update_every=1
    ):
        terms = []

        if variable.lb != 0:
            terms.append(pl.lit(f"{variable.lb:.12g} <= "))

        terms.append(VAR_KEY)

        if variable.ub != float("inf"):
            terms.append(pl.lit(f" <= {variable.ub:.12g}"))

        terms.append(pl.lit("\n"))

        if len(terms) < 3:
            continue

        df = (
            var_map.apply(variable.data, to_col=None)
            .select(pl.concat_str(terms).str.concat(""))
            .item()
        )

        f.write(df)


def binaries_to_file(m: "Model", f, var_map: Mapper):
    """
    Write out binaries of a model to a lp file.
    """
    for variable in create_section(
        io_progress_bar(
            m.binary_variables,
            prefix="Writing binary variables to file",
            update_every=1,
        ),
        f,
        "binary",
    ):
        lines = (
            var_map.apply(variable.data, to_col=None)
            .select(pl.col(VAR_KEY).str.concat("\n"))
            .item()
        )
        f.write(lines + "\n")


def integers_to_file(m: "Model", f, var_map: Mapper):
    """
    Write out integers of a model to a lp file.
    """
    for variable in create_section(
        io_progress_bar(
            m.integer_variables,
            prefix="Writing integer variables to file",
            update_every=5,
        ),
        f,
        "general",
    ):
        lines = (
            var_map.apply(variable.data, to_col=None)
            .select(pl.col(VAR_KEY).str.concat("\n"))
            .item()
        )
        f.write(lines + "\n")


def create_section(iterable: Iterable[T], f, section_header) -> Iterable[T]:
    wrote = False
    for item in iterable:
        if not wrote:
            f.write(f"\n\n{section_header}\n\n")
            wrote = True
        yield item


def get_var_map(m: "Model", use_var_names):
    if use_var_names:
        if m.var_map is not None:
            return m.var_map
        var_map = NamedVariableMapper(Variable)
    else:
        var_map = Base36VarMapper(Variable)

    for v in m.variables:
        var_map.add(v)
    return var_map


def to_file(
    m: "Model", file_path: Optional[Union[str, Path]] = None, use_var_names=False
) -> Path:
    """
    Write out a model to a lp file.

    Args:
        m: The model to write out.
        file_path: The path to write the model to. If None, a temporary file is created. The caller is responsible for
            deleting the file after use.
        use_var_names: If True, variable names are used in the lp file. Otherwise, variable
            indices are used.

    Returns:
        The path to the lp file.
    """
    if file_path is None:
        with NamedTemporaryFile(
            prefix="pyoframe-problem-", suffix=".lp", mode="w", delete=False
        ) as f:
            file_path = f.name

    file_path = Path(file_path)
    assert file_path.suffix == ".lp", f"File format `{file_path.suffix}` not supported."

    if file_path.exists():
        file_path.unlink()

    const_map = (
        NamedMapper(Constraint) if use_var_names else Base36ConstMapper(Constraint)
    )
    for c in m.constraints:
        const_map.add(c)
    var_map = get_var_map(m, use_var_names)
    m.io_mappers = IOMappers(var_map, const_map)

    with open(file_path, mode="w") as f:
        objective_to_file(m, f, var_map)
        constraints_to_file(m, f, var_map, const_map)
        bounds_to_file(m, f, var_map)
        binaries_to_file(m, f, var_map)
        integers_to_file(m, f, var_map)
        f.write("\nend\n")

    return file_path
