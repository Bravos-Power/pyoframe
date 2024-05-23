"""
Module containing all import/export functionalities.
"""

from io import TextIOWrapper
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, TypeVar, Union
from tqdm import tqdm

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
        tqdm(m.constraints, desc="Writing constraints to file"), f, "s.t."
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

    for variable in tqdm(m.variables, desc="Writing bounds to file"):
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
        tqdm(m.binary_variables, "Writing binary variables to file"), f, "binary"
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
        tqdm(m.integer_variables, "Writing integer variables to file"), f, "general"
    ):
        lines = (
            var_map.apply(variable.data, to_col=None)
            .select(pl.col(VAR_KEY).str.concat("\n"))
            .item()
        )
        f.write(lines + "\n")


T = TypeVar("T")


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
