"""
Module containing all import/export functionalities.
"""

from io import TextIOWrapper
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, TypeVar, Union

from pyoframe.constants import VAR_KEY, Config
from pyoframe.constraints import Constraint
from pyoframe.variables import Variable
from pyoframe.io_mappers import (
    Base62ConstMapper,
    Base62VarMapper,
    IOMappers,
    Mapper,
    NamedMapper,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Model

import polars as pl


def objective_to_file(m: "Model", f: TextIOWrapper, var_map):
    """
    Write out the objective of a model to a lp file.
    """
    assert m.objective is not None, "No objective set."

    f.write(f"{m.objective.sense.value}\n\nobj:\n\n")
    result = m.objective.to_str(var_map=var_map, include_prefix=False)
    f.writelines(result)


def constraints_to_file(m: "Model", f: TextIOWrapper, var_map, const_map):
    for constraint in create_section(m.constraints, f, "s.t."):
        f.writelines(constraint.to_str(var_map=var_map, const_map=const_map) + "\n")


def bounds_to_file(m: "Model", f, var_map):
    """
    Write out variables of a model to a lp file.
    """
    for variable in create_section(m.variables, f, "bounds"):
        lb = f"{variable.lb:.12g}"
        ub = f"{variable.ub:.12g}"

        df = (
            var_map.apply(variable.data, to_col=None)
            .select(
                pl.concat_str(
                    pl.lit(f"{lb} <= "), VAR_KEY, pl.lit(f" <= {ub}\n")
                ).str.concat("")
            )
            .item()
        )

        f.writelines(df)


def binaries_to_file(m: "Model", f, var_map: Mapper):
    """
    Write out binaries of a model to a lp file.
    """
    for variable in create_section(m.binary_variables, f, "binary"):
        lines = (
            var_map.apply(variable.data, to_col=None)
            .select(pl.col(VAR_KEY).str.concat("\n"))
            .item()
        )
        f.writelines(lines + "\n")


def integers_to_file(m: "Model", f, var_map: Mapper):
    """
    Write out integers of a model to a lp file.
    """
    for variable in create_section(m.integer_variables, f, "general"):
        lines = (
            var_map.apply(variable.data, to_col=None)
            .select(pl.col(VAR_KEY).str.concat("\n"))
            .item()
        )
        f.writelines(lines + "\n")


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
        var_map = NamedMapper(Variable)
    else:
        var_map = Base62VarMapper(Variable)

    for v in m.variables:
        var_map.add(v)
    return var_map


def to_file(
    m: "Model", file_path: Optional[Union[str, Path]], use_var_names=False
) -> Path:
    """
    Write out a model to a lp file.
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
        NamedMapper(Constraint) if use_var_names else Base62ConstMapper(Constraint)
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
        f.write("end\n")

    return file_path
