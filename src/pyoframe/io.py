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
from typing import TYPE_CHECKING, Iterable, Optional, TypeVar, Union

from pyoframe.constants import VAR_KEY, Config
from pyoframe.var_mapping import Base62EncodedVariables, VariableMapping

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Model

import polars as pl


def objective_to_file(m: "Model", f: TextIOWrapper, var_map):
    """
    Write out the objective of a model to a lp file.
    """
    assert m.objective is not None, "No objective set."

    f.write(f"{m.objective.sense.value}\n\nobj:\n\n")
    result = m.objective.to_str(var_map=var_map, include_name=False)
    f.writelines(result)


def constraints_to_file(m: "Model", f: TextIOWrapper, var_map):
    for constraint in create_section(m.constraints, f, "s.t."):
        f.writelines(constraint.to_str(var_map=var_map) + "\n")


def bounds_to_file(m: "Model", f, var_map):
    """
    Write out variables of a model to a lp file.
    """
    for variable in create_section(m.variables, f, "bounds"):
        lb = f"{variable.lb:.12g}"
        ub = f"{variable.ub:.12g}"

        df = (
            var_map.map_vars(variable.data)
            .select(
                pl.concat_str(
                    pl.lit(f"{lb} <= "), VAR_KEY, pl.lit(f" <= {ub}\n")
                ).str.concat("")
            )
            .item()
        )

        f.writelines(df)


def binaries_to_file(m: "Model", f, var_map: VariableMapping):
    """
    Write out binaries of a model to a lp file.
    """
    for variable in create_section(m.binary_variables, f, "binary"):
        lines = (
            var_map.map_vars(variable.data)
            .select(pl.col(VAR_KEY).str.concat("\n"))
            .item()
        )
        f.writelines(lines + "\n")


def integers_to_file(m: "Model", f, var_map: VariableMapping):
    """
    Write out integers of a model to a lp file.
    """
    for variable in create_section(m.integer_variables, f, "general"):
        lines = (
            var_map.map_vars(variable.data)
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


def to_file(m: "Model", fn: Optional[Union[str, Path]], use_var_names=None) -> Path:
    """
    Write out a model to a lp file.
    """
    if fn is None:
        with NamedTemporaryFile(
            prefix="pyoframe-problem-", suffix=".lp", mode="w", delete=False
        ) as f:
            fn = f.name

    fn = Path(fn)
    assert fn.suffix == ".lp", f"File format `{fn.suffix}` not supported."

    if fn.exists():
        fn.unlink()

    if use_var_names is None:
        use_var_names = Config.preserve_full_names
    var_map = m.var_map if use_var_names else Base62EncodedVariables()

    with open(fn, mode="w") as f:
        objective_to_file(m, f, var_map)
        constraints_to_file(m, f, var_map)
        bounds_to_file(m, f, var_map)
        binaries_to_file(m, f, var_map)
        integers_to_file(m, f, var_map)
        f.write("end\n")

    return fn
