from pathlib import Path
from typing import Any, Iterable, Tuple
from convop.constraints import Constraints
from convop.model import Model
from convop.parameters import Parameters
from convop.variables import Variables

import polars as pl


class ModelBuilder:
    def __init__(self):
        self.m = Model()

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, (Variables, Constraints, Parameters)):
            __value.name = __name
        return super().__setattr__(__name, __value)


def load_parameters(
    df_or_path: str | pl.DataFrame | Path,
    param_names: str | Iterable[str],
    dim: int | None = None,
) -> Tuple[Parameters, ...] | Parameters:
    param_names = list(param_names)

    if isinstance(df_or_path, (str, Path)):
        df_or_path = read_file(df_or_path)

    assert len(df_or_path.columns) == len(set(df_or_path.columns))

    if dim is None:
        dim = len(df_or_path.columns) - len(param_names)

    index_columns = df_or_path.columns[:dim]
    for index_col in index_columns:
        assert (
            index_col not in param_names
        ), "Index column cannot also be a parameter. Is your dimension correct?"

    params = []
    for param_name in param_names:
        assert param_name in df_or_path.columns
        params.append(Parameters(df_or_path, index_columns, param_name, param_name))

    if len(params) == 1:
        return params[0]
    return tuple(params)


def read_file(path: str | Path) -> pl.DataFrame:
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == ".csv":
        return pl.read_csv(path)
    elif path.suffix == ".parquet":
        return pl.read_parquet(path)
    else:
        raise ValueError(
            f"File type ({path.suffix}) not supported. Please use .csv or .parquet."
        )
