from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple, overload

import polars as pl

from convop.parameters import Parameter


@overload
def load_parameters(
    df_or_path: str | pl.DataFrame | Path,
    param_names: str,
    dim: int | None = None,
) -> Parameter: ...


@overload
def load_parameters(
    df_or_path: str | pl.DataFrame | Path,
    param_names: Sequence[str],
    dim: int | None = None,
) -> Tuple[Parameter, ...]: ...


def load_parameters(
    df_or_path: str | pl.DataFrame | Path,
    param_names: str | Sequence[str],
    dim: int | None = None,
) -> Tuple[Parameter, ...] | Parameter:
    """Reads a DataFrame or file and returns a Parameters object for each column in param_names."""
    if isinstance(param_names, str):
        param_names = [param_names]

    if isinstance(df_or_path, (str, Path)):
        df_or_path = _read_file(df_or_path)

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
        assert (
            param_name in df_or_path.columns
        ), f"Expected column '{param_name}' was not found in DataFrame."
        params.append(Parameter(df_or_path, index_columns, param_name, param_name))

    if len(params) == 1:
        return params[0]
    return tuple(params)


def _read_file(path: str | Path) -> pl.DataFrame:
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
