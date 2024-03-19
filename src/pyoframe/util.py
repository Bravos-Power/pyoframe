from typing import Any, Iterable
import polars as pl
import pandas as pd

from pyoframe.dataframe import RESERVED_COL_KEYS


def _parse_inputs_as_iterable(
    inputs: tuple[Any, ...] | tuple[Iterable[Any]],
) -> Iterable[Any]:
    # Inspired from the polars library
    if not inputs:
        return []

    # Treat elements of a single iterable as separate inputs
    if len(inputs) == 1 and _is_iterable(inputs[0]):
        return inputs[0]

    return inputs


def _is_iterable(input: Any | Iterable[Any]) -> bool:
    # Inspired from the polars library
    return isinstance(input, Iterable) and not isinstance(
        input,
        (str, bytes, pl.DataFrame, pl.Series, pd.DataFrame, pd.Series, pd.Index, dict),
    )
