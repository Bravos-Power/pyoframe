import polars as pl
from typing import Tuple, Union, TYPE_CHECKING, overload
import io

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.constraints import Expression


@overload
def csvs_to_dataframe(
    csv_strings: str,
) -> pl.DataFrame: ...


@overload
def csvs_to_dataframe(
    *csv_strings: str,
) -> Tuple[pl.DataFrame, ...]: ...


def csvs_to_dataframe(
    *csv_strings: str,
) -> Union[Tuple[pl.DataFrame, ...], pl.DataFrame]:
    dfs = []
    for csv_string in csv_strings:
        csv_string = "\n".join(line.strip() for line in csv_string.splitlines())
        dfs.append(pl.read_csv(io.StringIO(csv_string)))
    if len(dfs) == 1:
        return dfs[0]
    return tuple(dfs)


@overload
def csvs_to_expr(
    csv_strings: str,
) -> "Expression": ...


@overload
def csvs_to_expr(
    *csv_strings: str,
) -> Tuple["Expression", ...]: ...


def csvs_to_expr(
    *csv_strings: str,
) -> Union[Tuple["Expression", ...], "Expression"]:
    if len(csv_strings) == 1:
        return csvs_to_dataframe(*csv_strings).to_expr()
    return tuple((df.to_expr() for df in csvs_to_dataframe(*csv_strings)))
