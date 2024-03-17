from typing import List
import polars as pl

COEF_KEY = "__coeff"
VAR_KEY = "__variable_id"


RESERVED_COL_KEYS = (COEF_KEY, VAR_KEY)


def concat_dimensions(
    df: pl.DataFrame,
    prefix: str | None = None,
    keep_dims: bool = True,
    ignore_columns=RESERVED_COL_KEYS,
    replace_spaces: bool = True,
) -> pl.DataFrame:
    """
    Returns a new DataFrame with the column 'concated_dim'. Reserved columns are ignored.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame.
    prefix : str, optional
        The prefix to be added to the concated dimension.
    keep_dims : bool, optional
        If True, the original dimensions are kept in the new DataFrame.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame(
    ...     {
    ...         "dim1": [1, 2, 3, 1, 2, 3],
    ...         "dim2": ["Y", "Y", "Y", "N", "N", "N"],
    ...     }
    ... )
    >>> concat_dimensions(df)
    shape: (6, 3)
    ┌──────┬──────┬──────────────┐
    │ dim1 ┆ dim2 ┆ concated_dim │
    │ ---  ┆ ---  ┆ ---          │
    │ i64  ┆ str  ┆ str          │
    ╞══════╪══════╪══════════════╡
    │ 1    ┆ Y    ┆ [1,Y]        │
    │ 2    ┆ Y    ┆ [2,Y]        │
    │ 3    ┆ Y    ┆ [3,Y]        │
    │ 1    ┆ N    ┆ [1,N]        │
    │ 2    ┆ N    ┆ [2,N]        │
    │ 3    ┆ N    ┆ [3,N]        │
    └──────┴──────┴──────────────┘
    >>> concat_dimensions(df, prefix="x")
    shape: (6, 3)
    ┌──────┬──────┬──────────────┐
    │ dim1 ┆ dim2 ┆ concated_dim │
    │ ---  ┆ ---  ┆ ---          │
    │ i64  ┆ str  ┆ str          │
    ╞══════╪══════╪══════════════╡
    │ 1    ┆ Y    ┆ x[1,Y]       │
    │ 2    ┆ Y    ┆ x[2,Y]       │
    │ 3    ┆ Y    ┆ x[3,Y]       │
    │ 1    ┆ N    ┆ x[1,N]       │
    │ 2    ┆ N    ┆ x[2,N]       │
    │ 3    ┆ N    ┆ x[3,N]       │
    └──────┴──────┴──────────────┘
    >>> concat_dimensions(df, keep_dims=False)
    shape: (6, 1)
    ┌──────────────┐
    │ concated_dim │
    │ ---          │
    │ str          │
    ╞══════════════╡
    │ [1,Y]        │
    │ [2,Y]        │
    │ [3,Y]        │
    │ [1,N]        │
    │ [2,N]        │
    │ [3,N]        │
    └──────────────┘
    >>> # Properly handles cases with no dimensions and ignores reserved columns
    >>> df = pl.DataFrame({VAR_KEY: [1, 2]})
    >>> concat_dimensions(df, prefix="x")
    shape: (2, 2)
    ┌───────────────┬──────────────┐
    │ __variable_id ┆ concated_dim │
    │ ---           ┆ ---          │
    │ i64           ┆ str          │
    ╞═══════════════╪══════════════╡
    │ 1             ┆ x            │
    │ 2             ┆ x            │
    └───────────────┴──────────────┘
    """
    if prefix is None:
        prefix = ""
    dimensions = [col for col in df.columns if col not in ignore_columns]
    if dimensions:
        pl_expr = pl.concat_str(
            pl.lit(prefix + "["),
            pl.concat_str(*dimensions, separator=","),
            pl.lit("]"),
        )
    else:
        pl_expr = pl.lit(prefix)

    df = df.with_columns(concated_dim=pl_expr)

    if replace_spaces:
        df = df.with_columns(pl.col("concated_dim").str.replace_all(" ", "_"))

    if keep_dims:
        return df
    else:
        return df.drop(*dimensions)


def get_dimensions(df: pl.DataFrame) -> List[str]:
    """
    Returns the dimensions of the DataFrame. Reserved columns do not count as dimensions.

    Examples
    --------
    >>> import pandas as pd
    >>> get_dimensions(pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}))
    ['x', 'y']
    """
    return [col for col in df.columns if col not in RESERVED_COL_KEYS]
