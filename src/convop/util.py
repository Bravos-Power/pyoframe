import polars as pl


def align_and_concat(
    left: pl.DataFrame, right: pl.DataFrame, how: str = "vertical_relaxed"
) -> pl.DataFrame:
    assert sorted(left.columns) == sorted(right.columns)
    right = right.select(left.columns)
    return pl.concat([left, right], how=how)
