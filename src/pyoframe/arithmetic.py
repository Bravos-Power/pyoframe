from typing import TYPE_CHECKING, List
import polars as pl
from polars.testing import assert_frame_equal

from pyoframe.constants import CONST_TERM, RESERVED_COL_KEYS, VAR_KEY, MissingStrategy

if TYPE_CHECKING:
    from pyoframe.constraints import Expression


def add_expressions(self: "Expression", other: "Expression") -> "Expression":
    """Adds two expressions together."""
    data, other_data = self.data, other.data
    dims, other_dims = self.dimensions, other.dimensions
    if dims is None:
        dims = []
    if other_dims is None:
        other_dims = []

    if set(dims) != set(other_dims):
        dims_missing = set(other_dims) - set(dims)
        if not dims_missing <= set(self.allowed_new_dims):
            raise ValueError(
                f"Dimensions {dims_missing} were found in the right expression but not the left. Consider using .add_dim() on the left if this is intentional."
            )
        dims_missing_in_other = set(dims) - set(other_dims)
        if not dims_missing_in_other <= set(other.allowed_new_dims):
            raise ValueError(
                f"Dimensions {dims_missing_in_other} were found in the left expression but not the right. Consider using .add_dim() on the right if this is intentional."
            )
    data = _make_like_other(
        data, other_data, self.missing_strategy, other.missing_strategy
    )
    other_data = _make_like_other(
        other_data, data, other.missing_strategy, self.missing_strategy
    )
    assert sorted(data.columns) == sorted(other_data.columns)
    other_data = other_data.select(data.columns)  # Match column order

    dims = get_dimensions(data)
    if dims is None:
        dims = []
    data = pl.concat([data, other_data], how="vertical_relaxed")
    data = data.group_by(dims + [VAR_KEY], maintain_order=True).sum()

    return self._new(data)


def _make_like_other(
    self: pl.DataFrame, other: pl.DataFrame, strategy, other_strategy
) -> pl.DataFrame:
    dims = get_dimensions(self)
    other_dims = get_dimensions(other)
    if other_dims is None:
        return self

    if dims is None:
        shared_dims = []
    else:
        shared_dims = [dim for dim in dims if dim in other_dims]

    other = other.select(other_dims).unique(maintain_order=True)

    if not shared_dims:
        return self.join(other, how="cross")

    result = self.join(other, on=shared_dims, how="outer")

    left_has_missing = result.get_column(shared_dims[0]).null_count() > 0
    if left_has_missing:
        if strategy != MissingStrategy.FILL and other_strategy != MissingStrategy.DROP:
            raise ValueError(
                "Missing values found in the left expression. Consider using left.fill_missing() or right.drop_missing()"
            )
        result = result.drop_nulls(shared_dims[0])

    right_has_missing = result.get_column(shared_dims[0] + "_right").null_count() > 0
    if right_has_missing:
        if strategy == MissingStrategy.DROP:
            result = result.drop_nulls(shared_dims[0] + "_right")
        elif other_strategy != MissingStrategy.FILL:
            raise ValueError(
                "Missing values found in the right expression. Consider using right.fill_missing() or left.drop_missing()"
            )
        else:
            ...

    result = result.with_columns(
        *(pl.coalesce(d, d + "_right").alias(d) for d in shared_dims)
    ).drop([d + "_right" for d in shared_dims])
    return result


def get_dimensions(df: pl.DataFrame) -> List[str] | None:
    """
    Returns the dimensions of the DataFrame. Reserved columns do not count as dimensions.
    If there are no dimensions, returns None to force caller to handle this special case.

    Examples
    --------
    >>> import polars as pl
    >>> get_dimensions(pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}))
    ['x', 'y']
    >>> get_dimensions(pl.DataFrame({"__variable_id": [1, 2, 3]}))

    """
    res = [col for col in df.columns if col not in RESERVED_COL_KEYS]
    return res if res else None
