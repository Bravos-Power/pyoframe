from typing import TYPE_CHECKING, List
import polars as pl

from pyoframe.constants import COEF_KEY, RESERVED_COL_KEYS, VAR_KEY, UnmatchedStrategy

if TYPE_CHECKING:
    from pyoframe.constraints import Expression


def add_expressions(expressions: List["Expression"]) -> "Expression":
    assert len(expressions) > 1, "Need at least two expressions to add together."

    dims = expressions[0].dimensions_unsafe
    has_dimension_conflict = any(
        sorted(expr.dimensions_unsafe) != sorted(dims) for expr in expressions[1:]
    )
    requires_join = dims and any(
        expr.unmatched_strategy != UnmatchedStrategy.KEEP for expr in expressions
    )

    # If we cannot use .concat compute the sum in a pairwise manner
    if len(expressions) > 2 and (has_dimension_conflict or requires_join):
        return sum(expressions)  # type: ignore

    if has_dimension_conflict:
        assert len(expressions) == 2
        expressions = [
            _add_dimension(expressions[0], expressions[1]),
            _add_dimension(expressions[1], expressions[0]),
        ]
        assert sorted(expressions[0].dimensions_unsafe) == sorted(
            expressions[1].dimensions_unsafe
        )

    if requires_join:
        assert len(expressions) == 2
        left, right = expressions[0], expressions[1]
        dims = left.dimensions
        assert dims is not None
        assert sorted(dims) == sorted(right.dimensions_unsafe)

        # Order so that drop always comes before keep, and keep always comes before error
        if (left.unmatched_strategy, right.unmatched_strategy) in (
            (UnmatchedStrategy.ERROR, UnmatchedStrategy.DROP),
            (UnmatchedStrategy.ERROR, UnmatchedStrategy.KEEP),
            (UnmatchedStrategy.KEEP, UnmatchedStrategy.DROP),
        ):
            left, right = right, left

        def get_indices(expr):
            return expr.data.select(dims).unique(maintain_order=True)

        left_data, right_data = left.data, right.data

        match (left.unmatched_strategy, right.unmatched_strategy):
            case (UnmatchedStrategy.DROP, UnmatchedStrategy.DROP):
                left_data = left.data.join(get_indices(right), how="inner", on=dims)
                right_data = right.data.join(get_indices(left), how="inner", on=dims)
            case (UnmatchedStrategy.ERROR, UnmatchedStrategy.ERROR):
                outer_join = get_indices(left).join(
                    get_indices(right), how="outer", on=dims
                )
                if (
                    outer_join.get_column(dims[0]).null_count() > 0
                    or outer_join.get_column(dims[0] + "_right").null_count() > 0
                ):
                    raise ValueError(
                        "Dataframe has unmatched values. If this is intentional, consider using .drop_unmatched() or .keep_unmatched()"
                    )
            case (UnmatchedStrategy.DROP, UnmatchedStrategy.KEEP):
                left_data = get_indices(right).join(left.data, how="left", on=dims)
            case (UnmatchedStrategy.DROP, UnmatchedStrategy.ERROR):
                left_data = get_indices(right).join(left.data, how="left", on=dims)
                if left_data.get_column(COEF_KEY).null_count() > 0:
                    raise ValueError(
                        "Cannot add expressions with unmatched values. Consider using .drop_unmatched() or .keep_unmatched()"
                    )
            case _:
                assert False, "This code should've never been reached!"

        expr_data = (left_data, right_data)
    else:
        expr_data = (expr.data for expr in expressions)

    data = pl.concat(expr_data, how="vertical_relaxed")
    data = data.group_by(dims + [VAR_KEY], maintain_order=True).sum()
    return expressions[0]._new(data)


def _add_dimension(self: "Expression", target: "Expression") -> "Expression":
    target_dims = target.dimensions_unsafe
    dims = self.dimensions_unsafe
    dims_in_common = [dim for dim in dims if dim in target_dims]
    missing_dims = set(target_dims) - set(dims)

    # We're already at the size of our target
    if not missing_dims:
        return self

    if not set(missing_dims) <= set(self.allowed_new_dims):
        raise ValueError(
            f"Dimensions {missing_dims} cannot be added to the expression with dimensions {dims}. Consider using .add_dim() if this is intentional."
        )

    target_data = target.data.select(target_dims).unique(maintain_order=True)

    if not dims_in_common:
        return self._new(self.data.join(target_data, how="cross"))

    # If both are drop, we only care about the inner join
    if (self.unmatched_strategy, target.unmatched_strategy) == (
        UnmatchedStrategy.DROP,
        UnmatchedStrategy.DROP,
    ):
        return self._new(self.data.join(target_data, on=dims_in_common, how="inner"))

    result = self.data.join(target_data, on=dims_in_common, how="left")
    right_has_missing = result.get_column(dims_in_common[0] + "_right").null_count() > 0
    if right_has_missing:
        if self.unmatched_strategy == UnmatchedStrategy.DROP:
            result = result.drop_nulls(dims_in_common[0] + "_right")
        else:
            raise ValueError(
                f"Cannot add dimension {missing_dims} since it contains unmatched values. Consider using .drop_unmatched()"
            )
    return self._new(result)


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
    result = [col for col in df.columns if col not in RESERVED_COL_KEYS]
    return result if result else None
