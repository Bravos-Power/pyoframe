from typing import TYPE_CHECKING, List, Optional
import polars as pl

from pyoframe.constants import (
    COEF_KEY,
    RESERVED_COL_KEYS,
    VAR_KEY,
    UnmatchedStrategy,
    Config,
    PyoframeError,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.core import Expression


def _add_expressions(*expressions: "Expression") -> "Expression":
    try:
        return _add_expressions_core(*expressions)
    except PyoframeError as error:
        raise PyoframeError(
            "Failed to add expressions:\n"
            + " + ".join(
                e.to_str(include_header=True, include_data=False) for e in expressions
            )
            + "\nDue to error:\n"
            + str(error)
        ) from error


def _add_expressions_core(*expressions: "Expression") -> "Expression":
    # Mapping of how a sum of two expressions should propogate the unmatched strategy
    propogatation_strategies = {
        (UnmatchedStrategy.DROP, UnmatchedStrategy.DROP): UnmatchedStrategy.DROP,
        (
            UnmatchedStrategy.UNSET,
            UnmatchedStrategy.UNSET,
        ): UnmatchedStrategy.UNSET,
        (UnmatchedStrategy.KEEP, UnmatchedStrategy.KEEP): UnmatchedStrategy.KEEP,
        (UnmatchedStrategy.DROP, UnmatchedStrategy.KEEP): UnmatchedStrategy.UNSET,
        (UnmatchedStrategy.DROP, UnmatchedStrategy.UNSET): UnmatchedStrategy.DROP,
        (UnmatchedStrategy.KEEP, UnmatchedStrategy.UNSET): UnmatchedStrategy.KEEP,
    }

    assert len(expressions) > 1, "Need at least two expressions to add together."

    dims = expressions[0].dimensions

    if dims is None:
        requires_join = False
        dims = []
    elif Config.disable_unmatched_checks:
        requires_join = any(
            expr.unmatched_strategy
            not in (UnmatchedStrategy.KEEP, UnmatchedStrategy.UNSET)
            for expr in expressions
        )
    else:
        requires_join = any(
            expr.unmatched_strategy != UnmatchedStrategy.KEEP for expr in expressions
        )

    has_dim_conflict = any(
        sorted(dims) != sorted(expr.dimensions_unsafe) for expr in expressions[1:]
    )

    # If we cannot use .concat compute the sum in a pairwise manner
    if len(expressions) > 2 and (has_dim_conflict or requires_join):
        result = expressions[0]
        for expr in expressions[1:]:
            result = _add_expressions_core(result, expr)
        return result

    if has_dim_conflict:
        assert len(expressions) == 2
        expressions = (
            _add_dimension(expressions[0], expressions[1]),
            _add_dimension(expressions[1], expressions[0]),
        )
        assert sorted(expressions[0].dimensions_unsafe) == sorted(
            expressions[1].dimensions_unsafe
        )

    dims = expressions[0].dimensions_unsafe
    # Check no dims conflict
    assert all(
        sorted(dims) == sorted(expr.dimensions_unsafe) for expr in expressions[1:]
    )
    if requires_join:
        assert len(expressions) == 2
        assert dims != []
        left, right = expressions[0], expressions[1]

        # Order so that drop always comes before keep, and keep always comes before default
        if (left.unmatched_strategy, right.unmatched_strategy) in (
            (UnmatchedStrategy.UNSET, UnmatchedStrategy.DROP),
            (UnmatchedStrategy.UNSET, UnmatchedStrategy.KEEP),
            (UnmatchedStrategy.KEEP, UnmatchedStrategy.DROP),
        ):
            left, right = right, left

        def get_indices(expr):
            return expr.data.select(dims).unique(maintain_order=True)

        left_data, right_data = left.data, right.data

        strat = (left.unmatched_strategy, right.unmatched_strategy)

        propogate_strat = propogatation_strategies[strat]  # type: ignore

        if strat == (UnmatchedStrategy.DROP, UnmatchedStrategy.DROP):
            left_data = left.data.join(get_indices(right), how="inner", on=dims)
            right_data = right.data.join(get_indices(left), how="inner", on=dims)
        elif strat == (UnmatchedStrategy.UNSET, UnmatchedStrategy.UNSET):
            assert (
                not Config.disable_unmatched_checks
            ), "This code should not be reached when unmatched checks are disabled."
            outer_join = get_indices(left).join(
                get_indices(right), how="outer", on=dims
            )
            if outer_join.get_column(dims[0]).null_count() > 0:
                raise PyoframeError(
                    "Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()\n"
                    + str(outer_join.filter(outer_join.get_column(dims[0]).is_null()))
                )
            if outer_join.get_column(dims[0] + "_right").null_count() > 0:
                raise PyoframeError(
                    "Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()\n"
                    + str(
                        outer_join.filter(
                            outer_join.get_column(dims[0] + "_right").is_null()
                        )
                    )
                )
        elif strat == (UnmatchedStrategy.DROP, UnmatchedStrategy.KEEP):
            left_data = get_indices(right).join(left.data, how="left", on=dims)
        elif strat == (UnmatchedStrategy.DROP, UnmatchedStrategy.UNSET):
            left_data = get_indices(right).join(left.data, how="left", on=dims)
            if left_data.get_column(COEF_KEY).null_count() > 0:
                raise PyoframeError(
                    "Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()\n"
                    + str(left_data.filter(left_data.get_column(COEF_KEY).is_null()))
                )
        elif strat == (UnmatchedStrategy.KEEP, UnmatchedStrategy.UNSET):
            assert (
                not Config.disable_unmatched_checks
            ), "This code should not be reached when unmatched checks are disabled."
            unmatched = right.data.join(get_indices(left), how="anti", on=dims)
            if len(unmatched) > 0:
                raise PyoframeError(
                    "Dataframe has unmatched values. If this is intentional, use .drop_unmatched() or .keep_unmatched()\n"
                    + str(unmatched)
                )
        else:  # pragma: no cover
            assert False, "This code should've never been reached!"

        expr_data = [left_data, right_data]
    else:
        propogate_strat = expressions[0].unmatched_strategy
        expr_data = [expr.data for expr in expressions]

    # Sort columns to allow for concat
    expr_data = [e.select(sorted(e.columns)) for e in expr_data]

    data = pl.concat(expr_data, how="vertical_relaxed")
    data = data.group_by(dims + [VAR_KEY], maintain_order=True).sum()

    new_expr = expressions[0]._new(data)
    new_expr.unmatched_strategy = propogate_strat

    return new_expr


def _add_dimension(self: "Expression", target: "Expression") -> "Expression":
    target_dims = target.dimensions
    if target_dims is None:
        return self
    dims = self.dimensions
    if dims is None:
        dims_in_common = []
        missing_dims = target_dims
    else:
        dims_in_common = [dim for dim in dims if dim in target_dims]
        missing_dims = [dim for dim in target_dims if dim not in dims]

    # We're already at the size of our target
    if not missing_dims:
        return self

    if not set(missing_dims) <= set(self.allowed_new_dims):
        raise PyoframeError(
            f"Dataframe has missing dimensions {missing_dims}. If this is intentional, use .add_dim()\n{self.data}"
        )

    target_data = target.data.select(target_dims).unique(maintain_order=True)

    if not dims_in_common:
        return self._new(self.data.join(target_data, how="cross"))

    # If drop, we just do an inner join to get into the shape of the other
    if self.unmatched_strategy == UnmatchedStrategy.DROP:
        return self._new(self.data.join(target_data, on=dims_in_common, how="inner"))

    result = self.data.join(target_data, on=dims_in_common, how="left")
    right_has_missing = result.get_column(missing_dims[0]).null_count() > 0
    if right_has_missing:
        raise PyoframeError(
            f"Cannot add dimension {missing_dims} since it contains unmatched values. If this is intentional, consider using .drop_unmatched()"
        )
    return self._new(result)


def _get_dimensions(df: pl.DataFrame) -> Optional[List[str]]:
    """
    Returns the dimensions of the DataFrame. Reserved columns do not count as dimensions.
    If there are no dimensions, returns None to force caller to handle this special case.

    Examples:
        >>> import polars as pl
        >>> _get_dimensions(pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}))
        ['x', 'y']
        >>> _get_dimensions(pl.DataFrame({"__variable_id": [1, 2, 3]}))
    """
    result = [col for col in df.columns if col not in RESERVED_COL_KEYS]
    return result if result else None
