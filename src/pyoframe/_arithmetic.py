"""
Defines helper functions for doing arithmetic operations on expressions (e.g. addition).
"""

from typing import TYPE_CHECKING, List, Optional

import polars as pl

from pyoframe.constants import (
    COEF_KEY,
    CONST_TERM,
    KEY_TYPE,
    POLARS_VERSION,
    QUAD_VAR_KEY,
    RESERVED_COL_KEYS,
    VAR_KEY,
    Config,
    PyoframeError,
    UnmatchedStrategy,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.core import Expression


def _multiply_expressions(self: "Expression", other: "Expression") -> "Expression":
    """
    Multiply two or more expressions together.

    Examples:
        >>> import pyoframe as pf
        >>> m = pf.Model("min")
        >>> m.x1 = pf.Variable()
        >>> m.x2 = pf.Variable()
        >>> m.x3 = pf.Variable()
        >>> result = 5 * m.x1 * m.x2
        >>> result
        <Expression size=1 dimensions={} terms=1 degree=2>
        5 x2 * x1
        >>> result * m.x3
        Traceback (most recent call last):
        ...
        pyoframe.constants.PyoframeError: Failed to multiply expressions:
        <Expression size=1 dimensions={} terms=1 degree=2> * <Expression size=1 dimensions={} terms=1>
        Due to error:
        Cannot multiply a quadratic expression by a non-constant.
    """
    try:
        return _multiply_expressions_core(self, other)
    except PyoframeError as error:
        raise PyoframeError(
            "Failed to multiply expressions:\n"
            + " * ".join(
                e.to_str(include_header=True, include_data=False) for e in [self, other]
            )
            + "\nDue to error:\n"
            + str(error)
        ) from error


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


def _multiply_expressions_core(self: "Expression", other: "Expression") -> "Expression":
    self_degree, other_degree = self.degree(), other.degree()
    if self_degree + other_degree > 2:
        # We know one of the two must be a quadratic since 1 + 1 is not greater than 2.
        raise PyoframeError("Cannot multiply a quadratic expression by a non-constant.")
    if self_degree < other_degree:
        self, other = other, self
        self_degree, other_degree = other_degree, self_degree
    if other_degree == 1:
        assert (
            self_degree == 1
        ), "This should always be true since the sum of degrees must be <=2."
        return _quadratic_multiplication(self, other)

    assert (
        other_degree == 0
    ), "This should always be true since other cases have already been handled."
    multiplier = other.data.drop(
        VAR_KEY
    )  # QUAD_VAR_KEY doesn't need to be dropped since we know it doesn't exist

    dims = self.dimensions_unsafe
    other_dims = other.dimensions_unsafe
    dims_in_common = [dim for dim in dims if dim in other_dims]

    data = (
        self.data.join(
            multiplier,
            on=dims_in_common if len(dims_in_common) > 0 else None,
            how="inner" if dims_in_common else "cross",
        )
        .with_columns(pl.col(COEF_KEY) * pl.col(COEF_KEY + "_right"))
        .drop(COEF_KEY + "_right")
    )

    return self._new(data)


def _quadratic_multiplication(self: "Expression", other: "Expression") -> "Expression":
    """
    Multiply two expressions of degree 1.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"dim": [1, 2, 3], "value": [1, 2, 3]})
        >>> m = pf.Model()
        >>> m.x1 = pf.Variable()
        >>> m.x2 = pf.Variable()
        >>> expr1 = df * m.x1
        >>> expr2 = df * m.x2 * 2 + 4
        >>> expr1 * expr2
        <Expression size=3 dimensions={'dim': 3} terms=6 degree=2>
        [1]: 4 x1 +2 x2 * x1
        [2]: 8 x1 +8 x2 * x1
        [3]: 12 x1 +18 x2 * x1
        >>> (expr1 * expr2) - df * m.x1 * df * m.x2 * 2
        <Expression size=3 dimensions={'dim': 3} terms=3>
        [1]: 4 x1
        [2]: 8 x1
        [3]: 12 x1
    """
    dims = self.dimensions_unsafe
    other_dims = other.dimensions_unsafe
    dims_in_common = [dim for dim in dims if dim in other_dims]

    data = (
        self.data.join(
            other.data,
            on=dims_in_common if len(dims_in_common) > 0 else None,
            how="inner" if dims_in_common else "cross",
        )
        .with_columns(pl.col(COEF_KEY) * pl.col(COEF_KEY + "_right"))
        .drop(COEF_KEY + "_right")
        .rename({VAR_KEY + "_right": QUAD_VAR_KEY})
        # Swap VAR_KEY and QUAD_VAR_KEY so that VAR_KEy is always the larger one
        .with_columns(
            pl.when(pl.col(VAR_KEY) < pl.col(QUAD_VAR_KEY))
            .then(pl.col(QUAD_VAR_KEY))
            .otherwise(pl.col(VAR_KEY))
            .alias(VAR_KEY),
            pl.when(pl.col(VAR_KEY) < pl.col(QUAD_VAR_KEY))
            .then(pl.col(VAR_KEY))
            .otherwise(pl.col(QUAD_VAR_KEY))
            .alias(QUAD_VAR_KEY),
        )
    )

    data = _sum_like_terms(data)

    return self._new(data)


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
                get_indices(right),
                how="full" if POLARS_VERSION.major >= 1 else "outer",
                on=dims,
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

    # Add quadratic column if it is needed and doesn't already exist
    if any(QUAD_VAR_KEY in df.columns for df in expr_data):
        expr_data = [
            (
                df.with_columns(pl.lit(CONST_TERM).alias(QUAD_VAR_KEY).cast(KEY_TYPE))
                if QUAD_VAR_KEY not in df.columns
                else df
            )
            for df in expr_data
        ]

    # Sort columns to allow for concat
    expr_data = [
        e.select(dims + [c for c in e.columns if c not in dims]) for e in expr_data
    ]

    data = pl.concat(expr_data, how="vertical_relaxed")
    data = _sum_like_terms(data)

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
        # TODO actually suggest using e.g. .add_dim("a", "b") instead of just "use .add_dim()"
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


def _sum_like_terms(df: pl.DataFrame) -> pl.DataFrame:
    """Combines terms with the same variables. Removes quadratic column if they all happen to cancel."""
    dims = [c for c in df.columns if c not in RESERVED_COL_KEYS]
    var_cols = [VAR_KEY] + ([QUAD_VAR_KEY] if QUAD_VAR_KEY in df.columns else [])
    df = (
        df.group_by(dims + var_cols, maintain_order=True)
        .sum()
        .filter(pl.col(COEF_KEY) != 0)
    )
    if QUAD_VAR_KEY in df.columns and (df.get_column(QUAD_VAR_KEY) == CONST_TERM).all():
        df = df.drop(QUAD_VAR_KEY)
    return df


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
