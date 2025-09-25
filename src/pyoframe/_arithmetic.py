"""Defines helper functions for doing arithmetic operations on expressions (e.g. addition)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from pyoframe._constants import (
    COEF_KEY,
    CONST_TERM,
    KEY_TYPE,
    QUAD_VAR_KEY,
    RESERVED_COL_KEYS,
    VAR_KEY,
    Config,
    PyoframeError,
    UnmatchedStrategy,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe._core import Expression


def multiply(self: Expression, other: Expression) -> Expression:
    """Multiplies two expressions together.

    Examples:
        >>> import pyoframe as pf
        >>> m = pf.Model()
        >>> m.x1 = pf.Variable()
        >>> m.x2 = pf.Variable()
        >>> m.x3 = pf.Variable()
        >>> result = 5 * m.x1 * m.x2
        >>> result
        <Expression terms=1 type=quadratic>
        5 x2 * x1
        >>> result * m.x3
        Traceback (most recent call last):
        ...
        pyoframe._constants.PyoframeError: Cannot multiply the two expressions below because the result would be a cubic. Only quadratic or linear expressions are allowed.
        Expression 1 (quadratic):   ((5 * x1) * x2)
        Expression 2 (linear):      x3
    """
    self_degree, other_degree = self.degree(), other.degree()
    product_degree = self_degree + other_degree
    if product_degree > 2:
        assert product_degree <= 4, (
            "Unexpected because expressions should not exceed degree 2."
        )
        res_type = "cubic" if product_degree == 3 else "quartic"
        raise PyoframeError(
            f"""Cannot multiply the two expressions below because the result would be a {res_type}. Only quadratic or linear expressions are allowed.
Expression 1 ({self.degree(return_str=True)}):\t{self.name}
Expression 2 ({other.degree(return_str=True)}):\t{other.name}"""
        )

    if self_degree == 1 and other_degree == 1:
        return _quadratic_multiplication(self, other)

    # save names to use in debug messages before any swapping occurs
    self_name, other_name = self.name, other.name
    if self_degree < other_degree:
        self, other = other, self
        self_degree, other_degree = other_degree, self_degree

    assert other_degree == 0, (
        "This should always be true since other cases have already been handled."
    )

    # QUAD_VAR_KEY doesn't need to be dropped since we know it doesn't exist
    multiplier = other.data.drop(VAR_KEY)

    dims = self._dimensions_unsafe
    other_dims = other._dimensions_unsafe
    dims_in_common = [dim for dim in dims if dim in other_dims]

    data = (
        self.data.join(
            multiplier,
            on=dims_in_common if len(dims_in_common) > 0 else None,
            how="inner" if dims_in_common else "cross",
            maintain_order=(
                "left" if Config.maintain_order and dims_in_common else None
            ),
        )
        .with_columns(pl.col(COEF_KEY) * pl.col(COEF_KEY + "_right"))
        .drop(COEF_KEY + "_right")
    )

    return self._new(data, name=f"({self_name} * {other_name})")


def _quadratic_multiplication(self: Expression, other: Expression) -> Expression:
    """Multiplies two expressions of degree 1.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"dim": [1, 2, 3], "value": [1, 2, 3]})
        >>> m = pf.Model()
        >>> m.x1 = pf.Variable()
        >>> m.x2 = pf.Variable()
        >>> expr1 = df * m.x1
        >>> expr2 = df * m.x2 * 2 + 4
        >>> expr1 * expr2
        <Expression height=3 terms=6 type=quadratic>
        ┌─────┬───────────────────┐
        │ dim ┆ expression        │
        │ (3) ┆                   │
        ╞═════╪═══════════════════╡
        │ 1   ┆ 4 x1 +2 x2 * x1   │
        │ 2   ┆ 8 x1 +8 x2 * x1   │
        │ 3   ┆ 12 x1 +18 x2 * x1 │
        └─────┴───────────────────┘
        >>> (expr1 * expr2) - df * m.x1 * df * m.x2 * 2
        <Expression height=3 terms=3 type=linear>
        ┌─────┬────────────┐
        │ dim ┆ expression │
        │ (3) ┆            │
        ╞═════╪════════════╡
        │ 1   ┆ 4 x1       │
        │ 2   ┆ 8 x1       │
        │ 3   ┆ 12 x1      │
        └─────┴────────────┘

    """
    dims = self._dimensions_unsafe
    other_dims = other._dimensions_unsafe
    dims_in_common = [dim for dim in dims if dim in other_dims]

    data = (
        self.data.join(
            other.data,
            on=dims_in_common if len(dims_in_common) > 0 else None,
            how="inner" if dims_in_common else "cross",
            maintain_order=(
                "left" if Config.maintain_order and dims_in_common else None
            ),
        )
        .with_columns(pl.col(COEF_KEY) * pl.col(COEF_KEY + "_right"))
        .drop(COEF_KEY + "_right")
        .rename({VAR_KEY + "_right": QUAD_VAR_KEY})
        # Swap VAR_KEY and QUAD_VAR_KEY so that VAR_KEY is always the larger one
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

    return self._new(data, name=f"({self.name} * {other.name})")


def add(*expressions: Expression) -> Expression:
    """Add multiple expressions together."""
    # Mapping of how a sum of two expressions should propagate the unmatched strategy
    propagation_strategies = {
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
            expr._unmatched_strategy
            not in (UnmatchedStrategy.KEEP, UnmatchedStrategy.UNSET)
            for expr in expressions
        )
    else:
        requires_join = any(
            expr._unmatched_strategy != UnmatchedStrategy.KEEP for expr in expressions
        )

    has_dim_conflict = any(
        sorted(dims) != sorted(expr._dimensions_unsafe) for expr in expressions[1:]
    )

    # If we cannot use .concat compute the sum in a pairwise manner
    if len(expressions) > 2 and (has_dim_conflict or requires_join):
        result = expressions[0]
        for expr in expressions[1:]:
            result = add(result, expr)
        return result

    if has_dim_conflict:
        assert len(expressions) == 2

        left, right = expressions[0], expressions[1]
        left_dims, right_dims = left._dimensions_unsafe, right._dimensions_unsafe

        missing_left = [dim for dim in right_dims if dim not in left_dims]
        missing_right = [dim for dim in left_dims if dim not in right_dims]
        common_dims = [dim for dim in left_dims if dim in right_dims]

        if not (
            set(missing_left) <= set(left._allowed_new_dims)
            and set(missing_right) <= set(right._allowed_new_dims)
        ):
            _raise_addition_error(
                left,
                right,
                f"their\n\tdimensions are different ({left_dims} != {right_dims})",
                "If this is intentional, use .over(…) to broadcast. Learn more at\n\thttps://bravos-power.github.io/pyoframe/learn/concepts/special-functions/#adding-expressions-with-differing-dimensions-using-over",
            )

        left_old = left
        if missing_left:
            left = _broadcast(left, right, common_dims, missing_left)
        if missing_right:
            right = _broadcast(
                right, left_old, common_dims, missing_right, swapped=True
            )

        assert sorted(left._dimensions_unsafe) == sorted(right._dimensions_unsafe)
        expressions = (left, right)

    dims = expressions[0]._dimensions_unsafe
    # Check no dims conflict
    assert all(
        sorted(dims) == sorted(expr._dimensions_unsafe) for expr in expressions[1:]
    )
    if requires_join:
        assert len(expressions) == 2
        assert dims != []
        left, right = expressions[0], expressions[1]

        # Order so that drop always comes before keep, and keep always comes before default
        if swap := (
            (left._unmatched_strategy, right._unmatched_strategy)
            in (
                (UnmatchedStrategy.UNSET, UnmatchedStrategy.DROP),
                (UnmatchedStrategy.UNSET, UnmatchedStrategy.KEEP),
                (UnmatchedStrategy.KEEP, UnmatchedStrategy.DROP),
            )
        ):
            left, right = right, left

        def get_indices(expr):
            return expr.data.select(dims).unique(maintain_order=Config.maintain_order)

        left_data, right_data = left.data, right.data

        strat = (left._unmatched_strategy, right._unmatched_strategy)

        propagate_strat = propagation_strategies[strat]  # type: ignore

        if strat == (UnmatchedStrategy.DROP, UnmatchedStrategy.DROP):
            left_data = left.data.join(
                get_indices(right),
                on=dims,
                maintain_order="left" if Config.maintain_order else None,
            )
            right_data = right.data.join(
                get_indices(left),
                on=dims,
                maintain_order="left" if Config.maintain_order else None,
            )
        elif strat == (UnmatchedStrategy.UNSET, UnmatchedStrategy.UNSET):
            assert not Config.disable_unmatched_checks, (
                "This code should not be reached when unmatched checks are disabled."
            )
            outer_join = get_indices(left).join(
                get_indices(right),
                how="full",
                on=dims,
                maintain_order="left_right" if Config.maintain_order else None,
            )
            if outer_join.get_column(dims[0]).null_count() > 0:
                unmatched_vals = outer_join.filter(
                    outer_join.get_column(dims[0]).is_null()
                )
                _raise_unmatched_values_error(left, right, unmatched_vals, swap)
            if outer_join.get_column(dims[0] + "_right").null_count() > 0:
                unmatched_vals = outer_join.filter(
                    outer_join.get_column(dims[0] + "_right").is_null()
                )
                _raise_unmatched_values_error(left, right, unmatched_vals, swap)

        elif strat == (UnmatchedStrategy.DROP, UnmatchedStrategy.KEEP):
            left_data = get_indices(right).join(
                left.data,
                how="left",
                on=dims,
                maintain_order="left" if Config.maintain_order else None,
            )
        elif strat == (UnmatchedStrategy.DROP, UnmatchedStrategy.UNSET):
            left_data = get_indices(right).join(
                left.data,
                how="left",
                on=dims,
                maintain_order="left" if Config.maintain_order else None,
            )
            if left_data.get_column(COEF_KEY).null_count() > 0:
                _raise_unmatched_values_error(
                    left,
                    right,
                    left_data.filter(left_data.get_column(COEF_KEY).is_null()),
                    swap,
                )

        elif strat == (UnmatchedStrategy.KEEP, UnmatchedStrategy.UNSET):
            assert not Config.disable_unmatched_checks, (
                "This code should not be reached when unmatched checks are disabled."
            )
            unmatched = right.data.join(get_indices(left), how="anti", on=dims)
            if len(unmatched) > 0:
                _raise_unmatched_values_error(left, right, unmatched, swap)
        else:  # pragma: no cover
            assert False, "This code should've never been reached!"

        expr_data = [left_data, right_data]
    else:
        propagate_strat = expressions[0]._unmatched_strategy
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

    full_name = expressions[0].name
    for expr in expressions[1:]:
        name = expr.name
        full_name += f" - {name[1:]}" if name[0] == "-" else f" + {name}"

    new_expr = expressions[0]._new(data, name=f"({full_name})")
    new_expr._unmatched_strategy = propagate_strat

    return new_expr


def _raise_unmatched_values_error(
    left: Expression, right: Expression, unmatched_values: pl.DataFrame, swapped: bool
):
    if swapped:
        left, right = right, left

    _raise_addition_error(
        left,
        right,
        "of unmatched values",
        f"Unmatched values:\n{unmatched_values}\nIf this is intentional, use .drop_extras() or .keep_extras().",
    )


def _raise_addition_error(
    left: Expression, right: Expression, reason: str, postfix: str
):
    op = "add"
    right_name = right.name
    if right_name[0] == "-":
        op = "subtract"
        right_name = right_name[1:]
    raise PyoframeError(
        f"""Cannot {op} the two expressions below because {reason}.
Expression 1:\t{left.name}
Expression 2:\t{right_name}
{postfix}
"""
    )


# TODO consider returning a dataframe instead of an expression to simplify code (e.g. avoid copy_flags)
def _broadcast(
    self: Expression,
    target: Expression,
    common_dims: list[str],
    missing_dims: list[str],
    swapped: bool = False,
) -> Expression:
    target_data = target.data.select(target._dimensions_unsafe).unique(
        maintain_order=Config.maintain_order
    )

    if not common_dims:
        res = self._new(self.data.join(target_data, how="cross"), name=self.name)
        res._copy_flags(self)
        return res

    # If drop, we just do an inner join to get into the shape of the other
    if self._unmatched_strategy == UnmatchedStrategy.DROP:
        res = self._new(
            self.data.join(
                target_data,
                on=common_dims,
                maintain_order="left" if Config.maintain_order else None,
            ),
            name=self.name,
        )
        res._copy_flags(self)
        return res

    result = self.data.join(
        target_data,
        on=common_dims,
        how="left",
        maintain_order="left" if Config.maintain_order else None,
    )
    right_has_missing = result.get_column(missing_dims[0]).null_count() > 0
    if right_has_missing:
        _raise_unmatched_values_error(
            self,
            target,
            result.filter(result.get_column(missing_dims[0]).is_null()),
            swapped,
        )
    res = self._new(result, self.name)
    res._copy_flags(self)
    return res


def _sum_like_terms(df: pl.DataFrame) -> pl.DataFrame:
    """Combines terms with the same variables."""
    dims = [c for c in df.columns if c not in RESERVED_COL_KEYS]
    var_cols = [VAR_KEY] + ([QUAD_VAR_KEY] if QUAD_VAR_KEY in df.columns else [])
    df = df.group_by(dims + var_cols, maintain_order=Config.maintain_order).sum()
    return df


def _simplify_expr_df(df: pl.DataFrame) -> pl.DataFrame:
    """Removes the quadratic column and terms with a zero coefficient, when applicable.

    Specifically, zero coefficient terms are always removed, except if they're the only terms in which case the expression contains a single term.
    The quadratic column is removed if the expression is not a quadratic.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame(
        ...     {
        ...         VAR_KEY: [CONST_TERM, 1],
        ...         QUAD_VAR_KEY: [CONST_TERM, 1],
        ...         COEF_KEY: [1.0, 0],
        ...     }
        ... )
        >>> _simplify_expr_df(df)
        shape: (1, 2)
        ┌───────────────┬─────────┐
        │ __variable_id ┆ __coeff │
        │ ---           ┆ ---     │
        │ i64           ┆ f64     │
        ╞═══════════════╪═════════╡
        │ 0             ┆ 1.0     │
        └───────────────┴─────────┘
        >>> df = pl.DataFrame(
        ...     {
        ...         "t": [1, 1, 2, 2, 3, 3],
        ...         VAR_KEY: [CONST_TERM, 1, CONST_TERM, 1, 1, 2],
        ...         QUAD_VAR_KEY: [
        ...             CONST_TERM,
        ...             CONST_TERM,
        ...             CONST_TERM,
        ...             CONST_TERM,
        ...             CONST_TERM,
        ...             1,
        ...         ],
        ...         COEF_KEY: [1, 0, 0, 0, 1, 0],
        ...     }
        ... )
        >>> _simplify_expr_df(df)
        shape: (3, 3)
        ┌─────┬───────────────┬─────────┐
        │ t   ┆ __variable_id ┆ __coeff │
        │ --- ┆ ---           ┆ ---     │
        │ i64 ┆ i64           ┆ i64     │
        ╞═════╪═══════════════╪═════════╡
        │ 1   ┆ 0             ┆ 1       │
        │ 2   ┆ 0             ┆ 0       │
        │ 3   ┆ 1             ┆ 1       │
        └─────┴───────────────┴─────────┘
    """
    df_filtered = df.filter(pl.col(COEF_KEY) != 0)
    if len(df_filtered) < len(df):
        dims = [c for c in df.columns if c not in RESERVED_COL_KEYS]
        if dims:
            dim_values = df.select(dims).unique(maintain_order=Config.maintain_order)
            df = (
                dim_values.join(
                    df_filtered,
                    on=dims,
                    how="left",
                    maintain_order="left" if Config.maintain_order else None,
                )
                .with_columns(pl.col(COEF_KEY).fill_null(0))
                .fill_null(CONST_TERM)
            )
        else:
            df = df_filtered
            if df.is_empty():
                df = pl.DataFrame(
                    {VAR_KEY: [CONST_TERM], COEF_KEY: [0]},
                    schema={VAR_KEY: KEY_TYPE, COEF_KEY: pl.Float64},
                )

    if QUAD_VAR_KEY in df.columns and (df.get_column(QUAD_VAR_KEY) == CONST_TERM).all():
        df = df.drop(QUAD_VAR_KEY)

    return df


def _get_dimensions(df: pl.DataFrame) -> list[str] | None:
    """Returns the dimensions of the DataFrame.

    Reserved columns do not count as dimensions. If there are no dimensions,
    returns `None` to force caller to handle this special case.

    Examples:
        >>> import polars as pl
        >>> _get_dimensions(pl.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}))
        ['x', 'y']
        >>> _get_dimensions(pl.DataFrame({"__variable_id": [1, 2, 3]}))
    """
    result = [col for col in df.columns if col not in RESERVED_COL_KEYS]
    return result if result else None
