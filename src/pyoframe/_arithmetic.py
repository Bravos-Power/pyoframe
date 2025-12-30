"""Defines helper functions for doing arithmetic operations on expressions (e.g. addition)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from pyoframe._constants import (
    COEF_KEY,
    CONST_TERM,
    QUAD_VAR_KEY,
    RESERVED_COL_KEYS,
    VAR_KEY,
    Config,
    ExtrasStrategy,
    PyoframeError,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe._core import Expression

# Mapping of how a sum of two expressions should propagate the extras strategy
_extras_propagation_rules = {
    (ExtrasStrategy.DROP, ExtrasStrategy.DROP): ExtrasStrategy.DROP,
    (ExtrasStrategy.UNSET, ExtrasStrategy.UNSET): ExtrasStrategy.UNSET,
    (ExtrasStrategy.KEEP, ExtrasStrategy.KEEP): ExtrasStrategy.KEEP,
    (ExtrasStrategy.DROP, ExtrasStrategy.KEEP): ExtrasStrategy.UNSET,
    (ExtrasStrategy.DROP, ExtrasStrategy.UNSET): ExtrasStrategy.DROP,
    (ExtrasStrategy.KEEP, ExtrasStrategy.UNSET): ExtrasStrategy.KEEP,
}


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
        <Expression (quadratic) terms=1>
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
        <Expression (quadratic) height=3 terms=6>
        ┌─────┬───────────────────┐
        │ dim ┆ expression        │
        │ (3) ┆                   │
        ╞═════╪═══════════════════╡
        │ 1   ┆ 4 x1 +2 x2 * x1   │
        │ 2   ┆ 8 x1 +8 x2 * x1   │
        │ 3   ┆ 12 x1 +18 x2 * x1 │
        └─────┴───────────────────┘
        >>> (expr1 * expr2) - df * m.x1 * df * m.x2 * 2
        <Expression (linear) height=3 terms=3>
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
    assert len(expressions) > 1, "Need at least two expressions to add together."

    if Config.disable_extras_checks:
        no_checks_strats = (ExtrasStrategy.KEEP, ExtrasStrategy.UNSET)
    else:
        no_checks_strats = (ExtrasStrategy.KEEP,)

    no_extras_checks_required = (
        all(expr._extras_strategy in no_checks_strats for expr in expressions)
        # if only one dimensioned, then there is no such thing as extra labels,
        # labels will be set by the only dimensioned expression
        or sum(not expr.dimensionless for expr in expressions) <= 1
    )

    has_dim_conflict = any(
        sorted(expressions[0]._dimensions_unsafe) != sorted(expr._dimensions_unsafe)
        for expr in expressions[1:]
    )

    # If we cannot use .concat compute the sum in a pairwise manner, so far nobody uses this code
    if len(expressions) > 2:  # pragma: no cover
        assert False, "This code has not been tested."
        if has_dim_conflict or not no_extras_checks_required:
            result = expressions[0]
            for expr in expressions[1:]:
                result = add(result, expr)
            return result
        propagate_strat = expressions[0]._extras_strategy
        dims = expressions[0]._dimensions_unsafe
        expr_data = [expr.data for expr in expressions]
    else:
        left, right = expressions[0], expressions[1]

        if has_dim_conflict:
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
                    "If this is intentional, use .over(…) to broadcast. Learn more at\n\thttps://bravos-power.github.io/pyoframe/latest/learn/concepts/addition/#adding-expressions-with-differing-dimensions-using-over",
                )

            left_old = left
            if missing_left:
                left = _broadcast(left, right, common_dims, missing_left)
            if missing_right:
                right = _broadcast(
                    right, left_old, common_dims, missing_right, swapped=True
                )

            assert sorted(left._dimensions_unsafe) == sorted(right._dimensions_unsafe)

        dims = left._dimensions_unsafe

        if not no_extras_checks_required:
            expr_data, propagate_strat = _handle_extra_labels(left, right, dims)
        else:
            propagate_strat = left._extras_strategy
            expr_data = (left.data, right.data)

    # Add quadratic column if it is needed and doesn't already exist
    if any(QUAD_VAR_KEY in df.columns for df in expr_data):
        expr_data = [
            (
                df.with_columns(
                    pl.lit(CONST_TERM).alias(QUAD_VAR_KEY).cast(Config.id_dtype)
                )
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
    new_expr._extras_strategy = propagate_strat

    return new_expr


def _handle_extra_labels(
    left: Expression, right: Expression, dims: list[str]
) -> tuple[tuple[pl.DataFrame, pl.DataFrame], ExtrasStrategy]:
    assert dims != []
    # Order so that drop always comes before keep, and keep always comes before default
    if swapped := (
        (left._extras_strategy, right._extras_strategy)
        in (
            (ExtrasStrategy.UNSET, ExtrasStrategy.DROP),
            (ExtrasStrategy.UNSET, ExtrasStrategy.KEEP),
            (ExtrasStrategy.KEEP, ExtrasStrategy.DROP),
        )
    ):
        left, right = right, left

    def get_labels(expr):
        return expr.data.select(dims).unique(maintain_order=Config.maintain_order)

    left_data, right_data = left.data, right.data

    strat = (left._extras_strategy, right._extras_strategy)

    if strat == (ExtrasStrategy.DROP, ExtrasStrategy.DROP):
        left_data = left.data.join(
            get_labels(right),
            on=dims,
            maintain_order="left" if Config.maintain_order else None,
        )
        right_data = right.data.join(
            get_labels(left),
            on=dims,
            maintain_order="left" if Config.maintain_order else None,
        )
    elif strat == (ExtrasStrategy.UNSET, ExtrasStrategy.UNSET):
        assert not Config.disable_extras_checks, (
            "This code should not be reached when checks for extra values are disabled."
        )
        left_labels, right_labels = get_labels(left), get_labels(right)
        left_extras = left_labels.join(right_labels, how="anti", on=dims)
        right_extras = right_labels.join(left_labels, how="anti", on=dims)
        if len(left_extras) > 0:
            _raise_extras_error(
                left, right, left_extras, swapped, extras_on_right=False
            )
        if len(right_extras) > 0:
            _raise_extras_error(left, right, right_extras, swapped)

    elif strat == (ExtrasStrategy.DROP, ExtrasStrategy.KEEP):
        left_data = get_labels(right).join(
            left.data,
            on=dims,
            maintain_order="left" if Config.maintain_order else None,
        )
    elif strat == (ExtrasStrategy.DROP, ExtrasStrategy.UNSET):
        right_labels = get_labels(right)
        left_data = right_labels.join(
            left.data,
            how="left",
            on=dims,
            maintain_order="left" if Config.maintain_order else None,
        )
        if left_data.get_column(COEF_KEY).null_count() > 0:
            _raise_extras_error(
                left,
                right,
                right_labels.join(get_labels(left), how="anti", on=dims),
                swapped,
            )

    elif strat == (ExtrasStrategy.KEEP, ExtrasStrategy.UNSET):
        assert not Config.disable_extras_checks, (
            "This code should not be reached when checks for extra values are disabled."
        )
        extras = right.data.join(get_labels(left), how="anti", on=dims)
        if len(extras) > 0:
            _raise_extras_error(left, right, extras.select(dims), swapped)
    else:  # pragma: no cover
        assert False, "This code should've never been reached!"

    if swapped:
        left_data, right_data = right_data, left_data

    return (left_data, right_data), _extras_propagation_rules[strat]


def _raise_extras_error(
    left: Expression,
    right: Expression,
    extra_labels: pl.DataFrame,
    swapped: bool,
    extras_on_right: bool = True,
):
    if swapped:
        left, right = right, left
        extras_on_right = not extras_on_right

    expression_num = 2 if extras_on_right else 1

    with Config.print_polars_config:
        _raise_addition_error(
            left,
            right,
            f"expression {expression_num} has extra labels",
            f"Extra labels in expression {expression_num}:\n{extra_labels}\nUse .drop_extras() or .keep_extras() to indicate how the extra labels should be handled. Learn more at\n\thttps://bravos-power.github.io/pyoframe/latest/learn/concepts/addition",
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
    if self._extras_strategy == ExtrasStrategy.DROP:
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
    if result.get_column(missing_dims[0]).null_count() > 0:
        target_labels = target.data.select(target._dimensions_unsafe).unique(
            maintain_order=Config.maintain_order
        )
        _raise_extras_error(
            self,
            target,
            target_labels.join(self.data, how="anti", on=common_dims),
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
                    schema={VAR_KEY: Config.id_dtype, COEF_KEY: pl.Float64},
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
