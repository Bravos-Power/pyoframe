# Internal details

Pyoframe's inner workings involve a few tricks that you should be aware of if you wish to contribute to Pyoframe's code base.

## The zero variable

Whenever a [Model][pyoframe.Model] is instantiated, Pyoframe immediately
creates a variable whose value is fixed to `1` and has a variable id of `0` — _the Zero Variable_.
This allows Pyoframe to represent constant terms in mathematical expressions as
 multiples of the Zero Variable. For example, the expression `3 * var_8 + 5` is represented as `3 * var_8 + 5 * var_0`.
This eliminates the need to separately track
constant terms and also simplifies the [handling of quadratics](#quadratics).

## Quadratics

Internally, [Expression][pyoframe.Expression] is used to represent both linear and quadratic mathematical expressions. When a quadratic expression is formed, column `__quadratic_variable_id` is added to [Expression.data][pyoframe.Expression.data]. If an expression's quadratic terms happen to cancel out (e.g. `(ab + c) - ab`), this column is automatically removed.

Column `__quadratic_variable_id` records the ID of the _second_ variable in a quadratic term (the `b` in `3ab`). For linear terms, which have no second variable, this column contains the [Zero Variable](#the-zero-variable).

Quadratic terms are always stored such that the first term's variable ID (in column `__variable_id`) is greater or equal to the second term's variable id (in column `__quadratic_variable_id`). For example, `var_7 * var_8` would be rearranged and stored as `var_8 * var_7`. This helps simplify expressions and provides a useful guarantee: If the variable in the first column (`__variable_id`) is the Zero Variable (`var_0`) we know the variable in the second column must also be the Zero Variable and, thus, the term must be a constant.

## Division

Divisions are rearranged into multiplications when possible. Specifically, `a / b` is computed as `a * (1 / b)` (see `BaseOperableBlock.__truediv__`) except for the special case where `a` is a `float` or `int`. In that case, a Polars operation is used to compute the division (see `Expression.__rtruediv__`).