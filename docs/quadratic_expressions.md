# Quadratic Expressions

Quadratic expressions work as you'd expect. Simply multiply two linear expression together (or square an expression with `**2`) and you'll get a quadratic. The quadratic can then be used in constraints or the objective. 

## Example

## TODO

```python3
```

## Note for Pyoframe developers: Internal Representation of Quadratics

Internally, Pyoframe's `Expression` object is used for both linear and quadratic expressions. When the dataframe within an `Expression` object (i.e. `Expression.data`) contains an additional column (named `__quadratic_variable_id`) we know that the expression is a quadratic.

This extra column stores the ID of the second variable in quadratic terms. For terms with only one variable, this column contains ID `0` (a reserved variable ID which can thought of as meaning 'no variable'). The variables in a quadratic are rearranged such that the ID in the `__variable_id` column is always greater or equal than the variable ID in the `__quadratic_variable_id` (recall: a*b=b*a). This rearranging not only ensures that a*b+b*a=2a*b but also generates a useful property: If the variable ID in the first column (`__variable_id`) is `0` we know the variable ID in the second must also be `0` and therefore the term must be a constant.

The additional quadratic variable ID column is automatically dropped if through arithmetic the quadratic terms cancel out.