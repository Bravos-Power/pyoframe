# Quadratic Expressions

Quadratic expressions work as you'd expect. Simply multiply two linear expression together (or square an expression with `**2`) and you'll get a quadratic. The quadratic can then be used in constraints or the objective. 

## Example

## TODO

```python3
```

## Note for developers: Internal Representation of Quadratics

Internally, Pyoframe's `Expression` object is used for both linear and quadratic expressions. When the dataframe within an `Expression` object (i.e. `Expression.data`) contains an additional column (named `__quadratic_variable_id`) we know that the expression is a quadratic.

This extra column stores the ID of the second variable in quadratic terms. For terms with only one variable, this column contains ID `0` (a reserved variable ID which can thought of as equalling `1`). The variable ID in the `__variable_id` column is always greater or equal than the variable ID in the `__quadratic_variable_id`. This means that linear terms always have the variable id in the first column and `0` in the second column. Also, a `0` in the first column implies that the second column must also be `0` and therefore the term is a constant.