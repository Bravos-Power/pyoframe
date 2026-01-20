# Define an objective

To set an objective for your optimization problem, assign an expression to either the `.minimize` or `.maximize` attribute of the Model. For example:

```python
m.minimize = capital_costs + operating_costs
```

Note that the objective expression must be dimensionless (it makes no sense to have multiple objectives with different labels). You can use [`.sum()`](./create-expressions.md) to collapse a dimensioned expression into a dimensionless one.

## Define an objective incrementally

For larger models, it is often convenient to define the objective function incrementally. To do so, use the `+=` operator:

```python
m.minimize = 0

# Later in your code
m.minimize += capital_costs

# Somewhere else in your code
m.minimize += operating_costs
```

## Alternative approach

Alternatively, rather than use `.minimize` or `.maximize`, you can use `.objective` and define the direction using the `sense` argument during model creation:

```python
m = pf.Model(sense="min")
m.objective = capital_costs + operating_costs
```