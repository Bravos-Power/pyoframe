# Accessing the solver

## Accessing model attributes

Pyoframe lets you read and set solver attributes using `model.attr.<your-attribute>`. For example, if you'd like to prevent the solver from printing to the console you can do:

```python
m = pf.Model()
m.attr.Silent = True
```

We support all of [PyOptInterface's model attributes](https://metab0t.github.io/PyOptInterface/model.html#id1) as well as [Gurobi's attributes](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/model.html) (only when using Gurobi).

```pycon
>>> m.optimize()
>>> m.attr.TerminationStatus  # Read a PyOptInterface model attribute (available with all solvers)
<TerminationStatusCode.OPTIMAL: 2>
>>> m.attr.Status  # Read the Gurobi model attribute (available with Gurobi only)
2
```

## Accessing variable and constraint attributes

Similar to above, Pyoframe allows directly accessing the PyOptInterface or the solver's variable and constraint attributes.

```python
m = pf.Model()
m.X = pf.Variable()
m.X.attr.PrimalStart = 5
```

If the variable or constraint is dimensioned, the attribute can accept/return a dataframe instead of a constant.

## Accessing model parameters (Gurobi only)

Gurobi supports model attributes (see above) and model parameters ([full list here](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html)). You can read or set model parameters with `model.params.<your-parameter>`. For example:

```{.python continuation}
m.params.method = 2  # Use a barrier method to solve
```
