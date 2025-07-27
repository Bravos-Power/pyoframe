# Accessing the solver

## Model attributes

Pyoframe lets you read and set solver attributes using `model.attr.<your-attribute>`. For example, if you'd like to prevent the solver from printing to the console you can do:

```python
m = pf.Model()
m.attr.Silent = True
```

Pyoframe support all [PyOptInterface attributes](https://metab0t.github.io/PyOptInterface/model.html#id1) and, when using Gurobi, all [Gurobi attributes](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/model.html).

```pycon
>>> m.optimize()
>>> m.attr.TerminationStatus  # PyOptInterface attribute (always available)
<TerminationStatusCode.OPTIMAL: 2>
>>> m.attr.Status  # Gurobi attribute (only available with Gurobi)
2

```

## Model parameters (Gurobi only)

Gurobi supports model attributes (see above) and model parameters ([full list here](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html)). You can read or set model parameters with `model.params.<your-parameter>`. For example:

```python
m.params.method = 2  # Use a barrier method to solve
```

## Variable and constraint attributes

Similar to above, Pyoframe allows directly accessing the PyOptInterface or the solver's variable and constraint attributes.

```python
m = pf.Model()
m.X = pf.Variable()
m.X.attr.PrimalStart = 5
```

If the variable or constraint is dimensioned, the attribute can accept/return a DataFrame instead of a constant.


