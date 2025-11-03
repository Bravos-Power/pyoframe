# Pyoframe basics

Building a model in Pyoframe is easy as it involves only a few steps. Each of the following sections describes one of those steps.

1. [Create a `Model` object](#creating-a-model)
2. [Add decision variables to your model with `pf.Variable`](#defining-variables).
3. [Formulate key mathematical expressions](#formulating-expressions)
4. [Add constraints to your model with `<=`, `==`, and `>=` operators](#adding-constraints)
5. [Set the objective expression](#setting-the-objective)
6. [Optimize!](#optimizing-your-model)
7. [Read optimization results](#reading-results)

## Creating a model

Creating a model is simple:

```python
import pyoframe as pf

m = pf.Model()
```

By default, Pyoframe will try to use whichever solver is installed on your computer. You'll likely want to specify your preferred solver by writing, e.g., `pf.Model(solver="highs")`.

## Defining variables

The syntax to add a variable to a model is:

```python
m.my_variable_name = pf.Variable()  # (1)!
```

1. Curious to know why this works? Pyoframe overrides the `__setattr__` method of the `Model` class such that whenever you set a new attribute (in this case `my_variable_name`), the `Model` object records it and adds it to your solver.

By default, variables are unbounded. To set a lower bound and/or upper bound,use the `lb` and/or `ub` arguments.

```python
m.my_positive_variable = pf.Variable(lb=0)
```

Integer or binary variables can be created using the [VType][pyoframe.VType] enum or simply a string:

```python
m.my_binary_variable = pf.Variable(vtype="binary")
m.my_integer_variable = pf.Variable(vtype="integer")
```

### Dimensioned variables

The previous examples create a single (dimensionless) variable. Yet, often you'll want to create an array of variables. In Pyoframe, an array of variables is called a _dimensioned variable_ because dimensioned variables have one or more _dimensions_. Consider this example:

```
import pandas as pd

df = pd.DataFrame({"weekday": ["Mon", "Tue", "Wed", "Thu", "Fri"]})

m.my_dimensioned_var = pf.Variable(df)
```

In this example, `m.my_dimensioned_var` is a dimensioned variable with one dimension called `weekday`. Dimensioned variables can be thought of as a container containing many dimensionless variables, in this case, 5 variables labelled by the weekdays:

```pycon
>>> m.my_dimensioned_var

```

Here are all the ways in which you can create a dimensioned variable.

=== "Using a `DataFrame`"

    DataFrames are the most common way of creating variables in Pyoframe.

    If a `DataFrame` is passed to `pf.Variable`, a variable will be created for every row in the `DataFrame` and labelled according to the values in that row. Column names become the dimension names. This works for both Polars and Pandas.

    ```python
    import pandas as pd

    df = pd.DataFrame({"day": ["Mon", "Tue", "Wed", "Thu", "Fri"]})
    m.example_1 = pf.Variable(df)
    ```

=== "Using a `Series`"

    If a `Series` is passed to `pf.Variable`, it is treated as a 1-column DataFrame. This works for both Polars and Pandas.

    ```python
    import pandas as pd

    series = pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri"], name="day")
    m.example_2 = pf.Variable(series)
    ```

=== "Using an `Index`"

    If a Pandas `Index` is passed to `pf.Variable`, it is treated like a `DataFrame`.

    ```python
    import pandas as pd

    series = pd.Index(["Mon", "Tue", "Wed", "Thu", "Fri"], name="day")
    m.example_3 = pf.Variable(series)
    ```

=== "Using a `dict`"

    If a dictionary is passed to `pf.Variable`, the keys become the dimension names and the values are the labels.

    ```python
    m.example_4 = pf.Variable({"day": ["Mon", "Tue", "Wed", "Thu", "Fri"]})
    ```

=== "Using a `Set`"
    
    Pyoframe offers a class [Set][pyoframe.Set] to easily define dimensioned variables in a reusable way.

    ```python
    weekdays = pf.Set(day=["Mon", "Tue", "Wed", "Thu", "Fri"])
    m.example_5 = pf.Variable(weekdays)
    ```

!!! tip "Cartesian products"

    If multiple arguments are passed to `pf.Variable`, the cartesian product will be computed. For example,

    ```python
    chaoticness = pf.Set(chaoticness=["lawful", "neutral", "chaotic"])
    goodness = pf.Set(goodness=["good", "neutral", "evil"])

    m.Personality = pf.Variable(chaoticness, goodness)
    ```

    Can you guess the result? Here it is:

    ```pycon
    >>> m.Personality
    
    ```

## Formulating expressions



## Adding constraints

## Setting the objective

## Optimizing your model

## Reading results
