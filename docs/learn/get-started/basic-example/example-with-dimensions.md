# Integrate DataFrames

<!-- invisible-code-block: python
import os

os.chdir(os.path.join(os.getcwd(), "docs/learn/get-started/basic-example"))
-->

!!! tip "Pyoframe is built on DataFrames"
    Most other optimization libraries require you to massage your data, stored in a `DataFrame`, into other data structures.[^1] Not Pyoframe! DataFrames form the core of Pyoframe making it easy to seamlessly — and efficiently — integrate large datasets into your models.

[^1]: For example, Pyomo converts your DataFrames to individual Python objects, Linopy uses multi-dimensional matrices via xarray, and gurobipy requires Python lists, dictionaries and tuples.

We are going to re-build our [previous example](./example.md) using a dataset, `food_data.csv`, instead of hard-coded values. This way, we can add as many vegetarian proteins as we like without needing to write more code. If you're impatient, [skip to the end](#put-it-all-together) to see the final result.


> `food_data.csv`
>
> | food         | protein | cost |
> | ---------    | ------- | ---- |
> | tofu_block   | 18      | 4    |
> | chickpea_can | 15      | 3    |



### Step 1: Load the data

Load `food_data.csv` using [Polars](https://pola.rs/) or [Pandas](https://pandas.pydata.org/).

=== "Polars"

    ```python
    import polars as pl

    data = pl.read_csv("food_data.csv")
    ```

=== "Pandas"

    ```python
    import pandas as pd

    data = pd.read_csv("food_data.csv")
    ```

!!! tip "Pandas vs. Polars: Which should I use?"
    Pyoframe works the same whether you're using [Polars](https://pola.rs/) or [Pandas](https://pandas.pydata.org/), two similar libraries for manipulating data with DataFrames. We prefer using Polars because it is much faster, but you can use whichever library you're most comfortable with.
    
    Note that, internally, Pyoframe always uses Polars during computations to ensure the best performance. If you're using Pandas, your DataFrames will automatically be converted to Polars prior to computations. If needed, you can convert a Polars DataFrame back to Pandas using [`polars.DataFrame.to_pandas()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_pandas.html#polars.DataFrame.to_pandas).
 
### Step 2: Create the model

```python
import pyoframe as pf

m = pf.Model()
```

A [`pyoframe.Model`][pyoframe.Model] object sets the foundation onto which you will add optimization variables, constraints, and an objective.

### Step 3: Create a dimensioned variable

Previously, we created two variables: `m.tofu_blocks` and `m.chickpea_cans`. Instead, we now create a single variable **dimensioned over the column `food`**.

```python
m.Buy = pf.Variable(data["food"], lb=0, vtype="integer")
```

Printing the variable shows that it contains a `food` dimension with indices `tofu` and `chickpeas`!

```pycon
>>> m.Buy
<Variable name=Buy lb=0 size=2 dimensions={'food': 2}>
[tofu_block]: Buy[tofu_block]
[chickpea_can]: Buy[chickpea_can]

```

!!! tip "Capitalize model variables"

    We suggest capitalizing model variables (i.e. `m.Buy`, not `m.buy`) to make distinguishing what is and isn't a variable easy.

### Step 3: Create the objective with `pf.sum`

Previously we had:

<!-- skip: next -->

```python
m.minimize = 4 * m.tofu_blocks + 3 * m.chickpea_cans
```

How do we make use of our dimensioned variable `m.Buy` instead?

First, we multiply the variable by the protein amount.

```pycon
>>> data[["food", "cost"]] * m.Buy
<Expression size=2 dimensions={'food': 2} terms=2>
[tofu_block]: 4 Buy[tofu_block]
[chickpea_can]: 3 Buy[chickpea_can]

```

As you can see, Pyoframe with a bit of magic converted our `Variable` into an `Expression` where the coefficients are the protein amounts.

*[with a bit of magic]: Pyoframe always converts DataFrames into Expressions by taking the first columns as dimensions and the last column as values. Additionally, multiplication is always done between elements with the same dimensions.

Second, notice that our `Expression` still has a `food` dimension—it really contains two separate expressions, one for tofu and one for chickpeas. All objective functions must be a single expression (without dimensions) so let's sum over the `food` dimensions using `pf.sum()`.

```pycon
>>> pf.sum("food", data[["food", "cost"]] * m.Buy)
<Expression size=1 dimensions={} terms=2>
4 Buy[tofu_block] +3 Buy[chickpea_can]

```

This works and since `food` is the only dimensions we don't even need to specify it. Putting it all together:

```python
m.minimize = pf.sum(data[["food", "cost"]] * m.Buy)
```

### Step 4: Add the constraint

This is similar to how we created the objective, except now we're using `protein` and we turn our `Expression` into a `Constraint` by with the `>=` operation.

```python
m.protein_constraint = pf.sum(data[["food", "protein"]] * m.Buy) >= 50
```

<!-- invisible-code-block: python
m.optimize()
assert m.Buy.solution["solution"].to_list() == [2, 1]
-->

### Put it all together

<!-- clear-namespace -->

```python
import pandas as pd
import pyoframe as pf

data = pd.read_csv("food_data.csv")

m = pf.Model()
m.Buy = pf.Variable(data["food"], lb=0, vtype="integer")
m.minimize = pf.sum(data[["food", "cost"]] * m.Buy)
m.protein_constraint = pf.sum(data[["food", "protein"]] * m.Buy) >= 50

m.optimize()
```

So you should buy:

```pycon
>>> m.Buy.solution
┌──────────────┬──────────┐
│ food         ┆ solution │
│ ---          ┆ ---      │
│ str          ┆ i64      │
╞══════════════╪══════════╡
│ tofu_block   ┆ 2        │
│ chickpea_can ┆ 1        │
└──────────────┴──────────┘

```

Notice that since `m.Buy` is dimensioned, `m.Buy.solution` returned a DataFrame with the solution for each of indices.

!!! info "Returning Pandas DataFrames"

    Pyoframe currently always returns Polars DataFrames but you can easily convert them to Pandas using [`.to_pandas()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_pandas.html#polars.DataFrame.to_pandas). In the future, we plan to add support for automatically returning Pandas DataFrames. [Upvote the issue](https://github.com/Bravos-Power/pyoframe/issues/47) if you'd like this feature.
<!--  -->