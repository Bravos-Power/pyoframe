# Integrate DataFrames

<!-- invisible-code-block: python
import os

os.chdir(os.path.join(os.getcwd(), "docs/learn/get-started/basic-example"))
-->

You are going to re-build the [previous example](./example.md) using a dataset, `food_data.csv`, instead of hard-coded values. This way, you can add as many vegetarian proteins as you like without needing to write more code. If you're impatient, [skip to the end](#put-it-all-together) to see the final result.

!!! tip "Pyoframe is built on DataFrames"

    Most other optimization libraries require you to convert your data from its `DataFrame` format to another format.[^1] Not Pyoframe! DataFrames form the core of Pyoframe making it easy to seamlessly — and efficiently — integrate large datasets into your models.

[^1]: For example, Pyomo converts your DataFrames to individual Python objects, Linopy uses multi-dimensional matrices via xarray, and gurobipy requires Python lists, dictionaries and tuples. While gurobipy-pandas uses dataframes, it only works with Gurobi!


## The data

You can download the CSV file from [here](https://github.com/Bravos-Power/pyoframe/blob/7af213c52ad33b9c01c9a14baa4cffca1ded1046/docs/learn/get-started/basic-example/food_data.csv) or create it yourself with the following content:

> `food_data.csv`
>
> | food         | protein | cost |
> | ---------    | ------- | ---- |
> | tofu_block   | 18      | 4    |
> | chickpea_can | 15      | 3    |

## Step 1: Load the data

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

    Pyoframe works the same whether you're using [Polars](https://pola.rs/) or [Pandas](https://pandas.pydata.org/), two similar libraries for manipulating data with DataFrames. We prefer using Polars because it is much faster (and generally better), but you can use whichever library you're most comfortable with.
    
    Note that, internally, Pyoframe always uses Polars during computations to ensure the best performance. If you're using Pandas, your DataFrames will automatically be converted to Polars prior to computations.
 
## Step 2: Create the model

```python
import pyoframe as pf

m = pf.Model()
```

A [`pyoframe.Model`][pyoframe.Model] instance sets the foundation of your optimization model onto which you can add optimization variables, constraints, and an objective.

## Step 3: Create a dimensioned variable

Previously, you created two variables: `m.tofu_blocks` and `m.chickpea_cans`. Instead, create a single variable **dimensioned over the column `food`**.

```python
m.Buy = pf.Variable(data["food"], lb=0, vtype="integer")
```

Printing the variable shows that it contains a `food` dimension with labels `tofu` and `chickpeas`!

```pycon
>>> m.Buy
<Variable 'Buy' lb=0 height=2>
┌──────────────┬───────────────────┐
│ food         ┆ variable          │
│ (2)          ┆                   │
╞══════════════╪═══════════════════╡
│ tofu_block   ┆ Buy[tofu_block]   │
│ chickpea_can ┆ Buy[chickpea_can] │
└──────────────┴───────────────────┘

```

## Step 3: Create the objective with `.sum()`

Previously you had:

<!-- skip: next -->

```python
m.minimize = 4 * m.tofu_blocks + 3 * m.chickpea_cans
```

How do you make use of the dimensioned variable `m.Buy` instead?

First, multiply the variable by the protein amount.

```pycon
>>> data[["food", "cost"]] * m.Buy
<Expression (linear) height=2 terms=2>
┌──────────────┬─────────────────────┐
│ food         ┆ expression          │
│ (2)          ┆                     │
╞══════════════╪═════════════════════╡
│ tofu_block   ┆ 4 Buy[tofu_block]   │
│ chickpea_can ┆ 3 Buy[chickpea_can] │
└──────────────┴─────────────────────┘

```

As you can see, Pyoframe with a bit of magic converted the `Variable` into an `Expression` where the coefficients are the protein amounts.

*[with a bit of magic]: Pyoframe always converts DataFrames into Expressions by taking the first columns as dimensions and the last column as values. Additionally, multiplication is always done between elements with the same dimensions.

Second, notice that the `Expression` still has the `food` dimension—it really contains two separate expressions, one for tofu and one for chickpeas. All objective functions must be a single expression (without dimensions) so let's sum over the `food` dimension.

```pycon
>>> (data[["food", "cost"]] * m.Buy).sum("food")
<Expression (linear) terms=2>
4 Buy[tofu_block] +3 Buy[chickpea_can]

```

This works and since `food` is the only dimensions you don't even need to specify it. Putting it all together:

```python
m.minimize = (data[["food", "cost"]] * m.Buy).sum()
```

## Step 4: Add the constraint

This is similar to how you created the objective, except now you're using `protein` and you turn the `Expression` into a `Constraint` with the `>=` operation.

```python
m.protein_constraint = (data[["food", "protein"]] * m.Buy).sum() >= 50
```

<!-- invisible-code-block: python
m.optimize()
assert m.Buy.solution["solution"].to_list() == [2, 1]
-->

## Put it all together

If you've followed the steps above your code should look like:

<!-- clear-namespace -->

```python
import pandas as pd
import pyoframe as pf

data = pd.read_csv("food_data.csv")

m = pf.Model()
m.Buy = pf.Variable(data["food"], lb=0, vtype="integer")
m.minimize = (data[["food", "cost"]] * m.Buy).sum()
m.protein_constraint = (data[["food", "protein"]] * m.Buy).sum() >= 50

m.optimize()
```

And you can retrieve the problem's solution as follows:

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

Since `m.Buy` is dimensioned, `m.Buy.solution` returned a DataFrame with the solution for each of the labels!

<!--  -->