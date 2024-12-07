# Using Dataframes

Pyoframe's most powerful feature is its ability to work directly with your Dataframes! We support both Pandas and Polars dataframes although if you're looking for performance, [Polars is a lot faster](https://pola.rs/).

Here we'll walk you through the classical diet problem using Pyoframe.

## 1. Import your data

Let's generalize our previous [simple model](./02_build-simple-model.md) to a dataset! Say you having the following three files.

=== "`foods.csv`"

    {{ read_csv('./assets/foods.csv') }}

=== "`nutrients.csv`"

    {{ read_csv('./assets/nutrients.csv') }}

=== "`foods_to_nutrients.csv`"

    {{ read_csv('./assets/foods_to_nutrients.csv') }}

You'd like to find how to meet your daily nutritional needs (see `nutrients.csv`) by spending as little as possible on a mix of options (see `foods.csv`). Let's first import our data!

=== "Using Pandas"

    ```python
    import pyoframe as pf
    import pandas as pd

    foods = pd.read_csv("foods.csv")
    nutrients = pd.read_csv("nutrients.csv")
    foods_to_nutrients = pd.read_csv("foods_to_nutrients.csv")
    ```

=== "Using Polars"

    ```python
    import pyoframe as pf
    import polars as pl

    foods = pl.read_csv("foods.csv")
    nutrients = pl.read_csv("nutrients.csv")
    foods_to_nutrients = pl.read_csv("foods_to_nutrients.csv")
    ```

## 2. Build your model

Ok, the code below has some new concepts but don't worry just yet. It will all make sense!

```python3
m = pf.Model("min")

# Define our variable: how much to buy of each food
m.purchase_quantity = pf.Variable(
    foods[["food"]], # (1)!
    lb=0, 
    ub=foods[["food", "stock"]] # (2)!
)

m.nutrients = pf.sum( # (3)!
    over="food", 
    expr=m.purchase_quantity * foods_to_nutrients # (4)!
) 

m.min_nutrients = m.nutrients >= nutrients[["category", "min"]]
m.max_nutrients = m.nutrients <= nutrients[["category", "max"]]

m.objective = pf.sum(m.purchase_quantity * foods[["food", "cost"]]) # (5)!
```

1. This variable is actually 3 variables, one for each element in `foods[["food"]]`!
2. Dataframes can be used to set variable bounds. The last column is always used as the bound.
3. `pf.sum` sums the terms of `expr=` over the `food` dimension, returning a linear expression indexed only over `category`
4. Multiplication acts like matrix multiplication. Our `purchase_quantity` (indexed by `"food"`) is multiplied by a parameter that is indexed by `"food"` and `"category"`.
5. Build a linear expression of that sums the cost of buying each food.

## 3. Solve the model

This ones easy:

```python3
m.optimize()
```

!!! note "How does this work?"

    Under the hood, Pyoframe is writing your model to an [`.lp` file](https://docs.gurobi.com/projects/optimizer/en/current/reference/fileformats/modelformats.html#lp-format), asking Gurobi to read and solve it, and then loading Gurobi's results back into the `pf.Model()` object.

## 4. Read the results

```
m.purchase_quantity.solution.write_csv("results.csv")
```
This will create the following `results.csv` file:

{{ read_csv('./assets/results.csv') }}

Turns out hamburgers are just the best bang for your buck :person_shrugging: