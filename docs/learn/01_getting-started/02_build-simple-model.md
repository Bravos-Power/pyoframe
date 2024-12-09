# A simple model

Here's a simple model to show you Pyoframe's syntax. Click on the :material-plus-circle: buttons to discover what's happening.

```python
import pyoframe as pf

m = pf.Model("max") # (1)!

# You can buy tofu or chickpeas
m.tofu = pf.Variable(lb=0)  # (2)!
m.chickpeas = pf.Variable(lb=0)

# Youd want to maximize your protein intake (10g for tofu, 8g for chickpeas)
m.objective = 10 * m.tofu + 8 * m.chickpeas # (3)!

# You have $10 and tofu costs $4 while chickpeas cost $2.
m.budget_constraint = 4 * m.tofu + 2 * m.chickpeas <= 10 # (4)!

m.optimize()

print("You should buy:")
print(f"\t{m.tofu.solution} blocks of tofu")
print(f"\t{m.chickpeas.solution} cans of chickpeas")
```

1. Creating your model is always the starting point!
2. `lb=0` sets the variable's lower bound to ensure you can't buy a negative quantity of tofu!
3. Variables can be added and multiplied as you'd expect!
4. Using `<=`, `>=` or `==` will automatically create a constraint.

## Use dimensions

The above model would quickly become unworkable if we had more than just tofu and chickpeas. Let's create a `food` dimension to make this scalable. While were at it, let's also read our data from the following .csv file instead of hardcoding it.

> `food_data.csv`
>
> |food|protein|cost|
> |--|--|--|
> |tofu|10|4|
> |chickpeas|8|2|

### Load your data

Nothing special here. Load your data using your favourite dataframe library. We like [Polars](https://pola.rs/) because it's _fast_ but Pandas works too.

=== "Pandas"

    ```python
    import pandas as pd

    data = pd.read_csv("food_data.csv")
    ```

=== "Polars"

    ```python
    import polars as pl

    data = pl.read_csv("food_data.csv")
    ```

### Create the model

```python
import pyoframe as pf
m = pf.Model("max")
```

### Create an dimensioned variable
Previously, we created two variables: `m.tofu` and `m.chickpeas`. Instead, we now create a single variable dimensioned over `food`.

```python
m.Buy = pf.Variable(data[["food"]], lb=0)
```

If you print the variable, you'll see it actually contains a `tofu` and `chickpeas` variable!

```pycon
>>> m.Buy
<Variable name=Buy lb=0 size=2 dimensions={'food': 2}>
[tofu]: Buy[tofu]
[chickpeas]: Buy[chickpeas]
```

!!! tip "Tip"
    Naming your model's decision variables with an uppercase first letter (e.g. `m.Buy`) makes it to remember what's a variable and what isn't.

### Create the objective

Previously we had:
```python
m.objective = 10 * m.tofu + 8 * m.chickpeas
```

How do we make use of our dimensioned variable `m.Buy` instead?

First, we multiply the variable by the protein amount.

```pycon
>>> data[["food", "protein"]] * m.Buy
<Expression size=2 dimensions={'food': 2} terms=2>
[tofu]: 10 Buy[tofu]
[chickpeas]: 8 Buy[chickpeas]
```
As you can see, Pyoframe with a bit of magic converted our `Variable` into an `Expression` where the coefficients are the protein amounts!

*[with a bit of magic]:
    Pyoframe always converts dataframes into Expressions by taking the first columns as dimensions and the last column as values. Additionally, multiplication is always done between elements with the same dimensions.

Second, notice that our `Expression` still has a `food` dimension—it really contains two seperate expressions, one for tofu and one for chickpeas. Our model's objective must be a single expression (without dimensions) so let's sum over the `food` dimensions using `pf.sum()`.

```pycon
>>> pf.sum("food", data[["food", "protein"]] * m.Buy)
<Expression size=1 dimensions={} terms=2>
10 Buy[tofu] +8 Buy[chickpeas]
```

This works and since `food` is the only dimensions we don't even need to specify it. Putting it all together:

```python
m.objective = pf.sum(data[["food", "protein"]] * m.Buy)
```

### Adding the constraint

This is similar to how we created the objective, except now we're using `cost` and we turn our `Expression` into a `Constraint` by with the `<=` operation.

```python
m.budget_constraint = pf.sum(data[["food", "cost"]] * m.Buy) <= 10
```

### Putting it all together

```python
import pandas as pd
import pyoframe as pf

data = pd.read_csv("food_data.csv")

m = pf.Model("max")
m.Buy = pf.Variable(data[["food"]], lb=0)
m.objective = pf.sum(data[["food", "protein"]] * m.Buy)
m.budget_constraint = pf.sum(data[["food", "cost"]] * m.Buy) <= 10

m.optimize()
```

So you should buy:
```pycon
>>> m.Buy.solution
┌───────────┬──────────┐
│ food      ┆ solution │
│ ---       ┆ ---      │
│ str       ┆ f64      │
╞═══════════╪══════════╡
│ tofu      ┆ 0.0      │
│ chickpeas ┆ 5.0      │
└───────────┴──────────┘
```
Notice that since `m.Buy` is dimensioned, `m.Buy.solution` returned a dataframe with the solution for each of indices!

!!! info "info"
    Pyoframe currently always returns Polars dataframes although we plan to add support for returning Pandas dataframes in the future. [Upvote the issue](https://github.com/Bravos-Power/pyoframe/issues/47) if you'd like this feature and in the meantime use `.to_pandas()` to convert from a Polars dataframe.
