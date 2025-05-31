# A simple model

Here's a simple model to show you Pyoframe's syntax. Click on the :material-plus-circle: buttons to discover what's happening.

```python
import pyoframe as pf

m = pf.Model()

# You can buy tofu or chickpeas
m.tofu = pf.Variable(lb=0)  # (1)!
m.chickpeas = pf.Variable(lb=0)

# You want to maximize your protein intake (10g per tofu, 8g per chickpeas)
m.maximize = 10 * m.tofu + 8 * m.chickpeas # (2)!

# You must stay with your $10 budget (4$ per tofu, $2 per chickpeas)
m.budget_constraint = 4 * m.tofu + 2 * m.chickpeas <= 10 # (3)!

m.optimize()  # (4)!

print("You should buy:")
print(f"\t{m.tofu.solution} blocks of tofu")
print(f"\t{m.chickpeas.solution} cans of chickpeas")
```

```{.python continuation hide}
assert m.tofu.solution == 0
assert m.chickpeas.solution == 5
```

1. Create a variable with a lower bound of zero (`lb=0`) so that you can't buy a negative quantity of tofu!
2. Define your objective by setting the reserved variables `.maximize` or `.minimize`.
3. Creates constraints by using `<=`, `>=`, or `==`.
4. Pyoframe automatically detects your installed solver and optimizes your model!

## Use dimensions

The above model would quickly become unworkable if we had more than just tofu and chickpeas. I'll walk you through how we can make a `food` dimension to make this scalable. You can also skip to the end to see the example in full!

Note that instead of hardcoding our values, we'll be reading them from the following csv file.

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

```{.python continuation hide}
import polars as pl
import os
data = pl.read_csv(os.path.join(os.getcwd(), "docs/learn/01_getting-started/inputs/food_data.csv"))
```

### Create the model

```{.python continuation}
import pyoframe as pf
m = pf.Model()
```

### Create an dimensioned variable
Previously, we created two variables: `m.tofu` and `m.chickpeas`. Instead, we now create a single variable dimensioned over `food`.

```{.python continuation}
m.Buy = pf.Variable(data[["food"]], lb=0)
```

If you print the variable, you'll see it actually contains a `tofu` and `chickpeas` variable!

```pycon
>>> m.Buy
<Variable name=Buy lb=0 size=2 dimensions={'food': 2}>
[tofu]: Buy[tofu]
[chickpeas]: Buy[chickpeas]
```

```{.python hide continuation}
assert repr(m.Buy) ==  """<Variable name=Buy lb=0 size=2 dimensions={'food': 2}>
[tofu]: Buy[tofu]
[chickpeas]: Buy[chickpeas]"""
```

!!! tip "Tip"
    Naming your model's decision variables with an uppercase first letter (e.g. `m.Buy`) makes it easier to remember what's a variable and what isn't.

### Create the objective

Previously we had:
```{.python notest}
m.maximize = 10 * m.tofu + 8 * m.chickpeas
```

How do we make use of our dimensioned variable `m.Buy` instead?

First, we multiply the variable by the protein amount.

```pycon
>>> data[["food", "protein"]] * m.Buy
<Expression size=2 dimensions={'food': 2} terms=2>
[tofu]: 10 Buy[tofu]
[chickpeas]: 8 Buy[chickpeas]
```

```{.python continuation hide}
assert repr(data[["food", "protein"]] * m.Buy) == """<Expression size=2 dimensions={'food': 2} terms=2>
[tofu]: 10 Buy[tofu]
[chickpeas]: 8 Buy[chickpeas]"""
```

As you can see, Pyoframe with a bit of magic converted our `Variable` into an `Expression` where the coefficients are the protein amounts!

*[with a bit of magic]:
    Pyoframe always converts dataframes into Expressions by taking the first columns as dimensions and the last column as values. Additionally, multiplication is always done between elements with the same dimensions.

Second, notice that our `Expression` still has a `food` dimension—it really contains two separate expressions, one for tofu and one for chickpeas. Our model's objective must be a single expression (without dimensions) so let's sum over the `food` dimensions using `pf.sum()`.

```pycon
>>> pf.sum("food", data[["food", "protein"]] * m.Buy)
<Expression size=1 dimensions={} terms=2>
10 Buy[tofu] +8 Buy[chickpeas]
```

This works and since `food` is the only dimensions we don't even need to specify it. Putting it all together:

```{.python continuation}
m.maximize = pf.sum(data[["food", "protein"]] * m.Buy)
```

### Adding the constraint

This is similar to how we created the objective, except now we're using `cost` and we turn our `Expression` into a `Constraint` by with the `<=` operation.

```{.python continuation}
m.budget_constraint = pf.sum(data[["food", "cost"]] * m.Buy) <= 10
```

### Putting it all together

```{.python hide}
import os
from pathlib import Path
data_folder = Path(os.path.join(os.getcwd(), "docs/learn/01_getting-started/inputs"))
```

```{.python continuation}
import pandas as pd
import pyoframe as pf

data = pd.read_csv(data_folder / "food_data.csv")

m = pf.Model()
m.Buy = pf.Variable(data[["food"]], lb=0)
m.maximize = pf.sum(data[["food", "protein"]] * m.Buy)
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
