# Transforms

!!! info "Work in progress"

    This documentation could use some help. [Learn how you can contribute](../../contribute/index.md).

Pyoframe has a few special functions that make working with dataframes easy and intuitive. Here they are:

## `sum` and `sum_by`

## `Expression.map()`


## `DataFrame.to_expr()`

!!! abstract "Summary"

    [`pandas.DataFrame.to_expr()`](../../reference/external/pandas.DataFrame.to_expr.md) and [`polars.DataFrame.to_expr()`](../../reference/external/polars.DataFrame.to_expr.md) allow users to manually convert their DataFrames to Pyoframe [Expressions][pyoframe.Expression] when Pyoframe is unable to perform an automatic conversation.

Pyoframe conveniently allows users to use [Polars DataFrames](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html) and [Pandas DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) in their mathematical expressions. To do so, Pyoframe automatically detects these DataFrames and converts them to Pyoframe [Expressions][pyoframe.Expression] whenever there is a mathematical operation (e.g., `*`, `-`, `+`) involving at least one Pyoframe object (e.g. [Variable][pyoframe.Variable], [Set][pyoframe.Set], [Expression][pyoframe.Expression], etc.).

However, if **neither** the left or right terms of a mathematical operation is a Pyoframe object, Pyoframe will not automatically convert DataFrames[^2]. In these situations, users can manually convert their DataFrames to Pyoframe expressions using `.to_expr()`.

Additionally, users should use `.to_expr()` whenever they wish to use [over][pyoframe.Expression.over], [drop_extras][pyoframe.Expression.drop_extras], or [keep_extras][pyoframe.Expression.keep_extras] on a DataFrame.

!!! info "Under the hood"

    How is `.to_expr()` a valid Pandas and Polars method? `import pyoframe` causes Pyoframe to [monkey patch](https://stackoverflow.com/questions/5626193/what-is-monkey-patching) the Pandas and Polars libraries. One of the patches adds the `.to_expr()` method to both `pandas.DataFrame` and `polars.DataFrame` (see [`monkey_patch.py`](https://github.com/Bravos-Power/pyoframe/tree/main/src/pyoframe)).

!!! tip "Working with Pandas Series"

    You can call `.to_expr()` on a Pandas Series to produce an expression where the labels will be determined from the Series' index.

[^2]: After all, how could it? If a user decides to write code that adds two DataFrames together, Pyoframe shouldn't interfere.

### Example

Consider the following scenario where we have some population data on yearly births and deaths, as well as an immigration variable.

```python
import pyoframe as pf
import pandas as pd

population_data = pd.DataFrame(
    dict(year=[2025, 2026], births=[1e6, 1.1e6], deaths=[-1.2e6, -1.4e6])
)

model = pf.Model()
model.immigration = pf.Variable(dict(year=[2025, 2026]))
```

Now, saw we wanted an expression representing the total yearly population change. The following works just fine:

```pycon
>>> (
...     model.immigration
...     + population_data[["year", "births"]]
...     + population_data[["year", "deaths"]]
... )
<Expression height=2 terms=4 type=linear>
┌──────┬───────────────────────────┐
│ year ┆ expression                │
│ (2)  ┆                           │
╞══════╪═══════════════════════════╡
│ 2025 ┆ immigration[2025] -200000 │
│ 2026 ┆ immigration[2026] -300000 │
└──────┴───────────────────────────┘

```

But, if we simply change the order of the terms in our addition, we get an error:

```pycon
>>> (
...     population_data[["year", "births"]]
...     + population_data[["year", "deaths"]]
...     + model.immigration
... )
Traceback (most recent call last):
...
ValueError: Cannot create an expression with duplicate labels:
┌────────┬────────┬─────────┬───────────────┐
│ births ┆ deaths ┆ __coeff ┆ __variable_id │
│ ---    ┆ ---    ┆ ---     ┆ ---           │
│ f64    ┆ f64    ┆ i64     ┆ i32           │
╞════════╪════════╪═════════╪═══════════════╡
│ null   ┆ null   ┆ 4050    ┆ 0             │
│ null   ┆ null   ┆ 4052    ┆ 0             │
└────────┴────────┴─────────┴───────────────┘.

```

What happened? Since Python computes additions from left to right, the second re-arranged version failed because, in the first addition, neither operand is a Pyoframe object. As such, the addition is done by Pandas, not Pyoframe, which leads to unexpected results.

How do we avoid these weird behaviors? Users can manually convert their DataFrames to Pyoframe expressions ahead of time with `.to_expr()`. For example:

```pycon
>>> (
...     population_data[["year", "births"]].to_expr()
...     + population_data[["year", "deaths"]].to_expr()
...     + model.immigration
... )
<Expression height=2 terms=4 type=linear>
┌──────┬─────────────────────────────┐
│ year ┆ expression                  │
│ (2)  ┆                             │
╞══════╪═════════════════════════════╡
│ 2025 ┆ -200000 + immigration[2025] │
│ 2026 ┆ -300000 + immigration[2026] │
└──────┴─────────────────────────────┘

```
