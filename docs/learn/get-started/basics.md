# Learn the basics

Now that you've gotten a sense of what Pyoframe can do, it's time to learn how to build your own optimization models from scratch.

Building a Pyoframe model involves only a few steps. Each of the following sections describes one of those steps.

1. [Create the `Model`](#create-a-model)
2. [Add decision variables](#define-variables)
3. [Formulate key mathematical expressions](#formulate-expressions)
4. [Add constraints](#add-constraints)
5. [Set the objective](#set-the-objective)
6. [Optimize!](#optimize)
7. [Retrieve results](#retrieve-results)

## Create a model

Creating a model is simple:

```python
import pyoframe as pf

m = pf.Model()
```

By default, Pyoframe will try to use whichever solver is installed on your computer. To specify a particular solver write, e.g.: `pf.Model(solver="highs")`. Refer to [Model][pyoframe.Model] for more configuration options.

## Define variables

Like most Pyoframe objects, variables can either be dimensionless or dimensioned. Let's start with the simpler dimensionless case.

### Dimensionless variables

The syntax to add a variable to a model is:

```python
m.My_Var = pf.Variable()  # (1)!
```

1. Curious to know why this works? Pyoframe overrides the `__setattr__` method of the `Model` class such that whenever you set a new attribute (in this case `My_Var`), the `Model` object records it and adds it to your solver.


By default, variables are unbounded. To set a lower or upper bound, use the `lb` or `ub` arguments:

```python
m.My_Positive_Var = pf.Variable(lb=0)
```

Create integer or binary variables using the [VType][pyoframe.VType] enum or simply a string:

```python
m.My_Binary_Var = pf.Variable(vtype="binary")
m.My_Integer_Var = pf.Variable(vtype="integer")
```

!!! tip "Naming variables"
    I like to use upper case names for variables (i.e. `m.My_Var` instead of `m.my_var`) because it makes them easy to distinguish. But you're free to choose any name you like.

### Dimensioned variables

Often, you'll want to create a variable for every row in your data. To do so, simply pass your data to `pf.Variable`.

For example, in the following code, `m.WeekDay_Var` contains 5 variables, one for each weekday:

```python
import pandas as pd

df = pd.DataFrame({"weekday": ["Mon", "Tue", "Wed", "Thu", "Fri"]})

m.WeekDay_Var = pf.Variable(df)
```

You can confirm this by printing the variable:

```pycon
>>> m.WeekDay_Var
<Variable 'WeekDay_Var' height=5>
┌─────────┬──────────────────┐
│ weekday ┆ variable         │
│ (5)     ┆                  │
╞═════════╪══════════════════╡
│ Mon     ┆ WeekDay_Var[Mon] │
│ Tue     ┆ WeekDay_Var[Tue] │
│ Wed     ┆ WeekDay_Var[Wed] │
│ Thu     ┆ WeekDay_Var[Thu] │
│ Fri     ┆ WeekDay_Var[Fri] │
└─────────┴──────────────────┘

```

If you pass multiple arguments to `pf.Variable`, the [cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) will be computed. For example:
    
```pycon
>>> chaoticness = pf.Set(chaoticness=["lawful", "neutral", "chaotic"])
>>> goodness = pf.Set(goodness=["good", "neutral", "evil"])
>>> m.Personality = pf.Variable(chaoticness, goodness)
>>> m.Personality
<Variable 'Personality' height=9>
┌─────────────┬──────────┬──────────────────────────────┐
│ chaoticness ┆ goodness ┆ variable                     │
│ (3)         ┆ (3)      ┆                              │
╞═════════════╪══════════╪══════════════════════════════╡
│ lawful      ┆ good     ┆ Personality[lawful,good]     │
│ lawful      ┆ neutral  ┆ Personality[lawful,neutral]  │
│ lawful      ┆ evil     ┆ Personality[lawful,evil]     │
│ neutral     ┆ good     ┆ Personality[neutral,good]    │
│ neutral     ┆ neutral  ┆ Personality[neutral,neutral] │
│ neutral     ┆ evil     ┆ Personality[neutral,evil]    │
│ chaotic     ┆ good     ┆ Personality[chaotic,good]    │
│ chaotic     ┆ neutral  ┆ Personality[chaotic,neutral] │
│ chaotic     ┆ evil     ┆ Personality[chaotic,evil]    │
└─────────────┴──────────┴──────────────────────────────┘

```

In this last example, we used `pf.Set` instead of a `DataFrame`. `Variable` actually accepts a variety of input data types. The following tabs show equivalent ways of creating the same variable.

=== "`DataFrame`"

    A variable is created for ever row in the pandas or polars `DataFrame` and labelled according to the values in that row. Column names become the dimension names. Pandas indexes are ignored.

    ```python
    import pandas as pd

    df = pd.DataFrame({"day": ["Mon", "Tue", "Wed", "Thu", "Fri"]})
    m.Example_1 = pf.Variable(df)
    ```

=== "`Series`"

    A pandas or polars `Series` is treated as a 1-column DataFrame. Pandas indexes are ignored.

    ```python
    import pandas as pd

    series = pd.Series(["Mon", "Tue", "Wed", "Thu", "Fri"], name="day")
    m.Example_2 = pf.Variable(series)
    ```

=== "`Index`"

    A pandas `Index` is treated like a `DataFrame`.

    ```python
    import pandas as pd

    series = pd.Index(["Mon", "Tue", "Wed", "Thu", "Fri"], name="day")
    m.Example_3 = pf.Variable(series)
    ```

=== "`dict`"

    Dictionaries are shortcuts for writing `pf.Variable(pl.DataFrame(dict_data))`.

    ```python
    m.Example_4 = pf.Variable({"day": ["Mon", "Tue", "Wed", "Thu", "Fri"]})
    ```

=== "`Set`"
    
    Pyoframe offers a [Set][pyoframe.Set] class to easily define dimensioned variables in a reusable way.

    ```python
    weekdays = pf.Set(day=["Mon", "Tue", "Wed", "Thu", "Fri"])
    m.Example_5 = pf.Variable(weekdays)
    ```


<!-- invisible-code-block: python
from polars.testing import assert_frame_equal

for df in [
    m.Example_1.data,
    m.Example_2.data,
    m.Example_3.data,
    m.Example_4.data,
    m.Example_5.data,
]:
    assert "day" in df.columns
    assert len(df) == 5

-->


## Formulate expressions

Mathematical expressions are represented by the [`Expression`][pyoframe.Expression] class which is automatically formed when standard arithmetic operators (`+`, `-`, `*`, `/`, `**`) are used to combine variables with numbers, other variables, or other Expressions. When computing mathematical operations, **Pyoframe will automatically align your labels and dimensions.**

For example, consider the following model:

```python
import pyoframe as pf

m = pf.Model()
m.Hours_Worked = pf.Variable({"day": ["Mon", "Tue", "Wed", "Thu", "Fri"]}, lb=0)
m.Hours_Sleep = pf.Variable({"day": ["Fri", "Thu", "Wed", "Tue", "Mon"]}, lb=0)
m.hours_remaining = 24 - m.Hours_Worked - m.Hours_Sleep
```

Notice how the order of the days in `Hours_Sleep` is reversed. This is no problem because Pyoframe detects that the dimensions match (since they're both named `day`) and will align the labels:

```pycon
>>> m.hours_remaining
<Expression (linear) height=5 terms=15>
┌─────┬───────────────────────────────────────────┐
│ day ┆ expression                                │
│ (5) ┆                                           │
╞═════╪═══════════════════════════════════════════╡
│ Mon ┆ 24 - Hours_Worked[Mon] - Hours_Sleep[Mon] │
│ Tue ┆ 24 - Hours_Worked[Tue] - Hours_Sleep[Tue] │
│ Wed ┆ 24 - Hours_Worked[Wed] - Hours_Sleep[Wed] │
│ Thu ┆ 24 - Hours_Worked[Thu] - Hours_Sleep[Thu] │
│ Fri ┆ 24 - Hours_Worked[Fri] - Hours_Sleep[Fri] │
└─────┴───────────────────────────────────────────┘

```

### Using parameters

Often, our models need to incorporate external data. To do this, we need to use **parameters**.

In Pyoframe, a parameter is actually just an [Expression][pyoframe.Expression] that does not contain any Variables (aka. a constant).

You can convert your data to a parameter by passing it to [`pf.Param(data)`][pyoframe.Param]. The last column of the data is always treated as the parameter value, and all other columns are treated as labels. (See [Param][pyoframe.Param] for other ways to create parameters.)

For example, consider the following code that integrates the `holidays` DataFrame into a pay calculation:

```python
import pandas as pd
import pyoframe as pf

holidays = pd.DataFrame(
    {"day": ["Mon", "Tue", "Wed", "Thu", "Fri"], "is_holiday": [0, 0, 0, 0, 1]}
)
base_pay = 20
holiday_bonus = 10

m = pf.Model()
m.is_holiday = pf.Param(holidays)
m.Hours_Worked = pf.Variable(holidays["day"], lb=0)
m.pay = m.Hours_Worked * (base_pay + m.is_holiday * holiday_bonus)
```

Here, `m.is_holiday` is a parameter Expression:

```pycon
>>> m.is_holiday
<Expression (parameter) height=5 terms=5>
┌─────┬────────────┐
│ day ┆ expression │
│ (5) ┆            │
╞═════╪════════════╡
│ Mon ┆ 0          │
│ Tue ┆ 0          │
│ Wed ┆ 0          │
│ Thu ┆ 0          │
│ Fri ┆ 1          │
└─────┴────────────┘

```

And the resulting `m.pay` Expression correctly incorporates the holiday bonus only on Fridays:

```pycon
>>> m.pay
<Expression (linear) height=5 terms=5>
┌─────┬──────────────────────┐
│ day ┆ expression           │
│ (5) ┆                      │
╞═════╪══════════════════════╡
│ Mon ┆ 20 Hours_Worked[Mon] │
│ Tue ┆ 20 Hours_Worked[Tue] │
│ Wed ┆ 20 Hours_Worked[Wed] │
│ Thu ┆ 20 Hours_Worked[Thu] │
│ Fri ┆ 30 Hours_Worked[Fri] │
└─────┴──────────────────────┘

```

Note that often, you can skip defining parameters because whenever a Pyoframe object is combined with a DataFrame, Pyoframe will automatically convert the DataFrame to a parameter Expression. For example, the following works just fine:

```pycon
>>> m.bonus_pay = m.Hours_Worked * holidays * holiday_bonus
>>> m.bonus_pay
<Expression (linear) height=5 terms=5>
┌─────┬──────────────────────┐
│ day ┆ expression           │
│ (5) ┆                      │
╞═════╪══════════════════════╡
│ Mon ┆ 0                    │
│ Tue ┆ 0                    │
│ Wed ┆ 0                    │
│ Thu ┆ 0                    │
│ Fri ┆ 10 Hours_Worked[Fri] │
└─────┴──────────────────────┘

```

### Transforms

The page on [Transforms](../concepts/special-functions.md) describes additional ways to formulate Expressions (e.g. using `.sum(…)`, `.map(…)`, `.next(…)`).

## Add constraints

Create constraints by using the `<=`, `>=`, and `==` operators between Expressions. For example:

```python
m.Con_Max_Weekly_Hours = m.Hours_Worked.sum() <= 40
```

You can easily relax a constraint using the [`.relax(cost, max)`][pyoframe.Constraint.relax] method.

(It might be helpful to know that, internally, Pyoframe rearranges all constraint `a <= b` into the standard form `a - b <= 0` so that only a single left-hand-side Expression needs to be stored.)

!!! tip "Naming constraints"
    I like prefixing constraint names with `Con_` to easily distinguish them from variables and expressions.

## Set the objective

Set the objective by assigning an Expression to either the `.minimize` or `.maximize` attribute of the Model. For example:

```python
m.minimize = m.Hours_Worked.sum()
```

Note that the objective Expression must be dimensionless; you cannot have multiple objectives with different labels. This is why we use `.sum()` to aggregate over all days.

## Optimize!

Optimizing your model is as simple as calling the `.optimize()` method:

```python
m.optimize()
```

Read the [Solver interface](../concepts/solver-access.md) page for more information on configuring and using solvers.

## Retrieve results

Use [`.solution`][pyoframe.Variable.solution] to read the optimal values of Variables after optimization (e.g. `m.Hours_Worked.solution`). For dimensioned variables, `.solution` returns a polars DataFrame.

Similarly, use [`.dual`][pyoframe.Constraint.dual] to read the dual values (aka. shadow prices) of Constraints (e.g. `m.Con_Max_Weekly_Hours.dual`).

You can also output your model problem or solution using [`.write(…)`][pyoframe.Model.write].

!!! info "Returning Pandas DataFrames"

    Pyoframe currently always returns Polars DataFrames but you can easily convert them to Pandas using [`.to_pandas()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_pandas.html#polars.DataFrame.to_pandas). In the future, we plan to add support for automatically returning Pandas DataFrames. [Upvote the issue](https://github.com/Bravos-Power/pyoframe/issues/47) if you'd like this feature.
