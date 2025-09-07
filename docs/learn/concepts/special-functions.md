# Transforms

!!! info "Work in progress"

    This documentation could use some help. [Learn how you can contribute](../../contribute/index.md).

Pyoframe has a few special functions that make working with dataframes easy and intuitive. Here they are:

## `sum` and `sum_by`

## `Expression.map()`

## Adding elements with differing dimensions using `.over(…)`

<!-- TODO refine the use of the term "combine" below to clarify when .over() applies (addition, sub, constraints) and when it doesn't ("multiplication") -->

**To help catch mistakes, adding and subtracting expressions with differing dimensions is disallowed by default. [`.over(…)`][pyoframe.Expression.over] overrides this default, indicating that an addition or subtraction should be performed by "broadcasting" the differing dimensions.**

The following example helps illustrate when `.over(…)` should and shouldn't be used.

Say you're developing an optimization model to study aviation emissions. You'd like to add the air emissions with the ground emissions (emissions from [taxiing](https://en.wikipedia.org/wiki/Taxiing)) to create an expression representing the total emissions on a flight-by-flight basis. Unfortunately, doing so gives an error:

<!-- invisible-code-block: python 
```python
import pyoframe as pf
import polars as pl

air_data = pl.DataFrame({"flight_no": ["A4543", "K937"], "emissions": [1.4, 2.4]})
ground_data = pl.DataFrame(
    {"flight_number": ["A4543", "K937"], "emissions": [0.02, 0.05]}
)

model = pf.Model()
model.Fly = pf.Variable(air_data["flight_no"], vtype="binary")
model.air_emissions_by_flight = model.Fly * air_data
model.ground_emissions_by_flight = ground_data.to_expr()
```
-->


```pycon
>>> model.air_emissions_by_flight + model.ground_emissions_by_flight
Traceback (most recent call last):
...
pyoframe._constants.PyoframeError: Cannot add the two expressions below because their dimensions are different (['flight_no'] != ['flight_number']).
Expression 1:	air_emissions_by_flight
Expression 2:	ground_emissions_by_flight
If this is intentional, use .over(…) to broadcast. Learn more at https://bravos-power.github.io/pyoframe/learn/concepts/special-functions/#adding-elements-with-differing-dimensions-using-over

```

This error helps you catch a mistake. The error informs us that `model.air_emissions_by_flight` has dimension _`flight_no`_ but `model.ground_emissions_by_flight` has dimension _`flight_number`_ (not `flight_no`). Oops! Seems like the two datasets containing the emissions data had slightly different column names.

Benign mistakes like these are relatively common and Pyoframe's defaults help you catch these mistakes early. Now, lets examine a case where `.over(…)` is needed.

Say, you'd like to see what happens if, instead of minimizing total emissions, you were to minimize the emissions of the _most emitting flight_. Mathematically, you'd like to minimize $`E_{max}`$ where
$`E_{max} \geq e_i`$ for every flight $`i`$ with emissions $`e_i`$.

You might try the following in Pyoframe, but will get an error:

<!-- invisible-code-block: python 
```python
model.flight_emissions = (
    model.air_emissions_by_flight
    + model.ground_emissions_by_flight.rename({"flight_number": "flight_no"})
)
```
-->

```pycon
>>> model.E_max = pf.Variable()
>>> model.minimize = model.E_max
>>> model.emission_constraint = model.E_max >= model.flight_emissions
Traceback (most recent call last):
...
pyoframe._constants.PyoframeError: Cannot subtract the two expressions below because their dimensions are different ([] != ['flight_no']).
Expression 1:	E_max
Expression 2:	flight_emissions
If this is intentional, use .over(…) to broadcast. Learn more at https://bravos-power.github.io/pyoframe/learn/concepts/special-functions/#adding-elements-with-differing-dimensions-using-over

```

The error indicates that `E_max` has no dimensions while `flight_emissions` has dimensions `flight_no`. The error is raised because, by default, combining terms with differing dimensions is not allowed.

What we'd like to do is effectively 'copy' (aka. 'broadcast') `E_max` _over_ every flight number. `E_max.over("flight_no")` does just this:

```pycon
>>> model.E_max.over("flight_no")
<Expression terms=1 type=linear>
┌───────────┬────────────┐
│ flight_no ┆ expression │
╞═══════════╪════════════╡
│ *         ┆ E_max      │
└───────────┴────────────┘

```

Notice how applying `.over("flight_no")` added a dimension `flight_no` with value `*`. The asterix (`*`) indicates that `flight_no` will take the shape of whichever expression `E_max` is combined with. Since `E_max` is being combined with `flight_emissions`, `*` will be replaced with an entry for every flight number in `flight_emissions`. Now creating our constraint works properly:

```pycon
>>> model.emission_constraint = model.E_max.over("flight_no") >= model.flight_emissions
>>> model.emission_constraint
<Constraint 'emission_constraint' height=2 terms=6 type=linear>
┌───────────┬───────────────────────────────┐
│ flight_no ┆ constraint                    │
│ (2)       ┆                               │
╞═══════════╪═══════════════════════════════╡
│ A4543     ┆ E_max -1.4 Fly[A4543] >= 0.02 │
│ K937      ┆ E_max -2.4 Fly[K937] >= 0.05  │
└───────────┴───────────────────────────────┘

```



## `drop_unmatched` and `keep_unmatched`

## `DataFrame.to_expr()`

!!! abstract "Summary"
    [`pandas.DataFrame.to_expr()`](../../reference/pandas.DataFrame.to_expr.md) and [`polars.DataFrame.to_expr()`](../../reference/polars.DataFrame.to_expr.md) allow users to manually convert their DataFrames to Pyoframe [Expressions][pyoframe.Expression] when Pyoframe is unable to perform an automatic conversation.

Pyoframe conveniently allows users to use [Polars DataFrames](https://docs.pola.rs/api/python/stable/reference/dataframe/index.html) and [Pandas DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) in their mathematical expressions. To do so, Pyoframe automatically detects these DataFrames and converts them to Pyoframe [Expressions][pyoframe.Expression] whenever there is a mathematical operation (e.g., `*`, `-`, `+`) involving at least one Pyoframe object (e.g. [Variable][pyoframe.Variable], [Set][pyoframe.Set], [Expression][pyoframe.Expression], etc.).

However, if **neither** the left or right terms of a mathematical operation is a Pyoframe object, Pyoframe will not automatically convert DataFrames[^1]. In these situations, users can manually convert their DataFrames to Pyoframe expressions using `.to_expr()`.

Additionally, users should use `.to_expr()` whenever they wish to use [over][pyoframe.Expression.over], [drop_unmatched][pyoframe.Expression.drop_unmatched], or [keep_unmatched][pyoframe.Expression.keep_unmatched] on a DataFrame.

!!! info "Under the hood"
    How is `.to_expr()` a valid Pandas and Polars method? `import pyoframe` causes Pyoframe to [monkey patch](https://stackoverflow.com/questions/5626193/what-is-monkey-patching) the Pandas and Polars libraries. One of the patches adds the `.to_expr()` method to both `pandas.DataFrame` and `polars.DataFrame` (see [`monkey_patch.py`](https://github.com/Bravos-Power/pyoframe/tree/main/src/pyoframe)).

[^1]: After all, how could it? If a user decides to write code that adds two DataFrames together, Pyoframe shouldn't (and couldn't) interfere.

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
ValueError: Cannot create an expression with duplicate indices:
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