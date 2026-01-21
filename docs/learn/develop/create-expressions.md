# Create expressions

Mathematical expressions in Pyoframe are represented by the [`Expression`][pyoframe.Expression] class and can be created in a few ways.

## Using arithmetic operators

Expressions are automatically created whenever standard arithmetic operators (`+`, `-`, `*`, `/`, `**`) are used between Pyoframe objects. For example, the following code creates the expression `m.hours_remaining`:

```python
import pyoframe as pf

m = pf.Model()
m.Hours_Worked = pf.Variable({"day": ["Mon", "Tue", "Wed", "Thu", "Fri"]}, lb=0)
m.Hours_Sleep = pf.Variable({"day": ["Fri", "Thu", "Wed", "Tue", "Mon"]}, lb=0)
m.hours_remaining = 24 - m.Hours_Worked - m.Hours_Sleep
```

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

!!! warning "Pyoframe always aligns labels and dimensions"

    Pyoframe always performs operations label-by-label. For example, Pyoframe subtracted `m.Hours_Sleep` from `m.Hours_Worked` using the labels; the fact that the days were listed in reverse order in `m.Hours_Sleep` (see above) does not matter.

    When the left- and/or right-hand side expressions have labels not present in the other, it may be necessary to use `.keep_extras()` or `.drop_extras()` to specify how these extra labels should be handled. Similarly, if one of the two operands is missing a dimension, it may be necessary to use `.over` to force broadcasting. Read [Addition and its quirks](../concepts/addition.md) to learn more.



## Using parameters

External data can be incorporated into an optimization problem by using [`pf.Param(data)`][pyoframe.Param] which converts a DataFrame into a Pyoframe expression. The last column of the DataFrame will be treated as the expression value, and all other columns will be treated as labels. For example, the following code creates a Pyoframe expression equal to `1` on Friday and `0` otherwise.

```python
import pandas as pd

is_holiday = pd.DataFrame(
    {"day": ["Mon", "Tue", "Wed", "Thu", "Fri"], "is_holiday": [0, 0, 0, 0, 1]}
)
```

```pycon
>>> pf.Param(is_holiday)
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

The expression can then be used like any other Pyoframe object:

```python
m.holiday_hours_worked = m.Hours_Worked * pf.Param(is_holiday)
```

Note that `pf.Param` is automatically applied when a Pyoframe object is operated with a DataFrame so the previous line can be simplified to

```python
m.holiday_hours_worked = m.Hours_Worked * is_holiday
```

!!! tip "`pf.Param` also accepts file paths"

    `pf.Param` also accepts a file path to a `.csv` or `.parquet` file, see the [`Param`][pyoframe.Param] API documentation to learn more.

## Using transforms

The functions [`sum`][pyoframe.Expression.sum], [`sum_by`][pyoframe.Expression.sum_by], [`map`][pyoframe.Expression.map], [`next`][pyoframe.Variable.next], [`rolling_sum`][pyoframe.Expression.rolling_sum], and [`within`][pyoframe.Expression.within] are _transforms_ that make it easy to convert an expression or variable from one shape to another. For example, `sum` can be used to collapse a dimensioned expression into a dimensionless one:

```pycon
>>> m.Hours_Worked.sum()
<Expression (linear) terms=5>
Hours_Worked[Mon] + Hours_Worked[Tue] + Hours_Worked[Wed] + Hours_Worked[Thu] + Hours_Worked[Fri]

```