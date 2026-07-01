# Access results

## Access variable solutions

Access a variable's solution using the [`.solution`][pyoframe.Variable.solution] property.

<!-- invisible-code-block: python
import pyoframe as pf

m = pf.Model()
m.Hours_Worked = pf.Variable({"day": ["Mon", "Tue", "Wed", "Thu", "Fri"]}, lb=8, ub=8)
m.optimize()

-->

```pycon
>>> m.Hours_Worked.solution
┌─────┬──────────┐
│ day ┆ solution │
│ --- ┆ ---      │
│ str ┆ f64      │
╞═════╪══════════╡
│ Mon ┆ 8.0      │
│ Tue ┆ 8.0      │
│ Wed ┆ 8.0      │
│ Thu ┆ 8.0      │
│ Fri ┆ 8.0      │
└─────┴──────────┘

```

If the Variable is dimensioned, a DataFrame is returned. Otherwise, a `float` (or `int` for integer variables) is returned.

!!! warning "Pandas users"
    By default, Polars DataFrames are returned since that is what Pyoframe uses internally. Users who prefer Pandas can either convert the Polars DataFrame to Pandas using [`.to_pandas()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_pandas.html#polars.DataFrame.to_pandas) or can set [Config.output_pandas][pyoframe._Config.output_pandas] to `True` so that this conversion is performed automatically.

## Access dual values

Access the dual values (i.e., shadow prices) of a constraint using the [`Constraint.dual`][pyoframe.Constraint.dual] property in the same way that one [accesses a variable's solution](#access-variable-solutions).

## Evaluate an expression

Evaluate an expression using [.evaluate()][pyoframe.Expression.evaluate].

```pycon
>>> total_hours_expr = m.Hours_Worked.sum()
>>> total_hours_expr.evaluate()
40.0

```

## Output results to a file

Output your entire model problem or solution to a file using [`.write(…)`][pyoframe.Model.write].