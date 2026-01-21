# Read results

Use [`.solution`][pyoframe.Variable.solution] to read the optimal values of Variables after optimization (e.g. `m.Hours_Worked.solution`). For dimensioned variables, `.solution` returns a polars DataFrame.

Similarly, use [`.dual`][pyoframe.Constraint.dual] to read the dual values (aka. shadow prices) of Constraints (e.g. `m.Con_Max_Weekly_Hours.dual`).

You can also output your model problem or solution using [`.write(…)`][pyoframe.Model.write].

!!! info "Returning Pandas DataFrames"

    Pyoframe currently always returns Polars DataFrames but you can easily convert them to Pandas using [`.to_pandas()`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_pandas.html#polars.DataFrame.to_pandas). In the future, we plan to add support for automatically returning Pandas DataFrames. [Upvote the issue](https://github.com/Bravos-Power/pyoframe/issues/47) if you'd like this feature.
