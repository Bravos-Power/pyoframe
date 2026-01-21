# Create variables

To create a variable, attach it to a `Model`:

```python
import pyoframe as pf

m = pf.Model()

m.Var_Name = pf.Variable()
```

The variable can later be accessed via the model attribute,

```pycon
>>> m.Var_Name
<Variable 'Var_Name' >
Var_Name

```

!!! tip "Tip: Use uppercase names for variables"

    Uppercase names for variables (i.e. `Var_Name` not `var_name`) makes variables (the most important part of your model) easy to distinguish from other model attributes.


## Set bounds

By default, variables are unbounded. To set a lower or upper bound, use the `lb` or `ub` arguments. For example,

```python
m.Positive_Var = pf.Variable(lb=0)
```

!!! tip "Bounds can be expressions"

    `lb` and `ub` accepts fully formed [Pyoframe expressions](./create-expressions.md), not only constants.

## Set domain

By default, variables are continuous. To create a binary or integer variable use the `vtype` argument:

```python
m.Binary_Var = pf.Variable(vtype="binary")
m.Integer_Var = pf.Variable(vtype="integer")
```

## Use dimensions and labels

Passing a DataFrame to `pf.Variable` will create a variable for every row in the DataFrame. I call this a _dimensioned variable_.

=== "pandas"

    ```python
    import pandas as pd
    import pyoframe as pf

    years = pd.DataFrame({"year": [2025, 2026, 2027]})

    m = pf.Model()
    m.Yearly_Var = pf.Variable(years)
    ```

=== "polars"

    ```python
    import polars as pl
    import pyoframe as pf

    years = pl.DataFrame({"year": [2025, 2026, 2027]})

    m = pf.Model()
    m.Yearly_Var = pf.Variable(years)
    ```

Notice how the DataFrame's column name becomes the dimension name and the DataFrame's values become the variable's labels:

```pycon
>>> m.Yearly_Var
<Variable 'Yearly_Var' height=3>
┌──────┬──────────────────┐
│ year ┆ variable         │
│ (3)  ┆                  │
╞══════╪══════════════════╡
│ 2025 ┆ Yearly_Var[2025] │
│ 2026 ┆ Yearly_Var[2026] │
│ 2027 ┆ Yearly_Var[2027] │
└──────┴──────────────────┘

```

!!! warning "Labels must be unique"

    An error will be raised if the input DataFrame contains duplicate rows since every variable must have its own unique label.

### Combine dimensions

Passing multiple DataFrames to `pf.Variable()` will create a variable for every row in the [cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) of the DataFrames.


=== "pandas"

    ```python
    years = pd.DataFrame({"year": [2025, 2026, 2027]})
    locations = pd.DataFrame({"city": ["Toronto", "Mexico City"]})

    m = pf.Model()
    m.Cartesian_Var = pf.Variable(years, locations)
    ```

=== "polars"

    ```python
    years = pl.DataFrame({"year": [2025, 2026, 2027]})
    locations = pl.DataFrame({"city": ["Toronto", "Mexico City"]})

    m = pf.Model()
    m.Cartesian_Var = pf.Variable(years, locations)
    ```

```pycon
>>> m.Cartesian_Var
<Variable 'Cartesian_Var' height=6>
┌──────┬─────────────┬─────────────────────────────────┐
│ year ┆ city        ┆ variable                        │
│ (3)  ┆ (2)         ┆                                 │
╞══════╪═════════════╪═════════════════════════════════╡
│ 2025 ┆ Toronto     ┆ Cartesian_Var[2025,Toronto]     │
│ 2025 ┆ Mexico City ┆ Cartesian_Var[2025,Mexico_City] │
│ 2026 ┆ Toronto     ┆ Cartesian_Var[2026,Toronto]     │
│ 2026 ┆ Mexico City ┆ Cartesian_Var[2026,Mexico_City] │
│ 2027 ┆ Toronto     ┆ Cartesian_Var[2027,Toronto]     │
│ 2027 ┆ Mexico City ┆ Cartesian_Var[2027,Mexico_City] │
└──────┴─────────────┴─────────────────────────────────┘

```

!!! tip "Use a multi-column DataFrame to create sparse variables"

    An alternative way to create a variable with multiple dimensions (e.g. `year` and `city`) is to pass a single DataFrame with multiple columns to `pf.Variable`. This approach lets you control exactly which rows to include, allowing for sparsely populated variables instead of the cartesian product.

### Other approaches

<!-- invisible-code-block: python
years = pl.DataFrame({"year": [2025, 2026, 2027]})

m = pf.Model()
m.Yearly_Var = pf.Variable(years)
-->


DataFrames are not the only way to create a dimensioned variable. In the following examples, all the `m.Yearly_Var` are equivalent.

=== "Pyoframe sets"
    
    Pyoframe offers a [`Set`][pyoframe.Set] class to easily define dimensioned variables in a reusable way.

    ```python
    years = pf.Set(year=[2025, 2026, 2027])  # define once, reuse for multiple variables
    m.Yearly_Var_1 = pf.Variable(years)
    ```

=== "Dictionaries"

    Dictionaries are shortcuts for writing `pf.Variable(pl.DataFrame(dict_data))`.

    ```python
    m.Yearly_Var_2 = pf.Variable({"year": [2025, 2026, 2027]})
    ```

=== "Other Pyoframe objects"

    Passing a Pyoframe object such as an expression or another variable to `pf.Variable()` will create a variable with the same labels as the object.

    ```python
    m.Yearly_Var_3 = pf.Variable(
        m.Yearly_Var
    )  # Creates a variable with the same labels as m.Yearly_Var
    ```

=== "Series or indexes"

    A pandas `Index` or `Series` (or a polars `Series`) is treated as a DataFrame.

    ```python
    years = pd.Series([2025, 2026, 2027], name="year")
    m.Yearly_Var_4 = pf.Variable(years)
    ```

    ```python
    years = pd.Index([2025, 2026, 2027], name="year")
    m.Yearly_Var_5 = pf.Variable(years)
    ```





<!-- invisible-code-block: python
from polars.testing import assert_frame_equal

for con in [
    m.Yearly_Var,
    m.Yearly_Var_1,
    m.Yearly_Var_2,
    m.Yearly_Var_3,
    m.Yearly_Var_4,
    m.Yearly_Var_5,
]:
    assert "year" in con.data.columns
    assert len(con.data) == 3

-->


