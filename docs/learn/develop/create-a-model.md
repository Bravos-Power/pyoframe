# Create a model

To create a model write:

```python
import pyoframe as pf

m = pf.Model()
```

## Specify a solver

By default, Pyoframe will try to use whichever solver is installed on your computer. To specify a particular solver, use the `solver` argument.

=== "Gurobi"

    ```python
    m = pf.Model(solver="gurobi")
    ```

=== "HiGHS"

    ```python
    m = pf.Model(solver="highs")
    ```

=== "COPT"

    ```python
    m = pf.Model(solver="copt")
    ```

=== "Ipopt"

    ```python
    m = pf.Model(solver="ipopt")
    ```

## Advanced options

Additional options are detailed in the [`Model`][pyoframe.Model] API documentation.
