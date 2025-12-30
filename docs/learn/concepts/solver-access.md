# Solver interface

Pyoframe provides a friendly API that allows you to read and set the various attributes and parameters your solver has to offer.

## Model attributes

Pyoframe lets you read and set solver attributes using `model.attr.<your-attribute>`. For example, if you'd like to prevent the solver from printing to the console you can do:

```python
m = pf.Model()
m.attr.Silent = True
```

Pyoframe supports a set of [standard attributes](https://metab0t.github.io/PyOptInterface/model.html#id1) as well as additional [Gurobi attributes](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/model.html) and [COPT attributes](https://guide.coap.online/copt/en-doc/attribute.html).

```pycon
>>> m.optimize()
>>> m.attr.TerminationStatus  # PyOptInterface attribute (always available)
<TerminationStatusCode.OPTIMAL: 2>
>>> m.attr.Status  # Gurobi attribute (only available with Gurobi)
2

```

## Model parameters

Every solver has a set of parameters that you can read or set using `model.params.<your-param>`.

=== "Gurobi"

    Refer to the list of [Gurobi parameters](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html).
    
    ```python
    m = pf.Model("gurobi")
    m.params.Method = 2  # Use barrier method
    m.params.TimeLimit = 100
    ```

=== "COPT"

    Refer to the list of [COPT parameters](https://guide.coap.online/copt/en-doc/parameter.html).
    
    ```python
    m = pf.Model("copt")
    m.params.RelGap = 0.01
    m.params.TimeLimit = 100
    ```

=== "HiGHS"

    Refer to the list of [HiGHS options](https://ergo-code.github.io/HiGHS/stable/options/definitions/).
    
    ```python
    m = pf.Model("highs")
    m.params.time_limit = 100.0
    m.params.mip_rel_gap = 0.01
    ```

=== "Ipopt"

    Refer to the list of [Ipopt options](https://coin-or.github.io/Ipopt/OPTIONS.html).
    
    ```python
    m = pf.Model("ipopt")
    m.params.tol = 1e-6
    m.params.max_iter = 1000
    ```
    
    !!! warning
        Ipopt does not support reading parameters (only setting them).


## Variable and constraint attributes

Similar to above, Pyoframe allows directly accessing the PyOptInterface or the solver's variable and constraint attributes.

```python
m = pf.Model()
m.X = pf.Variable()
m.X.attr.PrimalStart = 5  # Set initial value for warm start
```

If the variable or constraint is dimensioned, the attribute can accept/return a DataFrame instead of a constant.

## License configuration (COPT and Gurobi)

Both COPT and Gurobi support advanced license configurations through the `solver_env` parameter:

<!-- skip: start "Example servers don't actually work" -->

=== "COPT"

    ```python
    # Cluster configuration
    m = pf.Model("copt", solver_env={"CLIENT_CLUSTER": "cluster.example.com"})
    ```

=== "Gurobi"

    ```python
    # Compute server
    m = pf.Model(
        "gurobi",
        solver_env={"ComputeServer": "server.example.com", "ServerPassword": "password"},
    )
    ```

<!-- skip: end -->