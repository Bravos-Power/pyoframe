# Solver interface  

## Model attributes

Pyoframe lets you read and set solver attributes using `model.attr.<your-attribute>`. For example, if you'd like to prevent the solver from printing to the console you can do:

```python
m = pf.Model()
m.attr.Silent = True
```

Pyoframe supports all [PyOptInterface attributes](https://metab0t.github.io/PyOptInterface/model.html#id1) and, when using Gurobi, all [Gurobi attributes](https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/model.html).

```pycon
>>> m.optimize()
>>> m.attr.TerminationStatus  # PyOptInterface attribute (always available)
<TerminationStatusCode.OPTIMAL: 2>
>>> m.attr.Status  # Gurobi attribute (only available with Gurobi)
2

```

## Model parameters

How to set solver parameters depends on the solver you're using, check the [PyOptInterface documentation](https://metab0t.github.io/PyOptInterface/getting_started.html) to review which parameters and attributes can be accessed for each solver:

=== "Gurobi"
    Gurobi supports the `params` interface for all parameters listed in their [documentation](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html).
    
    ```python
    m = pf.Model("gurobi")
    m.params.Method = 2  # Use barrier method
    m.params.TimeLimit = 100
    ```

=== "COPT"
    COPT supports the `params` interface for all parameters listed in their [documentation](https://guide.coap.online/copt/en-doc/parameter.html).
    
    ```python
    m = pf.Model("copt")
    m.params.RelGap = 0.01
    m.params.TimeLimit = 100
    ```

=== "HiGHS"
    HiGHS parameters must be set using the raw parameter interface. See [HiGHS options](https://ergo-code.github.io/HiGHS/stable/options/definitions/) for available parameters.
    
    ```python
    m = pf.Model("highs")
    m.poi.set_raw_parameter("time_limit", 100.0)
    m.poi.set_raw_parameter("mip_rel_gap", 0.01)
    ```

=== "Ipopt"
    Ipopt parameters must be set using the raw parameter interface. See [Ipopt options](https://coin-or.github.io/Ipopt/OPTIONS.html) for available parameters.
    
    ```python
    m = pf.Model("ipopt")
    m.poi.set_raw_parameter("tol", 1e-6)
    m.poi.set_raw_parameter("max_iter", 1000)
    ```
    
    !!! note
        Ipopt does not support `get_raw_parameter` to retrieve parameter values.


## Variable and constraint attributes

Similar to above, Pyoframe allows directly accessing the PyOptInterface or the solver's variable and constraint attributes.

```python
m = pf.Model()
m.X = pf.Variable()
m.X.attr.PrimalStart = 5  # Set initial value for warm start
```

If the variable or constraint is dimensioned, the attribute can accept/return a DataFrame instead of a constant.

## Solver-specific raw access

For advanced users who need access to solver-specific features not exposed through the standard interface:

```python
# Get raw parameter value
value = m.poi.get_raw_parameter("ParameterName")

# Set raw parameter value
m.poi.set_raw_parameter("ParameterName", value)

# Get raw attribute value
attr_value = m.poi.get_raw_attribute("AttributeName")
```

## License configuration (COPT and Gurobi)

Both COPT and Gurobi support advanced license configurations through the `solver_env` parameter:

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
