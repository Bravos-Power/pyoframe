## Install Pyoframe

```cmd
pip install pyoframe
```

## Install a solver

*[solver]: Solvers like HiGHS and Gurobi do the actual solving of your model. Pyoframe is a layer on top of the solver that makes it easy to build models and switch between solvers.

=== "HiGHS (free)"

    ```cmd
    pip install pyoframe[highs]
    ```

=== "Gurobi (commercial)"

    1. [Install Gurobi](https://www.gurobi.com/downloads/gurobi-software/) from their website.
    2. Ensure you have a valid Gurobi license installed on your machine.

    Note: installing Gurobi via pip will not work since we access Gurobi through its C API not through Python.

=== "Other Solvers"

    We'd be glad to add more solvers! Just [let us know](https://github.com/Bravos-Power/pyoframe/pull/79) what you'd like :)
