## Install Pyoframe
```
pip install pyoframe
```

## Install a solver

*[solver]: Solvers like HiGHS and Gurobi do the actual solving of your model. Pyoframe is a wrapper that makes it easy to build models but Pyoframe still needs a solver to work.

=== "HiGHS"

    `pip install pyoframe[highs]`

=== "Gurobi"

    1. [Install Gurobi](https://www.gurobi.com/downloads/gurobi-software/) from their website.
    2. Ensure you have a valid Gurobi license installed on your machine.

    Note: installing Gurobi via pip will not work since we access Gurobi through its C API not through Python.

=== "Other Solvers"

    We'd be glad to add more solvers! Just [let us know](https://github.com/Bravos-Power/pyoframe/pull/79) what you'd like :)
