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

=== "Ipopt (free, nonlinear)"

    1. [Download Ipopt binaries](https://github.com/coin-or/Ipopt/releases) from their releases. Version 3.14.x is the only one tested.
    2. Ensure to add the bin/ folder from the installation to your Path variable, since the binaries need to be accesed. Refer to [PyOptInterface's documentation](https://metab0t.github.io/PyOptInterface/getting_started.html#ipopt) for more detials.
    3. 
    ```cmd
    pip install pyoframe[ipopt]
    ```
    Note: Ipopt is a nonlinear solver for continuous variables only, **do not** use Ipopt if your problem has integer variables. 

=== "Other Solvers"

    We'd be glad to add more solvers! Just [let us know](https://github.com/Bravos-Power/pyoframe/pull/79) what you'd like :)
