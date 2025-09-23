# Install

## Step 1: Install Pyoframe

Install Pyoframe using your preferred package manager:

=== "pip"

    ```cmd
    pip install pyoframe
    ```

=== "uv"

    ```cmd
    uv add pyoframe
    ```

## Step 2: Choose a solver

Pyoframe makes it easy to **build** models, but a separate solver is needed to **solve** the model after it is built. Use the compatibility table below to choose a solver that fits your needs. If you're unsure, choose HiGHS. Note that both Gurobi and COPT have academic licensing available!

| Compatibility table                                           | HiGHS (free) | Gurobi (paid) | COPT (paid) | Ipopt (free) |
| ------------------------------------------------------------- | ------------ | ------------- | ----------- | ------------ |
| Linear programs (LPs)                                         | ✅           | ✅            | ✅          | ✅           |
| Mixed integer programs (MIPs)                                 | ✅*          | ✅            | ✅          | ❌           |
| Quadratic objective (convex)                                  | ✅           | ✅            | ✅          | ✅           |
| Quadratic objective (non-convex)                              | ❌           | ✅            | ❌          | ✅           |
| Quadratic constraints (convex)                                | ❌           | ✅            | ✅          | ✅           |
| Quadratic constraints (non-convex)                            | ❌           | ✅            | ❌          | ✅           |
| *Integer variables cannot be used with quadratic objectives.  |

!!! tip "Don't see your preferred solver?"
    Don't hesitate to [request another solver](https://github.com/Bravos-Power/pyoframe/issues/144). We can easily add support for other solvers, particularly Mosek, given sufficient interest.

## Step 3: Install the solver

Select your chosen solver and follow the installation instructions.

=== "HiGHS"

    To install [HiGHS](https://highs.dev/) run:

    ```cmd
    pip install pyoframe[highs]
    ```

=== "Gurobi"

    To install Gurobi:

    1. [Download Gurobi](https://www.gurobi.com/downloads/gurobi-software/) from their website (login required) and follow the installation instructions.
    2. Ensure you have a valid Gurobi license installed on your machine.

    !!! warning "Do not install Gurobi using `pip`"

        You should not install Gurobi using `pip` because Pyoframe uses Gurobi's C API, which the `pip` installation does not include.

=== "COPT"

    To install [COPT](https://www.shanshu.ai/copt):
    
    1. Download COPT from the mail they send after requesting a license and follow the installation instructions.
    2. Ensure you have a valid COPT license installed on your machine.
    3. Set the `COPT_HOME` environment variable to point to your COPT installation directory.
    
    !!! info "License configuration"
        COPT supports floating, cluster, and web licenses. Configure these through the `solver_env` parameter when creating your model if needed:
        ```python
        m = pf.Model("copt", solver_env={"FloatingServer": "server:port"})
        ```

=== "Ipopt"

    To install [ipopt](https://coin-or.github.io/Ipopt/):

    1. Run:
    ```console
    pip install pyoframe[ipopt]
    ```
    2. Download the [Ipopt binaries](https://github.com/coin-or/Ipopt/releases) from GitHub. Version 3.14.x is recommended since it is the latest version that we've tested.
    3. On Windows, unpack the zip and add the `bin` folder to your Path variable. If not on Windows, you may have to build the solver from source, see further details [here](https://metab0t.github.io/PyOptInterface/getting_started.html#ipopt).
