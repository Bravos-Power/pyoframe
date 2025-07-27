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

Pyoframe makes it easy to **build** models, but a separate solver is needed to **solve** the model after it is built. Use the compatibility table below to choose a solver that fits your needs. If you're unsure, choose HiGHS.

| Compatibility table | HiGHS (free) | Gurobi (paid) | Ipopt (free) |
| --- | --- | --- | ---|
| Linear programs (LPs) | ️✅ | ️✅ | ️✅ |
| Mixed integer programs (MIPs) | ️✅* | ️✅ | ❌ |
| Quadratic objective (convex) | ️✅ | ️✅ | ️✅ |
| Quadratic objective (non-convex) | ❌ | ️✅ | ️✅ |
| Quadratic constraints (convex and non-convex) | ❌  | ️✅ | ️✅ |
| *Integer variables cannot be used with quadratic objectives. |

!!! tip "Don't see your preferred solver?"
    Don't hesitate to [request another solver](https://github.com/Bravos-Power/pyoframe/issues/144). We can easily add support for other solvers, particularly COPT and Mosek, given sufficient interest.


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

    !!! warning "Cannot install using `pip`"
        You cannot install Gurobi using `pip` because Pyoframe uses Gurobi's C API, which `pip` does not install.

=== "Ipopt"

    To install [ipopt](https://coin-or.github.io/Ipopt/):

    1. Run: `pip install pyoframe[ipopt]`
    2. Download the [Ipopt binaries](https://github.com/coin-or/Ipopt/releases) from GitHub. Version 3.14.x is recommended since it is the latest version that we've tested.
    3. On Windows, unpack the zip and add the `bin` folder to your Path variable. If not on Windows, you may have to build the solver from source, see further details [here](https://metab0t.github.io/PyOptInterface/getting_started.html#ipopt).

    

