# Installation

## Step 1: Install Pyoframe

Install using your preferred package manager:

=== "pip"
    ```cmd
    pip install pyoframe
    ```
=== "uv"
    ```cmd
    uv add pyoframe
    ```

## Step 2: Install a solver

Pyoframe makes it easy to build models but the actual solving of your model is done by a separate solver. 
You'll need to install one of the following:

=== "HiGHS (open-source)"

    To install [HiGHS](https://highs.dev/) run:

    ```cmd
    pip install pyoframe[highs]
    ```

    !!! warning "Quadratics are unsupported in HiGHS"
        Pyoframe does not support quadratic constraints when using HiGHS due to limitations in pyoptinterface, the library we use to communicate with HiGHS.


=== "Gurobi (commercial)"

    To install Gurobi:

    1. [Download Gurobi](https://www.gurobi.com/downloads/gurobi-software/) from their website (login required) and follow the installation instructions.
    2. Ensure you have a valid Gurobi license installed on your machine.

    !!! info "pip installation not possible"
        Installing Gurobi via `pip` will not work. We use Gurobi's C API which is not available in the Python version of Gurobi.

=== "Ipopt (open-source, nonlinear)"

    To install [ipopt](https://coin-or.github.io/Ipopt/):

    1. Run: `pip install pyoframe[ipopt]`
    2. Download the [Ipopt binaries](https://github.com/coin-or/Ipopt/releases) from GitHub. Version 3.14.x is recommended since it is the latest version that we've tested.
    3. On Windows, unpack the zip and add the `bin` folder to your Path variable. If not on Windows, you may have to build the solver from source, see further details [here](https://metab0t.github.io/PyOptInterface/getting_started.html#ipopt).

    !!! warning "Continuous variables only"
        Ipopt is a nonlinear solver for continuous variables only. Use another solver if you need to use binary or integer variables. 

=== "Other solvers"

    We would gladly consider supporting other solvers. Create a [new issue](https://github.com/Bravos-Power/pyoframe/issues/new) or up-vote an existing one to show interest:

    - Issue tracking interest in [COPT solver](https://github.com/Bravos-Power/pyoframe/issues/143)
    - Issue tracking interest in [Mosek solver](https://github.com/Bravos-Power/pyoframe/issues/144)


