## Install Pyoframe
```
pip install pyoframe
```

## Install your solver

=== "Gurobi"

    **Don't install via pip!** We use Gurobi's C API so the pip (Python) install won't work. [Install directly from the Gurobi website](https://www.gurobi.com/downloads/gurobi-software/). Once installed, ensure you have an active license. 

=== "HiGHS"

    When installing pyoframe just use: `pip install pyoframe[highs]`

=== "Other Solvers"
    
    We'd be glad to add more solvers! Just [let us know](https://github.com/Bravos-Power/pyoframe/pull/79) what you'd like :)
