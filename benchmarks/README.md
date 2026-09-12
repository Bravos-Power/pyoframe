# Pyoframe benchmarks

This folder contains the code and instructions needed to benchmark Pyoframe's performance to other libraries. For benchmarking, we use [`snakemake`](https://snakemake.github.io/) to produce the inputs and run the benchmarks.

## How to run the benchmarks

1. First install pyoframe: `pip install --editable .`
1. `cd benchmarks`
2. `pip install --editable .`
3. If running the JuMP benchmark:
    a. Install Julia: `curl -fsSL https://install.julialang.org | sh`
    b. Install the Julia dependencies: `julia --project=. -e 'using Pkg; Pkg.resolve()'`
4. If running the AMPL benchmark:
    a. Install the AMPL-Gurobi connector (v13.0.2): `python -m amplpy.modules install gurobi==20260624`
    b. Active your AMPL license: `python -m amplpy.modules activate <your_license_id>`
4. Edit `config.yaml` to your liking (e.g. specify the problems and libraries to benchmark).
4. It's a good idea to run `python test.py` to make sure everything works. Look at the saved logs and check that all libraries are using the same version of gurobi.
5. Run `python run.py`. This will run all the benchmarks and take a while.
6. Run `python plot.py` to generate the plots.
6. View the plotted results in, for example, `results/facility_location/`

### Running energy planning benchmark

You'll need to complete the following additional steps.
1. Install the dependencies for [`scikit-sparse`](https://github.com/scikit-sparse/scikit-sparse), typically `sudo apt-get install libsuitesparse-dev`
2. `pip install --editable .[energy-planning]


## Running energy model benchmark locally

1. Download the California Test System data. Specifically, place the [load data](https://drive.google.com/file/d/1Sz8st7g4Us6oijy1UYMPUvkA1XeZlIr8/view?usp=drive_link), [generation data](https://drive.google.com/file/d/1CxLlcwAEUy-JvJQdAfVydJ1p9Ecot-4d/view?usp=drive_link), and [line data](https://github.com/staadecker/CATS-CaliforniaTestSystem/blob/master/GIS/CATS_lines.json) in the `/benchmarks/energy_planning/data`.

## Notes for self

### To change Gurobi versions you need to

1. Update the `GUROBI_HOME` env var to point to the new local installation
2. Update the gurobipy version (pinned in `pyproject.toml`)
3. Update the AMPL-Gurobi connector (see pin in above instructions)
4. Rebuild the Gurobi Julia package with the flag to use the local installation (`export GUROBI_JL_USE_GUROBI_JLL=false && julia --project=. -e 'using Pkg; Pkg.build("Gurobi")'`).
5. Delete and regenerate the Julia system images (by running `test.py` and then `run.py`).