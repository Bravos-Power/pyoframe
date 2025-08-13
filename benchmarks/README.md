# Pyoframe benchmarks

This folder contains the code and instructions needed to benchmark Pyoframe's performance to other libraries. For benchmarking, we use [`snakemake`](https://snakemake.github.io/) to produce the inputs and run the benchmarks.

## How to run the benchmarks

1. First install pyoframe: `pip install --editable .`
1. `cd benchmarks`
2. `pip install --editable .`
3. If running the JuMP benchmark:
    a. Install Julia: `curl -fsSL https://install.julialang.org | sh`
    b. Install the Julia dependencies: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
4. Edit `config.yaml` to your liking (e.g. specify the problems and libraries to benchmark).
5. Run `snakemake --cores 'all'`. This will run all the benchmarks and take a while.
6. View the plotted results in, for example, `facility_problem/results/benchmark_results.png`

### Running energy planning benchmark

You'll need to complete the following additional steps.
1. Install the dependencies for [`scikit-sparse`](https://github.com/scikit-sparse/scikit-sparse), typically `sudo apt-get install libsuitesparse-dev`
2. `pip install --editable .[energy-planning]


## Running energy model benchmark locally

1. Download the California Test System data. Specifically, place the [load data](https://drive.google.com/file/d/1Sz8st7g4Us6oijy1UYMPUvkA1XeZlIr8/view?usp=drive_link), [generation data](https://drive.google.com/file/d/1CxLlcwAEUy-JvJQdAfVydJ1p9Ecot-4d/view?usp=drive_link), and [line data](https://github.com/staadecker/CATS-CaliforniaTestSystem/blob/master/GIS/CATS_lines.json) in the `/benchmarks/energy_planning/data`.