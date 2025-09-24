---
hide:
  - navigation
---
# Contribute

Contributions are more than welcome! Submit a pull request, or [open an issue](https://github.com/Bravos-Power/pyoframe/issues/new) and I (Martin) will gladly answer your questions on how to contribute.

## Setup a development environment

1. Clone this repository.
```console
git clone https://github.com/Bravos-Power/pyoframe
```

2. Install the dependencies. 
```console
pip install --editable .[dev,docs]
```

3. Install the pre-commit hooks. 
```console
pre-commit install
```

4. Run `pytest` to make sure everything is working. The only errors you should see are those related to a solver not being installed. If not, [open an issue](https://github.com/Bravos-Power/pyoframe/issues/new)!

## Writing documentation

We use [Material Docs](https://squidfunk.github.io/mkdocs-material/) for documentation with several plugins to enable features like automatically compiling the docstrings into the reference API. Please follow the [Google docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and the [Google style guide](https://developers.google.com/style). Additionally, all Python code blocks in Markdown files are tested using Sybil. To properly setup the tests refer to this [Sybil documentation](https://sybil.readthedocs.io/en/latest/markdown.html#code-blocks).

## Helpful commands

- `pytest`: Runs all the tests. If you'd like to generate coverage information just add the flag `--cov`.
- `mkdocs serve`: Generates the documentation locally. Navigate to [`http://127.0.0.1:8000/pyoframe/`](http://127.0.0.1:8000/pyoframe/) to check it out.
- `coverage html`: Generate a webpage to view the coverage information generated after having run `pytest --cov`.
- `python -m tests.test_examples`: Regenerate the files in the `results` folder of an example (e.g. `tests/examples/sudoku/results/**`). You should only run this if the result files need to be regenerated, for example, if model variable names have changed.
- `ruff check`: Ensures all the linter tests pass
- `ruff format`: Ensures the code is properly formatted (this is run upon commit if you've installed the pre-commit hooks)
- `doccmd --language=python --no-pad-file --command="ruff format" docs/`: to format the code in the documentation.
- `mike serve` to see a local version of the documentation with the version selector.

## Additional tips

For core developers:

- If you use `.unique`, `.join`, `.sort`, or `.group_by` on a Polars dataframe, make sure to set the `maintain_order` parameter appropriately (typically, `maintain_order=Config.maintain_order`).

For repository maintainers:

- Our CI pipeline on Github Actions requires a Gurobi and COPT license to run. If the Gurobi license expires, generate a new one and copy the contents of the `guorbi.lic` file into the `GUROBI_WLS` Github secret (Settings -> Secrets and variables -> actions). Similarly, if the COPT license expires, request a new academic license (or email COPT sales for a free one) and copy the contents of both license files to the matching Github secrets.
