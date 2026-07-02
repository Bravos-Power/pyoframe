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

## Running the test suite

Run `pytest` to execute the test suite. (If you'd like to view coverage information add the flag `--cov` and then run `coverage html`.)

The only errors you should see when running the test suite are those related to a solver not being installed.

Pyoframe has several types of tests.

1. Your typical unit tests under the `tests/` folder.

2. Integration tests in `tests/test_examples.py`. These tests will run all the models in `tests/examples` and make sure that your changes haven't altered the model results (stored under `tests/examples/<model>/results`). In the rare cases where you _want_ the model results to change (e.g. if you've changed the model), you can regenerate the results using `python -m tests.test_examples`.

3. Doctests in the docstrings of the source code (`src/`).

4. Documentation tests (in `docs/`). All Python code blocks in the documentation are run to ensure the documentation doesn't become outdated. This is done using Sybil. Refer to the [Sybil documentation](https://sybil.readthedocs.io/en/latest/markdown.html#code-blocks) to learn how to create setup code or skip code blocks you don't wish to test.

!!! warning "Non-breaking spaces"
    Be aware that Pyoframe uses non-breaking spaces to improve the formatting of expressions. If your Sybil tests are unexpectedly failing, make sure that the expected output contains all the needed non-breaking spaces.

## Writing documentation

You can preview the documentation website by running `mkdocs serve` and navigating to [`http://127.0.0.1:8000/pyoframe/`](http://127.0.0.1:8000/pyoframe/).

We use [Material Docs](https://squidfunk.github.io/mkdocs-material/) for documentation with several plugins to enable features like automatically compiling the docstrings into the reference API. Please follow the [Google docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and the [Google style guide](https://developers.google.com/style). We use Mike to allow readers to view the documentation for previous releases (preview available via `mike serve`).

## Linting and formatting

We use Ruff for linting and formatting. The pre-commit hooks will run `ruff format` on commit. You should also make sure `ruff check` returns no errors before submitting a PR. To format code blocks in the documentation run: `doccmd --language=python --no-pad-file --command="ruff format" docs/`.

## Additional tips

I recommend skimming or reading the [Internal Details](../learn/concepts/internals.md) page for some background on how Pyoframe works.

For core developers:

- If you use `.unique`, `.join`, `.sort`, or `.group_by` on a Polars dataframe, make sure to set the `maintain_order` parameter appropriately (typically, `maintain_order=Config.maintain_order`).

## Details for Pyoframe maintainers

Documentation site analytics are available via [GoatCounter](https://pyoframe.goatcounter.com/).

Our Github Actions CI pipeline requires a Gurobi, COPT, and Mosek license to run. Occasionally, these licenses must be renewed.

- For Gurobi, the license can typically be [extended](https://license.gurobi.com/manager/licenses). Otherwise, [request](https://portal.gurobi.com/iam/licenses/list) a new 90-day WLS Compute Server license and copy the contents of the `guorbi.lic` file into the `GUROBI_WLS` Github secret (Settings -> Secrets and variables -> actions).

- For COPT, [request a new license](https://www.cardopt.com/copt) (or email COPT sales) and copt the content of both the `.dat` and `.lic` files into the corresponding Github secrets. Note that the license should be requested for a Linux computer with usernname `runner`. Also note that the COPT version used in the Github CI actions will need to be updated (in 3 places) to match the license version.

- For Mosek, a license can be [requested](https://www.mosek.com/products/academic-licenses/) and the contents of the `.lic` file added to the Github secret.
