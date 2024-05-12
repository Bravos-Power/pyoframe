# Pyoframe: Fast and low-memory linear programming models 

[![codecov](https://codecov.io/gh/Bravos-Power/pyoframe/graph/badge.svg?token=8258XESRYQ)](https://codecov.io/gh/Bravos-Power/pyoframe)
[![Build](https://github.com/Bravos-Power/pyoframe/actions/workflows/ci.yml/badge.svg)](https://github.com/Bravos-Power/pyoframe/actions/workflows/ci.yml)
[![Docs](https://github.com/Bravos-Power/pyoframe/actions/workflows/publish_doc.yml/badge.svg)](https://Bravos-Power.github.io/pyoframe/reference/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues Needing Triage](https://img.shields.io/github/issues-search/Bravos-Power/pyoframe?query=no%3Alabel%20is%3Aopen&label=Needs%20Triage)](https://github.com/Bravos-Power/pyoframe/issues?q=is%3Aopen+is%3Aissue+no%3Alabel)
[![Open Bugs](https://img.shields.io/github/issues-search/Bravos-Power/pyoframe?query=label%3Abug%20is%3Aopen&label=Open%20Bugs)](https://github.com/Bravos-Power/pyoframe/issues?q=is%3Aopen+is%3Aissue+label%3Abug)


A library to rapidly and memory-efficiently formulate large and sparse optimization models using Pandas or Polars dataframes.

## Contribute

Contributions are welcome! See [`CONTRIBUTE.md`](./CONTRIBUTE.md).

## Acknowledgments

Martin Staadecker first created this library while working for [Bravos Power](https://www.bravospower.com/) The library takes inspiration from Linopy and Pyomo, two prior libraries for optimization for which we are thankful.

## Troubleshooting Common Errors

### `datatypes of join keys don't match`

Often, this error indicates that two dataframes in your inputs representing the same dimension have different datatypes (e.g. 16bit integer and 64bit integer). This is not allowed and you should ensure for the same dimensions, datatypes are identical.