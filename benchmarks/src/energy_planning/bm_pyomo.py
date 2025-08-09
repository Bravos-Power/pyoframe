"""Pyomo implementation of the facility location benchmark.

Copyright (c) 2022: Miles Lubin and contributors

Use of this source code is governed by an MIT-style license that can be found
in the LICENSE.md file or at https://opensource.org/licenses/MIT.
See https://github.com/jump-dev/JuMPPaperBenchmarks
"""

from benchmarks.utils import PyomoBenchmark


class Bench(PyomoBenchmark):
    def build(self):
        raise NotImplementedError()
