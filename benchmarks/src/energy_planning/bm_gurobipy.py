"""GurobiPy implementation of the facility location benchmark.

Copyright (c) 2023: Yue Yang

Use of this source code is governed by an MIT-style license that can be found
in the LICENSE.md file or at https://opensource.org/licenses/MIT.
"""

from benchmark_utils import GurobiPyBenchmark


class Bench(GurobiPyBenchmark):
    def build(self):
        raise NotImplementedError()
