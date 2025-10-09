"""Pyomo implementation of the facility location benchmark.

Copyright (c) 2022: Miles Lubin and contributors

Use of this source code is governed by an MIT-style license that can be found
in the LICENSE.md file or at https://opensource.org/licenses/MIT.
See https://github.com/jump-dev/JuMPPaperBenchmarks
"""

import pyomo.environ as pyo
from benchmark_utils import PyomoBenchmark


class Bench(PyomoBenchmark):
    def build(self):
        N = self.size

        model = pyo.ConcreteModel()
        model.axis = pyo.RangeSet(1, N)
        model.x = pyo.Var(model.axis)
        model.obj = pyo.Objective(
            expr=lambda m: sum(m.x[i] for i in m.axis),
            sense=pyo.minimize,
        )

        return model
