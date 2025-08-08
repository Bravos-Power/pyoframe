"""Pyomo implementation of the facility location benchmark.

Copyright (c) 2022: Miles Lubin and contributors

Use of this source code is governed by an MIT-style license that can be found
in the LICENSE.md file or at https://opensource.org/licenses/MIT.
See https://github.com/jump-dev/JuMPPaperBenchmarks
"""

import pyomo.environ as pyo

from benchmarks.utils import PyomoBenchmark


class Bench(PyomoBenchmark):
    def build(self):
        N = self.size

        model = pyo.ConcreteModel()
        model.axis = pyo.RangeSet(1, N)
        model.x = pyo.Var(model.axis, model.axis)
        model.y = pyo.Var(model.axis, model.axis)
        model.obj = pyo.Objective(
            expr=lambda m: sum(2 * m.x[i, j] for i in m.axis for j in m.axis)
            + sum(m.y[i, j] for i in m.axis for j in m.axis),
            sense=pyo.minimize,
        )

        model.con1 = pyo.Constraint(
            model.axis, model.axis, rule=lambda m, i, j: m.x[i, j] - m.y[i, j] >= i
        )
        model.con2 = pyo.Constraint(
            model.axis, model.axis, rule=lambda m, i, j: m.x[i, j] + m.y[i, j] >= 0
        )

        return model
