"""GurobiPy implementation of the facility location benchmark.

Copyright (c) 2023: Yue Yang

Use of this source code is governed by an MIT-style license that can be found
in the LICENSE.md file or at https://opensource.org/licenses/MIT.
"""

from benchmark_utils.gurobipy import GurobiPyBenchmark
from gurobipy import GRB, Model, quicksum


class Bench(GurobiPyBenchmark):
    def build(self):
        N = self.size

        m = Model()

        x = m.addVars(range(N), range(N))
        y = m.addVars(range(N), range(N))

        for i in range(N):
            for j in range(N):
                m.addConstr(x[i, j] - y[i, j] >= i)
                m.addConstr(x[i, j] + y[i, j] >= 0)

        m.setObjective(
            quicksum(2 * x[i, j] for i in range(N) for j in range(N))
            + quicksum(y[i, j] for i in range(N) for j in range(N)),
            GRB.MINIMIZE,
        )

        return m
