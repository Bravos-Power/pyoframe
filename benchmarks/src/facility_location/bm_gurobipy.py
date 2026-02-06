"""GurobiPy implementation of the facility location benchmark.

Copyright (c) 2023: Yue Yang

Use of this source code is governed by an MIT-style license that can be found
in the LICENSE.md file or at https://opensource.org/licenses/MIT.
"""

from benchmark_utils.gurobipy import Benchmark
from gurobipy import GRB, Model


class Bench(Benchmark):
    def build(self):
        if isinstance(self.size, int):
            G = F = self.size
        else:
            G, F = self.size

        m = Model()

        # Create variables
        y = m.addVars(range(1, F + 1), range(1, 3), ub=1.0, vtype=GRB.CONTINUOUS)
        s = m.addVars(range(G + 1), range(G + 1), range(1, F + 1))
        z = m.addVars(range(G + 1), range(G + 1), range(1, F + 1), vtype=GRB.BINARY)
        r = m.addVars(
            range(G + 1),
            range(G + 1),
            range(1, F + 1),
            range(1, 3),
            name="r",
            lb=-GRB.INFINITY,
        )
        d = m.addVar()

        # Set objective
        m.setObjective(d, GRB.MINIMIZE)

        # Add constraints
        for i in range(G + 1):
            for j in range(G + 1):
                m.addConstr(z.sum(i, j, "*") == 1)

        M = 2 * 1.414
        for i in range(G + 1):
            for j in range(G + 1):
                for f in range(1, F + 1):
                    m.addConstr(s[i, j, f] - (M * (1 - z[i, j, f])) <= d)
                    m.addConstr(r[i, j, f, 1] == -y[f, 1] + (1.0 * i) / G)
                    m.addConstr(r[i, j, f, 2] == -y[f, 2] + (1.0 * j) / G)
                    m.addConstr(
                        r[i, j, f, 1] * r[i, j, f, 1] + r[i, j, f, 2] * r[i, j, f, 2]
                        <= s[i, j, f] * s[i, j, f]
                    )

        return m


if __name__ == "__main__":
    bench = Bench("gurobi", 2, block_solver=False)
    bench.run()
    print(bench.get_objective())
