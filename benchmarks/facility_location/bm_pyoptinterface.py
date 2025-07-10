# Copyright (c) 2023: Yue Yang
# https://github.com/metab0t/PyOptInterface_benchmark
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import numpy as np
import pyoptinterface as poi

from benchmarks.util import PyOptInterfaceBenchmark


class Benchmark(PyOptInterfaceBenchmark):
    def build(self):
        try:
            G, F = self.size
        except TypeError:
            G = F = self.size

        m = self.create_model()
        # Create variables
        y = add_ndarray_variable(m, (F, 2), lb=0.0, ub=1.0)
        s = add_ndarray_variable(m, (G + 1, G + 1, F), lb=0.0)
        z = add_ndarray_variable(m, (G + 1, G + 1, F), domain=poi.VariableDomain.Binary)
        r = add_ndarray_variable(m, (G + 1, G + 1, F, 2))
        d = m.add_variable()

        # Set objective
        m.set_objective(d * 1.0)

        # Add constraints
        for i in range(G + 1):
            for j in range(G + 1):
                expr = poi.quicksum(z[i, j, :])
                m.add_linear_constraint(expr, poi.Eq, 1.0)

        M = 2 * 1.414
        for i in range(G + 1):
            for j in range(G + 1):
                for f in range(F):
                    expr = s[i, j, f] - d - M * (1 - z[i, j, f])
                    m.add_linear_constraint(expr, poi.Eq, 0.0)
                    expr = r[i, j, f, 0] - i / G + y[f, 0]
                    m.add_linear_constraint(expr, poi.Eq, 0.0)
                    expr = r[i, j, f, 1] - j / G + y[f, 1]
                    m.add_linear_constraint(expr, poi.Eq, 0.0)
                    m.add_second_order_cone_constraint(
                        [s[i, j, f], r[i, j, f, 0], r[i, j, f, 1]]
                    )
        return m


def add_ndarray_variable(m, shape, **kwargs):
    array = np.empty(shape, dtype=object)
    array_flat = array.flat
    for i in range(array.size):
        array_flat[i] = m.add_variable(**kwargs)
    return array


if __name__ == "__main__":
    Benchmark("gurobi", 5).run()
