"""Pyoptinterface dummy benchmark."""

import pyoptinterface as poi
from benchmark_utils.pyoptinterface import PyOptInterfaceBenchmark


class Bench(PyOptInterfaceBenchmark):
    def build(self):
        N = self.size
        m = self.create_model()
        x = m.add_variables(range(N), range(N))
        y = m.add_variables(range(N), range(N))
        for i in range(N):
            for j in range(N):
                m.add_linear_constraint(x[i, j] - y[i, j], poi.Geq, i)
                m.add_linear_constraint(x[i, j] + y[i, j], poi.Geq, 0)
        expr = poi.ExprBuilder()
        poi.quicksum_(expr, x, lambda x: 2 * x)
        poi.quicksum_(expr, y)
        m.set_objective(expr)
        return m
