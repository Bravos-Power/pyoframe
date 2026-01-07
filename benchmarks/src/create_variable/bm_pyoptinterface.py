"""Pyoptinterface dummy benchmark."""

import pyoptinterface as poi
from benchmark_utils import PyOptInterfaceBenchmark


class Bench(PyOptInterfaceBenchmark):
    def build(self):
        N = self.size
        m = self.create_model()
        x = m.add_variables(range(N))
        expr = poi.ExprBuilder()
        poi.quicksum_(expr, x)
        m.set_objective(expr)
        return m
