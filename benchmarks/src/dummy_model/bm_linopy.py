"""Linopy dummy benchmark."""

import linopy
from benchmark_utils import LinopyBenchmark
from numpy import arange


class Bench(LinopyBenchmark):
    def build(self):
        N = self.size
        m = linopy.Model()
        x = m.add_variables(coords=[arange(N), arange(N)])
        y = m.add_variables(coords=[arange(N), arange(N)])
        m.add_constraints(x - y >= arange(N))
        m.add_constraints(x + y >= 0)
        m.add_objective((2 * x).sum() + y.sum())
        return m
