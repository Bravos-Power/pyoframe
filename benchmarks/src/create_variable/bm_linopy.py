"""Linopy dummy benchmark."""

import linopy
from benchmark_utils import LinopyBenchmark
from numpy import arange


class Bench(LinopyBenchmark):
    def build(self):
        N = self.size
        m = linopy.Model()
        x = m.add_variables(coords=[arange(N)])
        m.add_objective(x.sum())
        return m
