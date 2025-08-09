"""Linopy dummy benchmark."""

from benchmark_utils import LinopyBenchmark


class Bench(LinopyBenchmark):
    def build(self):
        raise NotImplementedError()
