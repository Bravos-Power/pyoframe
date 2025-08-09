"""Linopy dummy benchmark."""

from benchmarks.utils import LinopyBenchmark


class Bench(LinopyBenchmark):
    def build(self):
        raise NotImplementedError()
