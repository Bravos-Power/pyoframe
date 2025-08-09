"""Pyoptinterface dummy benchmark."""

from benchmarks.utils import PyOptInterfaceBenchmark


class Bench(PyOptInterfaceBenchmark):
    def build(self):
        raise NotImplementedError()
