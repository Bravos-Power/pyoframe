"""Pyoptinterface dummy benchmark."""

from benchmark_utils import PyOptInterfaceBenchmark


class Bench(PyOptInterfaceBenchmark):
    def build(self):
        raise NotImplementedError()
