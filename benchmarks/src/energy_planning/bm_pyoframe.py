"""Pyoframe implementation of the facility location benchmark."""

from benchmarks.utils import PyoframeBenchmark


class Bench(PyoframeBenchmark):
    def build(self):
        raise NotImplementedError()


if __name__ == "__main__":
    Bench("gurobi", 5).run()
