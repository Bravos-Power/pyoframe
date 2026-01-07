"""Pyoframe implementation of the facility location benchmark."""

from benchmark_utils import PyoframeBenchmark

from pyoframe import Model, Set, Variable


class Bench(PyoframeBenchmark):
    def build(self):
        m = Model()
        m.x = Variable(Set(i=range(self.size)))  # type: ignore
        m.minimize = m.x.sum()
        return m


if __name__ == "__main__":
    Bench("gurobi", 5).run()
