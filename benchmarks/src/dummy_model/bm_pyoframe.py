"""Pyoframe implementation of the facility location benchmark."""

from benchmark_utils import PyoframeBenchmark

from pyoframe import Model, Set, Variable, sum


class Bench(PyoframeBenchmark):
    def build(self):
        N = self.size
        m = Model()
        ibyj = Set(i=range(N), j=range(N))
        m.x = Variable(ibyj)
        m.y = Variable(ibyj)
        m.con1 = m.x - m.y >= Set(i=range(N)).to_expr().add_dim("j")
        m.con2 = m.x + m.y >= 0
        m.minimize = sum(2 * m.x) + sum(m.y)
        return m


if __name__ == "__main__":
    Bench("gurobi", 5).run()
