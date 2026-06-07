"""Pyoframe implementation of the facility location benchmark."""

from benchmark_utils.pyoframe import Benchmark

from pyoframe import Model, Param, Set, Variable


class Bench(Benchmark):
    def build(self, **kwargs):
        G = F = self.size

        grid = range(G + 1)

        customer_position_x = Param({"x": grid, "x_pos": [step / G for step in grid]})
        customer_position_y = Param({"y": grid, "y_pos": [step / G for step in grid]})

        m = Model()
        m.facilities = Set(f=range(1, F + 1))
        m.grid = Set(x=grid, y=grid)
        m.axis = Set(d=[1, 2])

        m.facility_position = Variable(m.facilities, m.axis, lb=0, ub=1)

        m.max_dist = Variable(lb=0)

        m.is_closest = Variable(m.grid, m.facilities, vtype="binary")
        m.con_only_one_closest = m.is_closest.sum("f") == 1
        m.dist_xy = Variable(m.grid, m.facilities, m.axis)
        m.con_dist_x = m.dist_xy.pick(d=1) == (
            customer_position_x.over("f") - m.facility_position.pick(d=1).over("x")
        ).over("y")
        m.con_dist_y = m.dist_xy.pick(d=2) == (
            customer_position_y.over("f") - m.facility_position.pick(d=2).over("y")
        ).over("x")
        m.dist = Variable(m.grid, m.facilities, lb=0)
        m.con_dist = m.dist**2 == m.dist_xy.pick(d=1) ** 2 + m.dist_xy.pick(d=2) ** 2

        # Twice the max distance which ensures that when is_closest is 0, the constraint is not binding.
        M = 2 * 1.414
        m.con_max_dist = m.max_dist.over("x", "y", "f") >= m.dist - M * (
            1 - m.is_closest
        )

        m.minimize = m.max_dist

        return m


if __name__ == "__main__":
    Bench("gurobi", size=2).run()
