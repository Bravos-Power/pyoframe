"""Pyoframe implementation of the facility location benchmark."""

from benchmark_utils.pyoframe import Benchmark

from pyoframe import Model, Param, Set, Variable


class Bench(Benchmark):
    def build(self, **kwargs):
        G = F = self.size
        assert G is not None

        grid = range(G + 1)

        customer_position_x = Param({"x": grid, "x_pos": [step / G for step in grid]})
        customer_position_y = Param({"y": grid, "y_pos": [step / G for step in grid]})

        model = Model()
        model.facilities = Set(f=range(1, F + 1))
        model.grid = Set(x=grid, y=grid)
        model.axis = Set(d=[1, 2])

        model.facility_position = Variable(model.facilities, model.axis, lb=0, ub=1)

        model.max_distance = Variable(lb=0)

        model.is_closest = Variable(model.grid, model.facilities, vtype="binary")
        model.con_only_one_closest = model.is_closest.sum("f") == 1
        model.dist_xy = Variable(model.grid, model.facilities, model.axis)
        model.con_dist_x = model.dist_xy.pick(d=1) == (
            customer_position_x.over("f") - model.facility_position.pick(d=1).over("x")
        ).over("y")
        model.con_dist_y = model.dist_xy.pick(d=2) == (
            customer_position_y.over("f") - model.facility_position.pick(d=2).over("y")
        ).over("x")
        model.dist = Variable(model.grid, model.facilities, lb=0)
        model.con_dist = (
            model.dist**2 == model.dist_xy.pick(d=1) ** 2 + model.dist_xy.pick(d=2) ** 2
        )

        # Twice the max distance which ensures that when is_closest is 0, the constraint is not binding.
        M = 2 * 1.414
        model.con_max_distance = model.max_distance.over(
            "x", "y", "f"
        ) >= model.dist - M * (1 - model.is_closest)

        model.minimize = model.max_distance

        return model


if __name__ == "__main__":
    Bench("gurobi", size=2).run()
