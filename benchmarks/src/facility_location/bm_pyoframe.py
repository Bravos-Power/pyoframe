"""Pyoframe implementation of the facility location benchmark."""

import polars as pl
from benchmark_utils import PyoframeBenchmark

from pyoframe import Model, Set, Variable


class Bench(PyoframeBenchmark):
    def build(self):
        G = F = self.size
        G = G + 1  # Add one to match Julia

        grid = range(G)
        customer_position_x = pl.DataFrame(
            {"x": grid, "x_pos": [step / (G - 1) for step in grid]}
        ).to_expr()
        customer_position_y = pl.DataFrame(
            {"y": grid, "y_pos": [step / (G - 1) for step in grid]}
        ).to_expr()

        model = Model(
            solver=self.solver,
            solver_uses_variable_names=self.use_var_names,
            print_uses_variable_names=False,
        )
        model.facilities = Set(f=range(F))
        model.x_axis = Set(x=grid)
        model.y_axis = Set(y=grid)
        model.axis = Set(d=[1, 2])

        model.facility_position = Variable(model.facilities, model.axis, lb=0, ub=1)

        model.max_distance = Variable(lb=0)

        model.is_closest = Variable(
            model.x_axis, model.y_axis, model.facilities, vtype="binary"
        )
        model.con_only_one_closest = model.is_closest.sum("f") == 1
        model.dist_x = Variable(model.x_axis, model.facilities)
        model.dist_y = Variable(model.y_axis, model.facilities)
        model.con_dist_x = model.dist_x == customer_position_x.over(
            "f"
        ) - model.facility_position.pick(d=1).over("x")
        model.con_dist_y = model.dist_y == customer_position_y.over(
            "f"
        ) - model.facility_position.pick(d=2).over("y")
        model.dist = Variable(model.x_axis, model.y_axis, model.facilities, lb=0)
        model.con_dist = model.dist**2 == (model.dist_x**2).over("y") + (
            model.dist_y**2
        ).over("x")

        # Twice the max distance which ensures that when is_closest is 0, the constraint is not binding.
        M = 2 * 1.414
        model.con_max_distance = model.max_distance.over(
            "x", "y", "f"
        ) >= model.dist - M * (1 - model.is_closest)

        model.minimize = model.max_distance

        return model


if __name__ == "__main__":
    Bench("gurobi", 5).run()
