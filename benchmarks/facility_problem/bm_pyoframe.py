# pyright: reportAttributeAccessIssue=false
import polars as pl

from pyoframe import Model, Set, Variable


def solve(solver, size=None, G=None, F=None, is_benchmarking=True, use_var_names=False):
    assert G is None or F is None or size is None, (
        "When G or F are provided, size must be None."
    )
    if G is None or F is None:
        assert size is not None, "Either G and F or size must be provided."
        G = F = size

    grid = range(G)
    customer_position_x = pl.DataFrame(
        {"x": grid, "x_pos": [step / (G - 1) for step in grid]}
    )
    customer_position_y = pl.DataFrame(
        {"y": grid, "y_pos": [step / (G - 1) for step in grid]}
    )

    model = Model(solver=solver, use_var_names=use_var_names)
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
    model.con_dist_x = model.dist_x == customer_position_x.to_expr().add_dim(
        "f"
    ) - model.facility_position.pick(d=1).add_dim("x")
    model.con_dist_y = model.dist_y == customer_position_y.to_expr().add_dim(
        "f"
    ) - model.facility_position.pick(d=2).add_dim("y")
    model.dist = Variable(model.x_axis, model.y_axis, model.facilities, lb=0)
    model.con_dist = model.dist**2 == (model.dist_x**2).add_dim("y") + (
        model.dist_y**2
    ).add_dim("x")

    # Twice the max distance which ensures that when is_closest is 0, the constraint is not binding.
    M = 2 * 1.414
    model.con_max_distance = model.max_distance.add_dim(
        "x", "y", "f"
    ) >= model.dist - M * (1 - model.is_closest)

    model.minimize = model.max_distance

    if is_benchmarking:
        model.attr.TimeLimitSec = 0
        model.attr.Silent = 0

    model.optimize()

    return model


if __name__ == "__main__":
    solve("gurobi", size=5)
