import pyoframe as pf
import polars as pl
import os
from pathlib import Path


def main(F, G, **kwargs):
    model = pf.Model("min")

    model.facilities = pf.Set(f=range(F))
    model.customers = pf.Set(c=range(G))

    model.facility_position_x = pf.Variable(model.facilities, lb=0, ub=1)
    model.facility_position_y = pf.Variable(model.facilities, lb=0, ub=1)
    model.customer_position_x = pl.DataFrame(
        {"c": range(G), "x": [step / G for step in range(G)]}
    ).to_expr()
    model.customer_position_y = pl.DataFrame(
        {"c": range(G), "y": [step / G for step in range(G)]}
    ).to_expr()

    model.max_distance = pf.Variable(lb=0)
    model.objective = model.max_distance

    model.is_closest = pf.Variable(model.customers, model.facilities, vtype="binary")
    model.con_only_one_closest = pf.sum("f", model.is_closest) == 1

    model.dist_x = pf.Variable(model.customers, model.facilities)
    model.dist_y = pf.Variable(model.customers, model.facilities)
    model.con_dist_x = model.dist_x == model.customer_position_x.add_dim(
        "f"
    ) - model.facility_position_x.add_dim("c")
    model.con_dist_y = model.dist_y == model.customer_position_y.add_dim(
        "f"
    ) - model.facility_position_y.add_dim("c")
    model.dist = pf.Variable(model.customers, model.facilities, lb=0)
    model.con_dist = model.dist**2 >= model.dist_x**2 + model.dist_y**2

    M = 2 * 1.414  # Twice the max distance which ensures that when is_closest is 0, the constraint is not binding.
    model.con_max_distance = model.max_distance.add_dim("c", "f") == model.dist - M * (
        1 - model.is_closest
    )

    model.solve(**kwargs)


if __name__ == "__main__":
    working_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    main(
        3,
        4,
        directory=working_dir / "results",
        use_var_names=True,
        solution_file=working_dir / "results" / "solution.sol",
    )
