# Copyright (c) 2023: Yue Yang
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import os
import time
import pyoframe as pf
import polars as pl


def solve_facility(solver, G, F):
    model = pf.Model("min")

    g_range = range(G)
    model.facilities = pf.Set(f=range(F))
    model.x_axis = pf.Set(x=g_range)
    model.y_axis = pf.Set(y=g_range)
    model.axis = pf.Set(d=[1, 2])
    model.customers = model.x_axis * model.y_axis

    model.facility_position = pf.Variable(model.facilities, model.axis, lb=0, ub=1)
    model.customer_position_x = pl.DataFrame(
        {"x": g_range, "x_pos": [step / (G - 1) for step in g_range]}
    ).to_expr()
    model.customer_position_y = pl.DataFrame(
        {"y": g_range, "y_pos": [step / (G - 1) for step in g_range]}
    ).to_expr()

    model.max_distance = pf.Variable(lb=0)

    model.is_closest = pf.Variable(model.customers, model.facilities, vtype="binary")
    model.con_only_one_closest = pf.sum("f", model.is_closest) == 1

    model.dist_x = pf.Variable(model.x_axis, model.facilities)
    model.dist_y = pf.Variable(model.y_axis, model.facilities)
    model.con_dist_x = model.dist_x == model.customer_position_x.add_dim(
        "f"
    ) - model.facility_position.select(d=1).add_dim("x")
    model.con_dist_y = model.dist_y == model.customer_position_y.add_dim(
        "f"
    ) - model.facility_position.select(d=2).add_dim("y")
    model.dist = pf.Variable(model.x_axis, model.y_axis, model.facilities, lb=0)
    model.con_dist = model.dist**2 == (model.dist_x**2).add_dim("y") + (
        model.dist_y**2
    ).add_dim("x")

    M = (
        2 * 1.414
    )  # Twice the max distance which ensures that when is_closest is 0, the constraint is not binding.
    model.con_max_distance = model.max_distance.add_dim(
        "x", "y", "f"
    ) >= model.dist - M * (1 - model.is_closest)

    model.objective = model.max_distance
    model.params.TimeLimit = 0
    model.params.Presolve = 0

    model.solve(log_to_console=False)


def main(Ns=[25, 50, 75, 100]):
    dir = os.path.realpath(os.path.dirname(__file__))

    for solver in ["poi-gurobi", "gurobi"]:
        for n in Ns:
            start = time.time()
            solve_facility(solver, n, n)
            run_time = round(time.time() - start, 1)
            content = f"pyoframe-{solver} fac-{n} -1 {run_time}"
            print(content)
            with open(dir + "/benchmarks.csv", "a") as io:
                io.write(f"{content}\n")
    return

if __name__ == "__main__":
    main()
