import os
from pathlib import Path

import polars as pl

import pyoframe as pf


def main(G=4, F=3, solver=None, **kwargs):
    model = pf.Model(solver=solver, sense="min")

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
    ) - model.facility_position.pick(d=1).add_dim("x")
    model.con_dist_y = model.dist_y == model.customer_position_y.add_dim(
        "f"
    ) - model.facility_position.pick(d=2).add_dim("y")
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

    model.optimize()
    return model


def draw_results(model, G, F):
    import tkinter

    root = tkinter.Tk()
    scale = 500
    padding = 20
    canvas = tkinter.Canvas(root, width=scale + 2 * padding, height=scale + 2 * padding)
    size = 5
    max_dist = model.max_distance.solution
    for x in range(G):
        for y in range(G):
            canvas.create_rectangle(
                x / (G - 1) * scale - size + padding,
                y / (G - 1) * scale - size + padding,
                x / (G - 1) * scale + size + padding,
                y / (G - 1) * scale + size + padding,
                fill="black",
            )
    for f, x, y in model.facility_position.solution.pivot(
        on="d", values="solution", index="f"
    ).iter_rows():
        canvas.create_rectangle(
            x * scale - size + padding,
            y * scale - size + padding,
            x * scale + size + padding,
            y * scale + size + padding,
            fill="red",
        )
        canvas.create_oval(
            (x - max_dist) * scale + padding,
            (y - max_dist) * scale + padding,
            (x + max_dist) * scale + padding,
            (y + max_dist) * scale + padding,
            outline="red",
        )
    canvas.pack()
    root.mainloop()


if __name__ == "__main__":
    G, F = 4, 3
    working_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    model = main(
        G,
        F,
        solver="gurobi",
        directory=working_dir / "results",
        use_var_names=True,
        solution_file=working_dir / "results" / "pyoframe-problem.sol",
    )
    draw_results(model, G, F)
