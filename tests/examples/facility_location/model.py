"""Example Pyoframe formulation of a facility location problem.

Inspired from the original JuMP paper.
"""

import pyoframe as pf


def solve_model(use_var_names=False, G=4, F=3):
    model = pf.Model(solver_uses_variable_names=use_var_names, sense="min")

    g_range = range(G)
    model.facilities = pf.Set(f=range(F))
    model.x_axis = pf.Set(x=g_range)
    model.y_axis = pf.Set(y=g_range)
    model.axis = pf.Set(d=[1, 2])
    model.customers = model.x_axis * model.y_axis

    model.facility_position = pf.Variable(model.facilities, model.axis, lb=0, ub=1)
    model.customer_position_x = pf.Param(
        {"x": g_range, "x_pos": [step / (G - 1) for step in g_range]}
    )
    model.customer_position_y = pf.Param(
        {"y": g_range, "y_pos": [step / (G - 1) for step in g_range]}
    )

    model.max_distance = pf.Variable(lb=0)

    model.is_closest = pf.Variable(model.customers, model.facilities, vtype="binary")
    model.con_only_one_closest = model.is_closest.sum("f") == 1

    model.dist_x = pf.Variable(model.x_axis, model.facilities)
    model.dist_y = pf.Variable(model.y_axis, model.facilities)
    model.con_dist_x = model.dist_x == model.customer_position_x.over(
        "f"
    ) - model.facility_position.pick(d=1).over("x")
    model.con_dist_y = model.dist_y == model.customer_position_y.over(
        "f"
    ) - model.facility_position.pick(d=2).over("y")
    model.dist = pf.Variable(model.x_axis, model.y_axis, model.facilities, lb=0)
    # expressed as non-convex quadratic constraint for Gurobi
    # other solvers could handle if it were expressed in cone form
    model.con_dist = model.dist**2 == (model.dist_x**2).over("y") + (
        model.dist_y**2
    ).over("x")

    M = (
        2 * 1.414
    )  # Twice the max distance which ensures that when is_closest is 0, the constraint is not binding.
    model.con_max_distance = model.max_distance.over(
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
    model = solve_model(G=G, F=F)
    draw_results(model, G, F)
