"""Example Pyoframe formulation of a facility location problem.

Inspired from the original JuMP paper.
"""

import os
import sys


def solve_model(use_var_names=False, G=4, F=3):
    sys.path.append(os.getcwd())
    from benchmarks.problem_facility_location.bm_pyoframe import solve

    return solve(
        G=G, F=F, solver=None, is_benchmarking=False, use_var_names=use_var_names
    )


def _draw_results(model):
    import tkinter

    root = tkinter.Tk()
    scale = 500
    padding = 20
    canvas = tkinter.Canvas(root, width=scale + 2 * padding, height=scale + 2 * padding)
    point_size = 5
    max_dist = model.max_distance.solution
    G = len(model.x_axis)
    for x in range(G):
        for y in range(G):
            canvas.create_rectangle(
                x / (G - 1) * scale - point_size + padding,
                y / (G - 1) * scale - point_size + padding,
                x / (G - 1) * scale + point_size + padding,
                y / (G - 1) * scale + point_size + padding,
                fill="black",
            )
    for f, x, y in model.facility_position.solution.pivot(
        on="d", values="solution", index="f"
    ).iter_rows():
        canvas.create_rectangle(
            x * scale - point_size + padding,
            y * scale - point_size + padding,
            x * scale + point_size + padding,
            y * scale + point_size + padding,
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
    model = solve_model()
    _draw_results(model)
