"""Cvxpy implementation of the facility location benchmark.

Note: we must use the SOC constraint because CVXPY does not support convex <= convex constraints.
See: https://www.cvxpy.org/tutorial/dcp/index.html#dcp-problems
"""

import cvxpy as cp
import numpy as np
from benchmark_utils.cvxpy import CvxpyBenchmark


class Bench(CvxpyBenchmark):
    def build(self):
        if isinstance(self.size, int):
            G = F = self.size
        else:
            G, F = self.size

        G = G + 1  # Add one to match Julia
        M = 2 * np.sqrt(2)

        # Facility locations: F x 2
        facility_loc = cp.Variable((F, 2), name="y", bounds=[0, 1])
        # Binary assignments: (G+1) x (G+1) x F
        is_nearest = cp.Variable((G, G, F), boolean=True, name="z")
        # Max distance
        max_dist = cp.Variable(name="d")
        dist = cp.Variable((G, G, F), name="dist", nonneg=True)
        dist_x = cp.Variable((G, F), name="dist_x")
        dist_y = cp.Variable((G, F), name="dist_y")

        # Constraint 1: assignment sums to 1 over facilities
        constraints = [cp.sum(is_nearest, axis=2) == 1]

        # Constraint 2: distance calculations
        grid_coords = (np.arange(G) / (G - 1))[:, None]
        constraints += [
            grid_coords - facility_loc[:, 0] == dist_x,
            grid_coords - facility_loc[:, 1] == dist_y,
        ]

        dist_x_with_y = cp.broadcast_to(dist_x[:, None, :], (G, G, F))
        dist_y_with_x = cp.broadcast_to(dist_y[None, :, :], (G, G, F))
        dist_xy = cp.hstack(
            [
                cp.vec(dist_x_with_y, order="C")[:, None],
                cp.vec(dist_y_with_x, order="C")[:, None],
            ]
        )
        constraints += [cp.norm2(dist_xy, axis=1) <= cp.vec(dist, order="C")]

        # Constraint 3: max distance constraint
        constraints += [max_dist + M * (1 - is_nearest) >= dist]

        # Objective: minimize max distance
        objective = cp.Minimize(max_dist)

        problem = cp.Problem(objective, constraints)
        # assert problem.is_dcp()
        return problem


if __name__ == "__main__":
    bench = Bench("gurobi", (4, 3), block_solver=False)
    bench.run()
    print(bench.get_objective())
    ...
