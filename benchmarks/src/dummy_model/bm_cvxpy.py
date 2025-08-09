"""Cvxpy implementation of the facility location benchmark.

Note: we must use the SOC constraint because CVXPY does not support convex <= convex constraints.
See: https://www.cvxpy.org/tutorial/dcp/index.html#dcp-problems
"""

import cvxpy
import numpy as np
from benchmark_utils import CvxpyBenchmark


class Bench(CvxpyBenchmark):
    def build(self):
        N = self.size
        x = cvxpy.Variable((N, N))
        y = cvxpy.Variable((N, N))
        constraints = []
        for i in range(N):
            constraints.append(x[:, i] - y[:, i] >= np.arange(N))
        constraints.append(x + y >= 0)
        objective = cvxpy.Minimize(2 * cvxpy.sum(x) + cvxpy.sum(y))
        return cvxpy.Problem(objective, constraints)
