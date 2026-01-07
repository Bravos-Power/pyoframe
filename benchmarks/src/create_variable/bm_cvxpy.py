"""Cvxpy implementation of the facility location benchmark.

Note: we must use the SOC constraint because CVXPY does not support convex <= convex constraints.
See: https://www.cvxpy.org/tutorial/dcp/index.html#dcp-problems
"""

import cvxpy
from benchmark_utils import CvxpyBenchmark


class Bench(CvxpyBenchmark):
    def build(self):
        N = self.size
        x = cvxpy.Variable((N,), pos=True)
        objective = cvxpy.Minimize(cvxpy.sum(x))
        return cvxpy.Problem(objective)
