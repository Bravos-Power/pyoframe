"""Cvxpy implementation of the facility location benchmark.

Note: we must use the SOC constraint because CVXPY does not support convex <= convex constraints.
See: https://www.cvxpy.org/tutorial/dcp/index.html#dcp-problems
"""

from benchmarks.utils import CvxpyBenchmark


class Bench(CvxpyBenchmark):
    def build(self):
        raise NotImplementedError()
