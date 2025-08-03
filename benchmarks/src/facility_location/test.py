"""Tests for the facility location benchmark."""

import pytest

from .bm_cvxpy import Bench as CvxpyBenchmark
from .bm_gurobipy import Bench as GurobiBenchmark
from .bm_pyoframe import Bench as PyoframeBenchmark
from .bm_pyomo import Bench as PyomoBenchmark
from .bm_pyoptinterface import Bench as PyOptInterfaceBenchmark


@pytest.mark.parametrize(
    "bench_cls",
    [
        CvxpyBenchmark,
        GurobiBenchmark,
        PyOptInterfaceBenchmark,
        PyoframeBenchmark,
        PyomoBenchmark,
    ],
    ids=lambda cls: cls.__bases__[0].__name__,
)
@pytest.mark.parametrize(
    "size",
    [
        ((4, 3), 0.4999993524133264),
        (3, 0.4999993524133264),
        (2, 0.5590167218381955),
        ((3, 4), 0.23570226039551578),
    ],
    ids=lambda s: str(s[0]),
)
def test_benchmark_objectives(bench_cls, size):
    size, expected_obj = size
    bench = bench_cls("gurobi", size, block_solver=False)
    bench.run()
    objective_value = bench.get_objective()
    assert objective_value == pytest.approx(expected_obj, abs=1e-5)
