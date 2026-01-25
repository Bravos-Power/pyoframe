from pyomo.environ import value
from pyomo.opt import SolverFactory

from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solver_factory = SolverFactory(self.solver)

    def set_timeout_to_zero(self, model) -> None:
        self._solver_factory.options["timelimit"] = 0.0

    def solve(self, model):
        self._solver_factory.solve(model, tee=True)

    def _get_objective(self, model) -> float:
        return value(model.obj)
