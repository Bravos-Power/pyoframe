import pulp

from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.solver == "gurobi":
            self._solver_instance = pulp.GUROBI()
        else:
            raise NotImplementedError(
                f"Solver {self.solver} not implemented in PulpBenchmark."
            )

    def set_timeout_to_zero(self, model) -> None:
        self._solver_instance.timeLimit = 0.0

    def solve(self, model: pulp.LpProblem):
        model.solve(self._solver_instance)

    def _get_objective(self, model) -> float:
        return model.objective.value()
