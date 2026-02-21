from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solver_kwargs = {}

    def set_timeout_to_zero(self, model) -> None:
        self._solver_kwargs["TimeLimit"] = 0.0

    def solve(self, model):
        self._solver_kwargs["OutputFlag"] = 1
        model.solve(self.solver, **self._solver_kwargs)

    def _get_objective(self, model) -> float:
        return model.objective.value
