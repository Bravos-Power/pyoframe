from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_timeout_to_zero(self, model) -> None:
        self.solver_args["TimeLimit"] = 0.0

    def solve(self, model):
        model.solve(self.solver, **self.solver_args)

    def _get_objective(self, model) -> float:
        return model.objective.value
