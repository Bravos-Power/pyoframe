from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def set_timeout_to_zero(self, model):
        model.setParam("TimeLimit", 0.0)

    def solve(self, model):
        model.optimize()

    def _get_objective(self, model) -> float:
        return model.getObjective().getValue()
