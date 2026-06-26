from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def set_timeout_to_zero(self, model):
        model.setParam("TimeLimit", 0.0)

    def solve(self, model):
        if self.solver_args is not None:
            for key, value in self.solver_args.items():
                model.setParam(key, value)
        model.optimize()

    def _get_objective(self, model) -> float:
        return model.getObjective().getValue()
