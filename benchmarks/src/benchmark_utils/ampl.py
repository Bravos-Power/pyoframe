from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def set_timeout_to_zero(self, model) -> None:
        model.setOption("timelimit", 0.0)

    def solve(self, model):
        model.option["solver"] = self.solver
        model.setOption("gurobi_options", "outlev=1")
        model.solve()

    def _get_objective(self, model) -> float:
        return model.get_objective().totalcost.get().value()
