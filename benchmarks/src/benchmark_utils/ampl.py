from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_timeout_to_zero = False

    def set_timeout_to_zero(self, model) -> None:
        self._set_timeout_to_zero = True

    def solve(self, model):
        model.option["solver"] = self.solver
        if self._set_timeout_to_zero:
            model.setOption("gurobi_options", "timelimit=0 outlev=1")
        else:
            model.setOption("gurobi_options", "outlev=1")
        model.solve()

    def _get_objective(self, model) -> float:
        return model.get_objective("obj").value()
