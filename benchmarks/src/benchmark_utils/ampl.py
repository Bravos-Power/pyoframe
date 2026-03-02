from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_timeout_to_zero = False

    def set_timeout_to_zero(self, model) -> None:
        self._set_timeout_to_zero = True

    def _get_objective(self, model) -> float:
        return model.get_objective("obj").value()

    def solve(self, model):
        model.option["solver"] = self.solver
        default_options = "outlev=1 " + self._convert_solver_args_to_gurobi_options()
        if self._set_timeout_to_zero:
            model.setOption("gurobi_options", default_options + " timelimit=0")
        else:
            model.setOption("gurobi_options", default_options)
        model.solve()

    def _convert_solver_args_to_gurobi_options(self):
        if self.solver_args is None:
            return ""
        options = []
        for key, value in self.solver_args.items():
            if isinstance(value, bool):
                value = int(value)
            options.append(f"{key}={value}")
        return " ".join(options)
