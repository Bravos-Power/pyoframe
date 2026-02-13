from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def __init__(self, *args, use_var_names=False, **kwargs):
        import pyoframe as pf

        super().__init__(*args, **kwargs)
        self.use_var_names = use_var_names
        pf.Config.default_solver = self.solver

        # if self.block_solver:
        # Slightly improves performance
        # The bottleneck is still the underlying library
        # pf.Config.maintain_order = False
        # pf.Config.disable_unmatched_checks = True

    def set_timeout_to_zero(self, model):
        model.attr.TimeLimitSec = 0

    def solve(self, model):
        model.optimize()
        return model

    def _get_objective(self, model) -> float:
        return model.objective.value
