import pyoptinterface as poi

from benchmark_utils import BaseBenchmark


class Benchmark(BaseBenchmark):
    def build(self, **kwargs):
        if self.solver == "gurobi":
            import pyoptinterface.gurobi as gurobi

            return gurobi.Model()
        elif self.solver == "copt":
            import pyoptinterface.copt as copt

            return copt.Model()
        elif self.solver == "highs":
            import pyoptinterface.highs as highs

            return highs.Model()
        elif self.solver == "mosek":
            import pyoptinterface.mosek as mosek

            return mosek.Model()
        else:
            raise ValueError(f"Unknown solver {self.solver}")

    def set_timeout_to_zero(self, model) -> None:
        model.set_model_attribute(poi.ModelAttribute.TimeLimitSec, 0.0)

    def solve(self, model):
        model.optimize()

    def _get_objective(self, model) -> float:
        return model.get_model_attribute(poi.ModelAttribute.ObjectiveValue)
