from abc import ABC, abstractmethod

from pyoframe.constants import SUPPORTED_SOLVERS


class Benchmark(ABC):
    def __init__(self, solver, size, block_solver=True):
        assert solver in self.get_supported_solvers(), (
            f"{solver} is not supported by {self.__class__.__name__}."
        )
        self.solver = solver
        self.size = size
        self.block_solver = block_solver

    @abstractmethod
    def build(self): ...

    @abstractmethod
    def solve(self, model): ...

    def run(self):
        model = self.build()
        return self.solve(model)

    def get_supported_solvers(self):
        return [s.name for s in SUPPORTED_SOLVERS]


class PyoframeBenchmark(Benchmark):
    def __init__(self, solver, size, use_var_names=False):
        super().__init__(solver, size)
        self.use_var_names = use_var_names

    def solve(self, model):
        if self.block_solver:
            model.attr.TimeLimitSec = 0
            model.attr.Silent = 0
        model.optimize()
        return model


class PyomoBenchmark(Benchmark):
    def solve(self, model):
        from pyomo.opt import SolverFactory

        opt = SolverFactory(self.solver)
        if self.block_solver:
            opt.options["timelimit"] = 0.0
            opt.options["presolve"] = False
        try:
            if self.solver == "gurobi_persistent":
                opt.set_instance(model)
                opt.solve(tee=True)
            else:
                opt.solve(model, tee=True)
        except ValueError as e:
            if self.block_solver and "bad status: aborted" in str(e):
                pass
            else:
                raise e
        return model


class GurobiPyBenchmark(Benchmark):
    def solve(self, model):
        if self.block_solver:
            model.setParam("OutputFlag", 0)
            model.setParam("TimeLimit", 0.0)
            model.setParam("Presolve", 0)
        model.optimize()
        return model

    def get_supported_solvers(self):
        return ["gurobi"]


class PyOptInterfaceBenchmark(Benchmark):
    def solve(self, model):
        import pyoptinterface as poi

        if self.block_solver:
            model.set_model_attribute(poi.ModelAttribute.Silent, True)
            model.set_model_attribute(poi.ModelAttribute.TimeLimitSec, 0.0)
            solver_name = model.get_model_attribute(poi.ModelAttribute.SolverName)
            if solver_name.lower() == "gurobi":
                model.set_raw_parameter("Presolve", 0)
            elif solver_name.lower() == "copt":
                model.set_raw_parameter("Presolve", 1)
        model.optimize()
        return model

    def create_model(self):
        from pyoptinterface import copt, gurobi, highs, mosek

        model_constructor = {
            "gurobi": gurobi.Model,
            "copt": copt.Model,
            "highs": highs.Model,
            "mosek": mosek.Model,
        }.get(self.solver, None)
        if model_constructor is None:
            raise ValueError(f"Unknown solver {self.solver}")

        return model_constructor()
