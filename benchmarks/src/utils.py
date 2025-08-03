from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

from pyoframe._constants import SUPPORTED_SOLVERS


class Benchmark(ABC):
    def __init__(self, solver, size=None, block_solver=True):
        assert solver in self.get_supported_solvers(), (
            f"{solver} is not supported by {self.__class__.__name__}."
        )
        self.solver = solver
        self.size: int | Tuple[int, int] | None = size
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
    def __init__(self, *args, use_var_names=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_var_names = use_var_names

        if self.block_solver:
            # Improve performance since we're not debugging
            import pyoframe as pf

            pf.Config.print_uses_variable_names = False
            pf.Config.maintain_order = False
            pf.Config.disable_unmatched_checks = True

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


class LinopyBenchmark(Benchmark):
    def solve(self, model):
        kwargs = {}

        if self.block_solver:
            kwargs["OutputFlag"] = 0
            kwargs["TimeLimit"] = 0.0
            kwargs["Presolve"] = 0

        model.solve(self.solver, **kwargs)
        return model


def mock_snakemake(rulename, **wildcards):
    """
    `mock_snakemake` is inspired from PyPSA-Eur (MIT license, see https://github.com/PyPSA/pypsa-eur/blob/master/scripts/_helpers.py#L476)

    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    """
    import os

    import snakemake as sm
    from snakemake.api import Workflow
    from snakemake.common import SNAKEFILE_CHOICES
    from snakemake.script import Snakemake
    from snakemake.settings.types import (
        ConfigSettings,
        DAGSettings,
        ResourceSettings,
        StorageSettings,
        WorkflowSettings,
    )

    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent
    current_dir = os.getcwd()
    os.chdir(root_dir)

    try:
        for p in SNAKEFILE_CHOICES:
            p = root_dir / p
            if os.path.exists(p):
                snakefile = p
                break
        else:
            raise FileNotFoundError(
                f"Could not find a Snakefile in {root_dir} or its subdirectories."
            )
        workflow = Workflow(
            ConfigSettings(),
            ResourceSettings(),
            WorkflowSettings(),
            StorageSettings(),
            DAGSettings(rerun_triggers=[]),
            storage_provider_settings=dict(),
        )
        workflow.include(snakefile)
        workflow.global_resources = {}
        rule = workflow.get_rule(rulename)
        dag = sm.dag.DAG(workflow, rules=[rule])
        job = sm.jobs.Job(rule, dag, wildcards)

        def make_accessable(*ios):
            for io in ios:
                for i, _ in enumerate(io):
                    io[i] = os.path.abspath(io[i])

        make_accessable(job.input, job.output, job.log)
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            job.dag.workflow.config,
            job.rule.name,
            None,
        )
        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    finally:
        os.chdir(current_dir)

    snakemake.mock = True
    return snakemake
