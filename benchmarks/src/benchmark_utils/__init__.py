"""Contains the base classes used for benchmarking.

Note that this contributes to every benchmark so we try to keep the imports mostly clear.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cvxpy as cp


class Benchmark(ABC):
    def __init__(self, solver, size=None, block_solver=True, input_dir=None):
        self.solver = solver
        self.size = size
        self.block_solver = block_solver
        self.input_dir = Path(input_dir) if input_dir else None

    @abstractmethod
    def build(self) -> Any: ...

    @abstractmethod
    def solve(self, model) -> Any: ...

    def get_objective(self) -> float:
        assert not self.block_solver, (
            "Cannot get objective value when block_solver is True."
        )
        return self._get_objective()

    @abstractmethod
    def _get_objective(self) -> float: ...

    def run(self):
        self.model = self.build()
        self.model = self.solve(self.model)
        return self.model


class PyoframeBenchmark(Benchmark):
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

    def solve(self, model):
        if self.block_solver:
            model.attr.TimeLimitSec = 0
        model.optimize()
        return model

    def _get_objective(self) -> float:
        return self.model.objective.value


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

    def _get_objective(self) -> float:
        from pyomo.environ import value

        return value(self.model.obj)


class GurobiPyBenchmark(Benchmark):
    def solve(self, model):
        if self.block_solver:
            model.setParam("OutputFlag", 0)
            model.setParam("TimeLimit", 0.0)
            model.setParam("Presolve", 0)
        model.optimize()
        return model

    def _get_objective(self) -> float:
        return self.model.getObjective().getValue()


class PyOptInterfaceBenchmark(Benchmark):
    def solve(self, model):
        import pyoptinterface as poi

        if self.block_solver:
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

    def _get_objective(self) -> float:
        import pyoptinterface as poi

        return self.model.get_model_attribute(poi.ModelAttribute.ObjectiveValue)


class LinopyBenchmark(Benchmark):
    def solve(self, model):
        kwargs = {}

        if self.block_solver:
            kwargs["OutputFlag"] = 0
            kwargs["TimeLimit"] = 0.0
            kwargs["Presolve"] = 0

        try:
            model.solve(self.solver, **kwargs)
        except Exception as e:
            if self.block_solver:
                pass
            else:
                raise e
        return model

    def _get_objective(self) -> float:
        return self.model.objective.value


class CvxpyBenchmark(Benchmark):
    def solve(self, model: cp.Problem):
        kwargs = {}

        if self.block_solver:
            kwargs["OutputFlag"] = 0
            kwargs["TimeLimit"] = 0.0
            kwargs["Presolve"] = 0

        model.solve(self.solver, **kwargs)
        return model

    def _get_objective(self) -> float:
        return self.model.objective.value


def mock_snakemake(rulename, **wildcards):
    """Returns a snakemake object to substitute the one available when running from snakemake.

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
    root_dir = script_dir.parent.parent
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


def run_notebook(
    notebook_path: Path,
    working_directory: Path,
    debug: bool = False,
    first_cell: str | None = None,
):
    """Runs a Jupyter notebook."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    if first_cell is not None:
        injected_cell = nbformat.v4.new_code_cell(source=first_cell)

        nb["cells"] = [injected_cell] + nb["cells"]
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": working_directory}})

    if debug:
        with open(notebook_path.parent / ".debug.ipynb", "w") as f:
            nbformat.write(nb, f)
