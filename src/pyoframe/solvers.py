"""
Code to interface with various solvers
"""

from abc import abstractmethod, ABC
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, TYPE_CHECKING

import polars as pl

from pyoframe.constants import (
    DUAL_KEY,
    SOLUTION_KEY,
    SLACK_COL,
    RC_COL,
    VAR_KEY,
    CONSTRAINT_KEY,
    Result,
    Solution,
    Status,
)
import contextlib
import pyoframe as pf

from pathlib import Path

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Model

available_solvers = []
solver_registry: Dict[str, Type["Solver"]] = {}

with contextlib.suppress(ImportError):
    import gurobipy

    available_solvers.append("gurobi")


def _register_solver(solver_name):
    def decorator(cls):
        solver_registry[solver_name] = cls
        return cls

    return decorator


def solve(
    m: "Model",
    solver=None,
    directory: Optional[Union[Path, str]] = None,
    use_var_names=False,
    log_fn=None,
    warmstart_fn=None,
    basis_fn=None,
    solution_file=None,
    log_to_console=True,
):
    if solver is None:
        if len(available_solvers) == 0:
            raise ValueError(
                "No solvers available. Please install a solving library like gurobipy."
            )
        solver = available_solvers[0]

    if solver not in solver_registry:
        raise ValueError(f"Solver {solver} not recognized or supported.")

    solver_cls = solver_registry[solver]
    m.solver = solver_cls(
        m,
        log_to_console,
        params={param: value for param, value in m.params},
        directory=directory,
    )
    m.solver_model = m.solver.create_solver_model(use_var_names)
    m.solver.solver_model = m.solver_model

    for attr_container in [m.variables, m.constraints, [m]]:
        for container in attr_container:
            for param_name, param_value in container.attr:
                m.solver.set_attr(container, param_name, param_value)

    result = m.solver.solve(log_fn, warmstart_fn, basis_fn, solution_file)
    result = m.solver.process_result(result)
    m.result = result

    if result.solution is not None:
        if m.objective is not None:
            m.objective.value = result.solution.objective

        for variable in m.variables:
            variable.solution = result.solution.primal

        if result.solution.dual is not None:
            for constraint in m.constraints:
                constraint.dual = result.solution.dual

    return result


class Solver(ABC):
    def __init__(self, model: "Model", log_to_console, params, directory):
        self._model = model
        self.solver_model: Optional[Any] = None
        self.log_to_console: bool = log_to_console
        self.params = params
        self.directory = directory

    @abstractmethod
    def create_solver_model(self, use_var_names) -> Any: ...

    @abstractmethod
    def set_attr(self, element, param_name, param_value): ...

    @abstractmethod
    def solve(self, log_fn, warmstart_fn, basis_fn, solution_file) -> Result: ...

    @abstractmethod
    def process_result(self, results: Result) -> Result: ...

    def load_rc(self):
        rc = self._get_all_rc()
        for variable in self._model.variables:
            variable.RC = rc

    def load_slack(self):
        slack = self._get_all_slack()
        for constraint in self._model.constraints:
            constraint.slack = slack

    @abstractmethod
    def _get_all_rc(self): ...

    @abstractmethod
    def _get_all_slack(self): ...

    def dispose(self):
        """
        Clean up any resources that wouldn't be cleaned up by the garbage collector.

        For now, this is only used by the Gurobi solver to call .dispose() on the solver model and Gurobi environment
        which helps close a connection to the Gurobi Computer Server. Note that this effectively disables commands that
        need access to the solver model (like .slack and .RC)
        """


class FileBasedSolver(Solver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.problem_file: Optional[Path] = None
        self.keep_files = self.directory is not None

    def create_solver_model(self, use_var_names) -> Any:
        problem_file = None
        directory = self.directory
        if directory is not None:
            if isinstance(directory, str):
                directory = Path(directory)
            if not directory.exists():
                directory.mkdir(parents=True)
            filename = (
                self._model.name if self._model.name is not None else "pyoframe-problem"
            )
            problem_file = directory / f"{filename}.lp"
        self.problem_file = self._model.to_file(
            problem_file, use_var_names=use_var_names
        )
        assert self._model.io_mappers is not None
        return self.create_solver_model_from_lp()

    @abstractmethod
    def create_solver_model_from_lp(self) -> Any: ...

    def set_attr(self, element, param_name, param_value):
        if isinstance(param_value, pl.DataFrame):
            if isinstance(element, pf.Variable):
                param_value = self._model.io_mappers.var_map.apply(param_value)
            elif isinstance(element, pf.Constraint):
                param_value = self._model.io_mappers.const_map.apply(param_value)
        return self.set_attr_unmapped(element, param_name, param_value)

    @abstractmethod
    def set_attr_unmapped(self, element, param_name, param_value): ...

    def process_result(self, results: Result) -> Result:
        if results.solution is not None:
            results.solution.primal = self._model.io_mappers.var_map.undo(
                results.solution.primal
            )
            if results.solution.dual is not None:
                results.solution.dual = self._model.io_mappers.const_map.undo(
                    results.solution.dual
                )

        return results

    def _get_all_rc(self):
        return self._model.io_mappers.var_map.undo(self._get_all_rc_unmapped())

    def _get_all_slack(self):
        return self._model.io_mappers.const_map.undo(self._get_all_slack_unmapped())

    @abstractmethod
    def _get_all_rc_unmapped(self): ...

    @abstractmethod
    def _get_all_slack_unmapped(self): ...


@_register_solver("gurobi")
class GurobiSolver(FileBasedSolver):
    # see https://www.gurobi.com/documentation/10.0/refman/optimization_status_codes.html
    CONDITION_MAP = {
        1: "unknown",
        2: "optimal",
        3: "infeasible",
        4: "infeasible_or_unbounded",
        5: "unbounded",
        6: "other",
        7: "iteration_limit",
        8: "terminated_by_limit",
        9: "time_limit",
        10: "optimal",
        11: "user_interrupt",
        12: "other",
        13: "suboptimal",
        14: "unknown",
        15: "terminated_by_limit",
        16: "internal_solver_error",
        17: "internal_solver_error",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.log_to_console:
            self.params["LogToConsole"] = 0
        self.env = None

    def create_solver_model_from_lp(self) -> Any:
        """
        Solve a linear problem using the gurobi solver.

        This function communicates with gurobi using the gurubipy package.
        """
        assert self.problem_file is not None
        self.env = gurobipy.Env(params=self.params)

        m = gurobipy.read(_path_to_str(self.problem_file), env=self.env)
        if not self.keep_files:
            self.problem_file.unlink()

        return m

    @lru_cache
    def _get_var_mapping(self):
        assert self.solver_model is not None
        vars = self.solver_model.getVars()
        return vars, pl.DataFrame(
            {VAR_KEY: self.solver_model.getAttr("VarName", vars)}
        ).with_columns(i=pl.int_range(pl.len()))

    @lru_cache
    def _get_constraint_mapping(self):
        assert self.solver_model is not None
        constraints = self.solver_model.getConstrs()
        return constraints, pl.DataFrame(
            {CONSTRAINT_KEY: self.solver_model.getAttr("ConstrName", constraints)}
        ).with_columns(i=pl.int_range(pl.len()))

    def set_attr_unmapped(self, element, param_name, param_value):
        assert self.solver_model is not None
        if isinstance(element, pf.Model):
            self.solver_model.setAttr(param_name, param_value)
        elif isinstance(element, pf.Variable):
            v, v_map = self._get_var_mapping()
            param_value = param_value.join(v_map, on=VAR_KEY, how="left").drop(VAR_KEY)
            self.solver_model.setAttr(
                param_name,
                [v[i] for i in param_value["i"]],
                param_value[param_name],
            )
        elif isinstance(element, pf.Constraint):
            c, c_map = self._get_constraint_mapping()
            param_value = param_value.join(c_map, on=CONSTRAINT_KEY, how="left").drop(
                CONSTRAINT_KEY
            )
            self.solver_model.setAttr(
                param_name,
                [c[i] for i in param_value["i"]],
                param_value[param_name],
            )
        else:
            raise ValueError(f"Element type {type(element)} not recognized.")

    def solve(self, log_fn, warmstart_fn, basis_fn, solution_file) -> Result:
        assert self.solver_model is not None
        m = self.solver_model
        if log_fn is not None:
            m.setParam("logfile", _path_to_str(log_fn))
        if warmstart_fn:
            m.read(_path_to_str(warmstart_fn))

        m.optimize()

        if basis_fn:
            try:
                m.write(_path_to_str(basis_fn))
            except gurobipy.GurobiError as err:
                print("No model basis stored. Raised error: %s", err)

        condition = m.status
        termination_condition = GurobiSolver.CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)

        if status.is_ok and (termination_condition == "optimal"):
            if solution_file:
                m.write(_path_to_str(solution_file))

            objective = m.ObjVal
            vars = m.getVars()
            sol = pl.DataFrame(
                {
                    VAR_KEY: m.getAttr("VarName", vars),
                    SOLUTION_KEY: m.getAttr("X", vars),
                }
            )

            constraints = m.getConstrs()
            try:
                dual = pl.DataFrame(
                    {
                        DUAL_KEY: m.getAttr("Pi", constraints),
                        CONSTRAINT_KEY: m.getAttr("ConstrName", constraints),
                    }
                )
            except gurobipy.GurobiError:
                dual = None

            solution = Solution(sol, dual, objective)
        else:
            solution = None

        return Result(status, solution)

    def _get_all_rc_unmapped(self):
        m = self._model.solver_model
        vars = m.getVars()
        return pl.DataFrame(
            {
                RC_COL: m.getAttr("RC", vars),
                VAR_KEY: m.getAttr("VarName", vars),
            }
        )

    def _get_all_slack_unmapped(self):
        m = self._model.solver_model
        constraints = m.getConstrs()
        return pl.DataFrame(
            {
                SLACK_COL: m.getAttr("Slack", constraints),
                CONSTRAINT_KEY: m.getAttr("ConstrName", constraints),
            }
        )

    def dispose(self):
        if self.solver_model is not None:
            self.solver_model.dispose()
        if self.env is not None:
            self.env.dispose()


def _path_to_str(path: Union[Path, str]) -> str:
    """
    Convert a pathlib.Path to a string.
    """
    return str(path.resolve()) if isinstance(path, Path) else path
