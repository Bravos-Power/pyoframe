"""
Code to interface with various solvers
"""

from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import polars as pl

from pyoframe.constants import (
    DUAL_KEY,
    NAME_COL,
    SOLUTION_KEY,
    SLACK_COL,
    RC_COL,
    Result,
    Solution,
    Status,
)
import contextlib

from pathlib import Path

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe.model import Model


def solve(m: "Model", solver, **kwargs):
    if solver == "gurobi":
        m.solver = GurobiSolver(m)
    else:
        raise ValueError(f"Solver {solver} not recognized or supported.")

    result = m.solver.solve(**kwargs)
    m.solver_model = result.solver_model

    if result.solution is not None:
        m.objective.value = result.solution.objective

        for variable in m.variables:
            variable.solution = result.solution.primal

        if result.solution.dual is not None:
            for constraint in m.constraints:
                constraint.dual = result.solution.dual

    return result


class Solver(ABC):
    def __init__(self, model):
        self._model = model

    @abstractmethod
    def solve(self, directory: Optional[Path] = None, **kwargs) -> Result: ...

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


class FileBasedSolver(Solver):
    def solve(
        self,
        directory: Optional[Union[Path, str]] = None,
        use_var_names=None,
        **kwargs,
    ) -> Result:
        problem_file = None
        if directory is not None:
            if isinstance(directory, str):
                directory = Path(directory)
            if not directory.exists():
                directory.mkdir(parents=True)
            filename = (
                self._model.name if self._model.name is not None else "pyoframe-problem"
            )
            problem_file = directory / f"{filename}.lp"
        problem_file = self._model.to_file(problem_file, use_var_names=use_var_names)
        assert self._model.io_mappers is not None

        results = self.solve_from_lp(problem_file, **kwargs)

        if results.solution is not None:
            results.solution.primal = self._model.io_mappers.var_map.undo(
                results.solution.primal
            )
            if results.solution.dual is not None:
                results.solution.dual = self._model.io_mappers.const_map.undo(
                    results.solution.dual
                )

        return results

    @abstractmethod
    def solve_from_lp(self, problem_file: Path, **kwargs) -> Result: ...

    def _get_all_rc(self):
        return self._model.io_mappers.var_map.undo(self._get_all_rc_unmapped())

    def _get_all_slack(self):
        return self._model.io_mappers.const_map.undo(self._get_all_slack_unmapped())

    @abstractmethod
    def _get_all_rc_unmapped(self): ...

    @abstractmethod
    def _get_all_slack_unmapped(self): ...


class GurobiSolver(FileBasedSolver):
    def solve_from_lp(
        self,
        problem_fn,
        log_fn=None,
        warmstart_fn=None,
        basis_fn=None,
        solution_file=None,
        env=None,
        **solver_options,
    ) -> Result:
        """
        Solve a linear problem using the gurobi solver.

        This function communicates with gurobi using the gurubipy package.
        """
        import gurobipy

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

        with contextlib.ExitStack() as stack:
            if env is None:
                env = stack.enter_context(gurobipy.Env())

            m = gurobipy.read(_path_to_str(problem_fn), env=env)
            if solver_options is not None:
                for key, value in solver_options.items():
                    m.setParam(key, value)
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
            termination_condition = CONDITION_MAP.get(condition, condition)
            status = Status.from_termination_condition(termination_condition)

            if status.is_ok:
                if solution_file:
                    m.write(_path_to_str(solution_file))

                objective = m.ObjVal
                vars = m.getVars()
                sol = pl.DataFrame(
                    {
                        NAME_COL: m.getAttr("VarName", vars),
                        SOLUTION_KEY: m.getAttr("X", vars),
                    }
                )

                constraints = m.getConstrs()
                try:
                    dual = pl.DataFrame(
                        {
                            DUAL_KEY: m.getAttr("Pi", constraints),
                            NAME_COL: m.getAttr("ConstrName", constraints),
                        }
                    )
                except gurobipy.GurobiError:
                    dual = None

                solution = Solution(sol, dual, objective)
            else:
                solution = None

        return Result(status, solution, m)

    def _get_all_rc_unmapped(self):
        m = self._model.solver_model
        vars = m.getVars()
        return pl.DataFrame(
            {
                RC_COL: m.getAttr("RC", vars),
                NAME_COL: m.getAttr("VarName", vars),
            }
        )

    def _get_all_slack_unmapped(self):
        m = self._model.solver_model
        constraints = m.getConstrs()
        return pl.DataFrame(
            {
                SLACK_COL: m.getAttr("Slack", constraints),
                NAME_COL: m.getAttr("ConstrName", constraints),
            }
        )


def _path_to_str(path: Union[Path, str]) -> str:
    """
    Convert a pathlib.Path to a string.
    """
    return str(path.resolve()) if isinstance(path, Path) else path
