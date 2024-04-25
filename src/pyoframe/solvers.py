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
    Result,
    Solution,
    Status,
)
import contextlib

from pathlib import Path

if TYPE_CHECKING:
    from pyoframe.model import Model


def solve(m: "Model", solver, **kwargs):
    if solver == "gurobi":
        result = GurobiSolver().solve(m, **kwargs)
    else:
        raise ValueError(f"Solver {solver} not recognized or supported.")

    # TODO load in results to model

    if result.solution is not None:
        for variable in m.variables:
            variable.value = result.solution.primal

        if result.solution.dual is not None:
            for constraint in m.constraints:
                constraint.dual = result.solution.dual

    return result


class Solver(ABC):
    @abstractmethod
    def solve(self, model, directory: Optional[Path] = None, **kwargs) -> Result: ...


class FileBasedSolver(Solver):
    def solve(
        self,
        model: "Model",
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
            filename = model.name if model.name is not None else "pyoframe-problem"
            problem_file = directory / f"{filename}.lp"
        problem_file = model.to_file(problem_file, use_var_names=use_var_names)
        assert model.io_mappers is not None

        results = self.solve_from_lp(problem_file, **kwargs)

        if results.solution is not None:
            results.solution.primal = model.io_mappers.var_map.undo(
                results.solution.primal
            )
            if results.solution.dual is not None:
                results.solution.dual = model.io_mappers.const_map.undo(
                    results.solution.dual
                )

        return results

    @abstractmethod
    def solve_from_lp(self, problem_file: Path, **kwargs) -> Result: ...


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

            m = gurobipy.read(path_to_str(problem_fn), env=env)
            if solver_options is not None:
                for key, value in solver_options.items():
                    m.setParam(key, value)
            if log_fn is not None:
                m.setParam("logfile", path_to_str(log_fn))
            if warmstart_fn:
                m.read(path_to_str(warmstart_fn))

            m.optimize()

            if basis_fn:
                try:
                    m.write(path_to_str(basis_fn))
                except gurobipy.GurobiError as err:
                    print("No model basis stored. Raised error: %s", err)

            condition = m.status
            termination_condition = CONDITION_MAP.get(condition, condition)
            status = Status.from_termination_condition(termination_condition)

            if status.is_ok:
                if solution_file:
                    m.write(path_to_str(solution_file))

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
                except gurobipy.GurobiError as e:
                    print("Dual values couldn't be parsed")
                    dual = None

                solution = Solution(sol, dual, objective)
            else:
                solution = None

        return Result(status, solution, m)


def path_to_str(path: Union[Path, str]) -> str:
    """
    Convert a pathlib.Path to a string.
    """
    return str(path.resolve()) if isinstance(path, Path) else path
