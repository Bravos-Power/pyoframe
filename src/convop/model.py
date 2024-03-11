from typing import Any, List
from convop.model_element import ModelElement
from convop.constraints import Constraint
from convop.objective import Objective
from convop.variables import Variable
from convop.io import to_file
from convop.solvers import solve


class Model:
    def __init__(self, name="model"):
        self._variables: List[Variable] = []
        self._constraints: List[Constraint] = []
        self._objective: Objective | None = None
        self.name = name

    @property
    def variables(self):
        return self._variables

    @property
    def constraints(self):
        return self._constraints

    @property
    def objective(self):
        return self._objective

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, ModelElement) and not __name.startswith("_"):
            assert not hasattr(
                self, __name
            ), f"Cannot create {__name} since it was already created."

            __value.name = __name

            if isinstance(__value, Objective):
                assert self.objective is None, "Cannot create more than one objective."
                self._objective = __value
            if isinstance(__value, Variable):
                self._variables.append(__value)
            elif isinstance(__value, Constraint):
                self._constraints.append(__value)

        return super().__setattr__(__name, __value)

    to_file = to_file
    solve = solve
